import time

import numpy as np

from evaluate import evaluate_analogies
from math_utils import log_sigmoid, sigmoid
from pairs import generate_pairs_afws, generate_pairs_vectorized, generate_pairs_weighted


def train_batch(W_in, W_out, centers, pos_ids, neg_ids, lr):
    """
    Vectorized SGNS forward-backward-update for one mini-batch.

    Hybrid approach: vectorized dot products and center gradient via einsum,
    then loop over K for negative updates (avoids large outer-product
    allocation while keeping scattered writes small and cache-friendly).

    Uses regular fancy indexing for updates (not np.add.at) for speed.
    Duplicate indices lose some gradient accumulation, which is acceptable
    for SGD — equivalent to a slight per-word learning rate reduction.

    Loss per sample:
        L = -log sigmoid(v_c · v_o) - sigmoid_k log sigmoid(-v_c · v_{n_k})

    Gradients (per sample):
        ∂L/∂v_c     = (sigmoid(v_c·v_o) - 1)·v_o  + sigmoid_k sigmoid(v_c·v_{n_k})·v_{n_k}
        ∂L/∂v_o     = (sigmoid(v_c·v_o) - 1)·v_c
        ∂L/∂v_{n_k} = sigmoid(v_c·v_{n_k})·v_c

    Returns average loss per sample (scalar).
    """
    B = len(centers)
    K = neg_ids.shape[1]

    # Lookup (fancy indexing returns copies)
    vc = W_in[centers]       # (B, D)
    vo = W_out[pos_ids]      # (B, D)
    vn = W_out[neg_ids]      # (B, K, D)

    # Dot products (vectorized)
    pos_dot = np.sum(vc * vo, axis=1)           # (B,)
    neg_dot = np.einsum('bd,bkd->bk', vc, vn)  # (B, K)
    np.clip(pos_dot, -10, 10, out=pos_dot)
    np.clip(neg_dot, -10, 10, out=neg_dot)

    # Sigmoid & loss
    pos_sig = sigmoid(pos_dot)   # (B,)
    neg_sig = sigmoid(neg_dot)   # (B, K)
    loss = (-log_sigmoid(pos_dot).sum() - log_sigmoid(-neg_dot).sum()) / B

    # Gradient coefficients
    pos_coeff = pos_sig - 1.0    # (B,)

    # Center gradient (vectorized via einsum)
    grad_vc = (pos_coeff[:, None] * vo
               + np.einsum('bk,bkd->bd', neg_sig, vn))  # (B, D)

    # Apply updates (regular indexing for speed)
    W_in[centers]  -= lr * grad_vc
    W_out[pos_ids] -= lr * (pos_coeff[:, None] * vc)

    # Negative updates — loop over K to avoid B×K×D allocation
    for k in range(K):
        W_out[neg_ids[:, k]] -= lr * (neg_sig[:, k:k+1] * vc)

    return float(loss)


def train_batch_weighted(W_in, W_out, centers, pos_ids, neg_ids, weights, lr):
    """
    SGNS forward-backward-update with per-sample position weights.

    Identical to train_batch but each sample's gradient is scaled by
    weights[i] = 1/d where d is the distance from center to context.

    Loss per sample:
        L = w_i * [-log sigmoid(v_c · v_o) - Σ_k log sigmoid(-v_c · v_{n_k})]

    Gradients (per sample):
        ∂L/∂v_c     = w_i * [(σ(v_c·v_o) - 1)·v_o  + Σ_k σ(v_c·v_{n_k})·v_{n_k}]
        ∂L/∂v_o     = w_i * [(σ(v_c·v_o) - 1)·v_c]
        ∂L/∂v_{n_k} = w_i * [σ(v_c·v_{n_k})·v_c]

    Returns average weighted loss per sample.
    """
    B = len(centers)
    K = neg_ids.shape[1]
    w = weights  # (B,) — 1/d values

    vc = W_in[centers]       # (B, D)
    vo = W_out[pos_ids]      # (B, D)
    vn = W_out[neg_ids]      # (B, K, D)

    pos_dot = np.sum(vc * vo, axis=1)
    neg_dot = np.einsum('bd,bkd->bk', vc, vn)
    np.clip(pos_dot, -10, 10, out=pos_dot)
    np.clip(neg_dot, -10, 10, out=neg_dot)

    pos_sig = sigmoid(pos_dot)
    neg_sig = sigmoid(neg_dot)
    loss = (-log_sigmoid(pos_dot) * w).sum() / B - (log_sigmoid(-neg_dot) * w[:, None]).sum() / B

    pos_coeff = pos_sig - 1.0    # (B,)

    # Scale coefficients by weight
    w_pos_coeff = pos_coeff * w  # (B,)
    w_neg_sig = neg_sig * w[:, None]  # (B, K)

    grad_vc = (w_pos_coeff[:, None] * vo
               + np.einsum('bk,bkd->bd', w_neg_sig, vn))  # (B, D)

    W_in[centers]  -= lr * grad_vc
    W_out[pos_ids] -= lr * (w_pos_coeff[:, None] * vc)

    for k in range(K):
        W_out[neg_ids[:, k]] -= lr * (w_neg_sig[:, k:k+1] * vc)

    return float(loss)


def gradient_check_weighted(save_path=None):
    """
    Verify analytic gradients for position-weighted SGNS vs finite-difference.
    V=20, D=5, B=2, K=3.  All relative errors must be < 1e-5.
    """
    rng = np.random.default_rng(42)
    V, D, K = 20, 5, 3
    Wi = rng.standard_normal((V, D)) * 0.1
    Wo = rng.standard_normal((V, D)) * 0.1

    centers = np.array([3, 7])
    pos_ids = np.array([5, 12])
    neg_ids = np.array([[1, 8, 15], [2, 9, 18]])
    weights = np.array([1.0, 0.5])  # e.g. 1/d=1, 1/d=0.5
    B = 2

    def loss_fn(Wi, Wo):
        vc = Wi[centers]; vo = Wo[pos_ids]
        L = 0.0
        for b in range(B):
            L += weights[b] * (-log_sigmoid(np.dot(vc[b], vo[b])))
            for k in range(K):
                nk = Wo[neg_ids[b, k]]
                L += weights[b] * (-log_sigmoid(-np.dot(vc[b], nk)))
        return L / B

    # Analytic gradients
    vc = Wi[centers]; vo = Wo[pos_ids]
    pd = np.sum(vc * vo, axis=1)
    ps = sigmoid(pd); pc = ps - 1.0
    g_vc = (pc * weights)[:, None] * vo
    g_vo = (pc * weights)[:, None] * vc
    g_vn = np.zeros((B, K, D))
    for k in range(K):
        nk = Wo[neg_ids[:, k]]
        nd = np.sum(vc * nk, axis=1)
        ns = sigmoid(nd)
        g_vc += (ns * weights)[:, None] * nk
        g_vn[:, k, :] = (ns * weights)[:, None] * vc
    g_vc /= B; g_vo /= B; g_vn /= B

    eps = 1e-5
    lines = []
    def report(msg):
        print(msg); lines.append(msg)

    report("=" * 60)
    report("GRADIENT VERIFICATION — POSITION-WEIGHTED SGNS")
    report(f"  Model: V={V}, D={D}, K={K}, B={B}")
    report(f"  Weights: {weights}")
    report("=" * 60)

    def check(name, param, indices, analytic):
        me = 0.0
        for bi in range(len(indices)):
            for d in range(D):
                orig = param[indices[bi], d]
                param[indices[bi], d] = orig + eps
                lp = loss_fn(Wi, Wo)
                param[indices[bi], d] = orig - eps
                lm = loss_fn(Wi, Wo)
                param[indices[bi], d] = orig
                num = (lp - lm) / (2 * eps)
                ana = analytic[bi, d] if analytic.ndim == 2 else analytic[bi]
                me = max(me, abs(num - ana) / (max(abs(num), abs(ana)) + 1e-12))
        ok = me < 1e-5
        report(f"  {name:25s} max rel error: {me:.2e}  {'PASS' if ok else 'FAIL'}")
        return ok

    p1 = check("W_in  (center)",   Wi, centers, g_vc)
    p2 = check("W_out (positive)", Wo, pos_ids, g_vo)

    me3 = 0.0
    for k in range(K):
        for bi in range(B):
            for d in range(D):
                orig = Wo[neg_ids[bi, k], d]
                Wo[neg_ids[bi, k], d] = orig + eps
                lp = loss_fn(Wi, Wo)
                Wo[neg_ids[bi, k], d] = orig - eps
                lm = loss_fn(Wi, Wo)
                Wo[neg_ids[bi, k], d] = orig
                num = (lp - lm) / (2 * eps)
                ana = g_vn[bi, k, d]
                me3 = max(me3, abs(num - ana) / (max(abs(num), abs(ana)) + 1e-12))
    p3 = me3 < 1e-5
    report(f"  {'W_out (negative)':25s} max rel error: {me3:.2e}  {'PASS' if p3 else 'FAIL'}")

    ok = p1 and p2 and p3
    report(f"\n  OVERALL: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")

    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    return ok


def gradient_check(save_path=None):
    """
    Verify analytic gradients vs finite-difference on a tiny model.
    V=20, D=5, B=2, K=3.  All relative errors must be < 1e-5.
    """
    rng = np.random.default_rng(42)
    V, D, K = 20, 5, 3
    Wi = rng.standard_normal((V, D)) * 0.1
    Wo = rng.standard_normal((V, D)) * 0.1

    centers = np.array([3, 7])
    pos_ids = np.array([5, 12])
    neg_ids = np.array([[1, 8, 15], [2, 9, 18]])
    B = 2

    def loss_fn(Wi, Wo):
        vc = Wi[centers]; vo = Wo[pos_ids]
        L = -log_sigmoid(np.sum(vc * vo, axis=1)).sum()
        for k in range(K):
            nk = Wo[neg_ids[:, k]]
            L -= log_sigmoid(-np.sum(vc * nk, axis=1)).sum()
        return L / B

    # Analytic gradients
    vc = Wi[centers]; vo = Wo[pos_ids]
    pd = np.sum(vc * vo, axis=1)
    ps = sigmoid(pd); pc = ps - 1.0
    g_vc = pc[:, None] * vo
    g_vo = pc[:, None] * vc
    g_vn = np.zeros((B, K, D))
    for k in range(K):
        nk = Wo[neg_ids[:, k]]
        nd = np.sum(vc * nk, axis=1)
        ns = sigmoid(nd)
        g_vc += ns[:, None] * nk
        g_vn[:, k, :] = ns[:, None] * vc
    g_vc /= B; g_vo /= B; g_vn /= B

    eps = 1e-5
    lines = []
    def report(msg):
        print(msg); lines.append(msg)

    report("=" * 60)
    report("GRADIENT VERIFICATION (Finite-Difference Check)")
    report(f"  Model: V={V}, D={D}, K={K}, B={B}")
    report("=" * 60)

    def check(name, param, indices, analytic):
        me = 0.0
        for bi in range(len(indices)):
            for d in range(D):
                orig = param[indices[bi], d]
                param[indices[bi], d] = orig + eps
                lp = loss_fn(Wi, Wo)
                param[indices[bi], d] = orig - eps
                lm = loss_fn(Wi, Wo)
                param[indices[bi], d] = orig
                num = (lp - lm) / (2 * eps)
                ana = analytic[bi, d] if analytic.ndim == 2 else analytic[bi]
                me = max(me, abs(num - ana) / (max(abs(num), abs(ana)) + 1e-12))
        ok = me < 1e-5
        report(f"  {name:25s} max rel error: {me:.2e}  {'PASS' if ok else 'FAIL'}")
        return ok

    p1 = check("W_in  (center)",   Wi, centers, g_vc)
    p2 = check("W_out (positive)", Wo, pos_ids, g_vo)

    me3 = 0.0
    for k in range(K):
        for bi in range(B):
            for d in range(D):
                orig = Wo[neg_ids[bi, k], d]
                Wo[neg_ids[bi, k], d] = orig + eps
                lp = loss_fn(Wi, Wo)
                Wo[neg_ids[bi, k], d] = orig - eps
                lm = loss_fn(Wi, Wo)
                Wo[neg_ids[bi, k], d] = orig
                num = (lp - lm) / (2 * eps)
                ana = g_vn[bi, k, d]
                me3 = max(me3, abs(num - ana) / (max(abs(num), abs(ana)) + 1e-12))
    p3 = me3 < 1e-5
    report(f"  {'W_out (negative)':25s} max rel error: {me3:.2e}  {'PASS' if p3 else 'FAIL'}")

    ok = p1 and p2 and p3
    report(f"\n  OVERALL: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")

    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    return ok


def estimate_pairs(corpus_ids, keep_probs, window, rng, n=5):
    """Rough estimate of training pairs per epoch (used for ETA)."""
    cs = 500_000
    nc = len(corpus_ids) // cs
    if nc == 0:
        return 0
    total = 0
    for _ in range(min(n, nc)):
        s = rng.integers(0, max(1, len(corpus_ids) - cs))
        chunk = corpus_ids[s:s + cs]
        kept = (rng.random(len(chunk), dtype=np.float32) < keep_probs[chunk]).sum()
        for d in range(1, window + 1):
            total += 2 * max(0, kept - d) * (window - d + 1) / window
    return int(total / min(n, nc) * nc)


def train(cfg, corpus_ids, keep_probs, neg_table, freqs, vocab_size,
          word2idx, idx2word, categories=None, label="baseline"):
    """Main training loop.  Returns embeddings and diagnostics."""
    rng = np.random.default_rng(cfg.seed)

    # Initialise embeddings (float32)
    W_in  = ((rng.random((vocab_size, cfg.embed_dim), dtype=np.float32) - 0.5)
             / cfg.embed_dim)
    W_out = np.zeros((vocab_size, cfg.embed_dim), dtype=np.float32)

    freqs_f32 = freqs.astype(np.float32)

    est = estimate_pairs(corpus_ids, keep_probs, cfg.window_size, rng)
    total_est = est * cfg.epochs
    print(f"    Est. pairs/epoch: {est:,}   total: {total_est:,}")
    print(f"    Params: {2*vocab_size*cfg.embed_dim:,} "
          f"({2*vocab_size*cfg.embed_dim*4/1e6:.0f} MB float32)")

    loss_h, tok_h, lr_h = [], [], []
    processed = 0
    log_every = 2000          # batches
    eval_every_epoch = 5      # evaluate periodically

    nc = len(corpus_ids) // cfg.chunk_size
    starts = np.arange(nc) * cfg.chunk_size

    t0 = time.time()
    run_loss = 0.0
    nb = 0

    for epoch in range(cfg.epochs):
        rng.shuffle(starts)
        ep_loss = 0.0; ep_nb = 0

        for cs in starts:
            chunk = corpus_ids[cs:cs + cfg.chunk_size]
            keep = rng.random(len(chunk), dtype=np.float32) < keep_probs[chunk]
            filt = chunk[keep]
            if len(filt) < 2:
                continue

            use_pw = getattr(cfg, 'use_position_weights', False)

            if use_pw:
                pairs = generate_pairs_weighted(filt, cfg.window_size, rng)
            elif cfg.use_afws:
                pairs = generate_pairs_afws(
                    filt, freqs_f32, cfg.afws_max_window,
                    cfg.afws_min_window, cfg.afws_alpha, rng)
            else:
                pairs = generate_pairs_vectorized(filt, cfg.window_size, rng)

            if len(pairs) == 0:
                continue

            for bi in range(0, len(pairs), cfg.batch_size):
                batch = pairs[bi:bi + cfg.batch_size]
                if len(batch) < 2:
                    continue
                B = len(batch)
                c_ids = batch[:, 0]
                p_ids = batch[:, 1]
                n_ids = neg_table[
                    rng.integers(0, len(neg_table), size=(B, cfg.neg_samples))]

                progress = min(processed / max(total_est, 1), 1.0)
                lr = max(cfg.lr_start * (1.0 - progress), cfg.lr_min)

                if use_pw:
                    dists = batch[:, 2].astype(np.float32)
                    pw_power = getattr(cfg, 'position_weight_power', 1.0)
                    w = 1.0 / np.power(dists, pw_power)  # 1/d^p
                    loss = train_batch_weighted(
                        W_in, W_out, c_ids, p_ids, n_ids, w, lr)
                else:
                    loss = train_batch(W_in, W_out, c_ids, p_ids, n_ids, lr)

                processed += B
                run_loss += loss
                nb += 1; ep_nb += 1; ep_loss += loss

                if nb % log_every == 0:
                    avg = run_loss / log_every
                    ela = time.time() - t0
                    pps = processed / ela
                    eta = (total_est - processed) / pps if pps > 0 else 0
                    print(f"    [{label}] E{epoch+1}/{cfg.epochs} "
                          f"B{nb:,} L={avg:.4f} lr={lr:.5f} "
                          f"{pps/1e6:.2f}M/s ETA {eta/60:.0f}m")
                    loss_h.append(avg); tok_h.append(processed); lr_h.append(lr)
                    run_loss = 0.0

        ep_avg = ep_loss / max(ep_nb, 1)
        print(f"    [{label}] Epoch {epoch+1} done — loss {ep_avg:.4f}")

        if categories and (epoch + 1) % eval_every_epoch == 0:
            _, s = evaluate_analogies(W_in, word2idx, idx2word, categories)
            print(f"    [{label}] Eval@E{epoch+1}: "
                  f"sem={s['semantic']['accuracy']*100:.1f}% "
                  f"syn={s['syntactic']['accuracy']*100:.1f}% "
                  f"all={s['overall']['accuracy']*100:.1f}%")

    elapsed = time.time() - t0
    print(f"    [{label}] Done in {elapsed:.0f}s ({processed:,} pairs)")
    return W_in, W_out, loss_h, tok_h, lr_h, elapsed
