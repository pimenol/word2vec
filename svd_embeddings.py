"""SVD-based embeddings from explicit PPMI matrix (Levy & Goldberg 2014).

Memory-efficient implementation for systems with limited RAM:
- Uses reduced vocabulary (top N words) for co-occurrence matrix
- In-place SPPMI computation
- Randomized truncated SVD (Halko et al. 2011)
- Extends SVD embeddings to full vocabulary via SGNS fallback

All operations use pure NumPy.
"""

import time

import numpy as np


def build_cooccurrence_matrix(corpus_ids, vocab_size, window,
                              weighted=True, max_vocab=None):
    """
    Build a dense co-occurrence matrix from the corpus.

    Args:
        corpus_ids: array of word integer IDs
        vocab_size: total vocabulary size
        window: context window size
        weighted: if True, weight by 1/d
        max_vocab: if set, only track top max_vocab words (by ID, which is
                   sorted by frequency). Words with ID >= max_vocab are ignored.

    Returns:
        C: (V_eff, V_eff) float32 co-occurrence matrix (symmetric)
        V_eff: effective vocabulary size used
    """
    V_eff = min(vocab_size, max_vocab) if max_vocab else vocab_size
    print(f"    Building co-occurrence matrix (V_eff={V_eff}, "
          f"window={window}, weighted={weighted})...")
    t0 = time.time()
    n = len(corpus_ids)
    C = np.zeros((V_eff, V_eff), dtype=np.float32)

    chunk_size = 200_000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = corpus_ids[start:end]
        for d in range(1, window + 1):
            if d >= len(chunk):
                break
            centers = chunk[:len(chunk) - d]
            contexts = chunk[d:]
            # Only keep pairs where both words are in reduced vocab
            mask = (centers < V_eff) & (contexts < V_eff)
            c_valid = centers[mask]
            x_valid = contexts[mask]
            weight = np.float32(1.0 / d if weighted else 1.0)
            np.add.at(C, (c_valid, x_valid), weight)
            np.add.at(C, (x_valid, c_valid), weight)

        if (start // chunk_size) % 20 == 0 and start > 0:
            elapsed = time.time() - t0
            pct = start / n * 100
            print(f"      {pct:.0f}% ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    nnz = np.count_nonzero(C)
    mem_mb = C.nbytes / 1e6
    print(f"    Done. {nnz:,} non-zero entries, {mem_mb:.0f} MB ({elapsed:.0f}s)")
    return C, V_eff


def compute_sppmi_inplace(C, neg_k=15):
    """
    Compute Shifted Positive PMI from co-occurrence matrix IN-PLACE.
    Overwrites C with SPPMI to save memory.

    SPPMI[i,j] = max(PMI[i,j] - log(k), 0)
    """
    print(f"    Computing SPPMI in-place (k={neg_k})...")
    t0 = time.time()

    D = float(C.sum())
    if D == 0:
        return C

    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)

    log_D = np.log(D)
    log_shift = np.log(float(neg_k))
    log_row = np.log(row_sums.astype(np.float64))
    log_col = np.log(col_sums.astype(np.float64))

    V = C.shape[0]
    chunk = 2000
    for i in range(0, V, chunk):
        end = min(i + chunk, V)
        block = C[i:end]  # view into C
        nz_rows, nz_cols = np.nonzero(block)
        if len(nz_rows) == 0:
            block[:] = 0
            continue
        vals = block[nz_rows, nz_cols].astype(np.float64)
        pmi_vals = np.log(vals) + log_D - log_row[i + nz_rows] - log_col[nz_cols]
        pmi_vals = np.maximum(pmi_vals - log_shift, 0.0)
        block[:] = 0
        block[nz_rows, nz_cols] = pmi_vals.astype(np.float32)

    elapsed = time.time() - t0
    nnz = np.count_nonzero(C)
    print(f"    Done. {nnz:,} non-zero SPPMI entries ({elapsed:.0f}s)")
    return C


def randomized_svd(M, dim=300, n_oversamples=20, n_power_iter=2):
    """
    Randomized truncated SVD (Halko, Martinsson, Tropp 2011).

    Returns: U (V, dim), S (dim,)
    """
    print(f"    Randomized SVD (dim={dim}, oversamples={n_oversamples}, "
          f"power_iter={n_power_iter})...")
    t0 = time.time()
    V = M.shape[0]
    k = dim + n_oversamples

    rng = np.random.default_rng(42)
    Omega = rng.standard_normal((V, k)).astype(np.float32)
    Y = M @ Omega

    for i in range(n_power_iter):
        print(f"      Power iteration {i+1}/{n_power_iter}...")
        Y, _ = np.linalg.qr(Y)
        Z = M.T @ Y
        Z, _ = np.linalg.qr(Z)
        Y = M @ Z

    Q, _ = np.linalg.qr(Y)
    del Y
    B = Q.T @ M  # (k, V)

    U_b, S, _ = np.linalg.svd(B, full_matrices=False)
    del B
    U = Q @ U_b
    del Q, U_b

    U = U[:, :dim]
    S = S[:dim]

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.0f}s). U: {U.shape}, "
          f"S range: [{S[-1]:.2f}, {S[0]:.2f}]")
    return U, S


def svd_embeddings(M, dim=300, power=0.5):
    """
    Compute embeddings from SPPMI matrix using randomized SVD.

    Embeddings = U * Sigma^power

    Args:
        M: (V_eff, V_eff) SPPMI matrix
        dim: embedding dimension
        power: exponent for singular values

    Returns:
        W: (V_eff, dim) embedding matrix
    """
    U, S = randomized_svd(M, dim=dim)
    S = np.maximum(S, 0.0)
    W = U * np.power(S + 1e-10, power)[None, :]
    print(f"    Embedding shape: {W.shape}, power={power}")
    return W.astype(np.float32)


def extend_svd_to_full_vocab(W_svd, W_sgns, V_eff, V_full):
    """
    Extend SVD embeddings from reduced to full vocabulary.

    For words in the reduced vocab (ID < V_eff), use SVD embeddings.
    For words outside (ID >= V_eff), use SGNS embeddings projected
    into the SVD space via truncated least-squares.

    In practice, for blending we just use the SVD embeddings directly
    for the reduced vocab and SGNS for the rest.
    """
    W_full = np.zeros((V_full, W_svd.shape[1]), dtype=np.float32)
    W_full[:V_eff] = W_svd

    if V_full > V_eff:
        # For OOV words, use SGNS embeddings (already available)
        # Project them into SVD space approximately
        # Simple approach: use SGNS embeddings directly (they're the
        # best we have for rare words anyway)
        norms_svd = np.linalg.norm(W_svd, axis=1, keepdims=True) + 1e-8
        mean_norm = norms_svd.mean()
        norms_sgns = np.linalg.norm(W_sgns[V_eff:], axis=1, keepdims=True) + 1e-8
        W_full[V_eff:] = W_sgns[V_eff:] / norms_sgns * mean_norm

    return W_full


def blend_embeddings(W_sgns, W_svd, alpha=0.5):
    """
    Blend SGNS and SVD embeddings via weighted average after L2-normalization.

    W_final = alpha * norm(W_sgns) + (1 - alpha) * norm(W_svd)
    """
    def normalize(W):
        norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
        return W / norms

    return alpha * normalize(W_sgns) + (1 - alpha) * normalize(W_svd)


def postprocess_embeddings(W, n_components=2):
    """
    All-but-the-Top post-processing (Mu et al. 2018).
    1. L2-normalize
    2. Mean-center
    3. Remove top-k principal components
    """
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    Wn = W / norms

    mean = Wn.mean(axis=0)
    Wn -= mean

    cov = Wn.T @ Wn / Wn.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top_components = eigenvectors[:, -n_components:]

    projections = Wn @ top_components
    Wn -= projections @ top_components.T

    return Wn.astype(np.float32)
