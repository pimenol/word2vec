#!/usr/bin/env python3
"""
Streamlined experiment runner — focuses on what works.

Phase 1: SVD-based refinement of baseline (no training needed)
Phase 2: Position-weighted training with gentle 1/sqrt(d) weighting
Phase 3: Combinations and final evaluation
"""

import os
import time

import numpy as np

from config import Config
from data import (
    build_neg_table,
    build_vocab,
    compute_subsample_probs,
    load_analogy_questions,
    load_text8,
    tokens_to_ids,
)
from evaluate import evaluate_analogies
from io_utils import print_eval
from svd_embeddings import (
    blend_embeddings,
    build_cooccurrence_matrix,
    compute_sppmi_inplace,
    extend_svd_to_full_vocab,
    postprocess_embeddings,
    svd_embeddings,
)
from train import gradient_check, gradient_check_weighted, train


def eval_full(W, word2idx, idx2word, categories, label, log):
    """Evaluate with raw and post-processed, return both summaries."""
    r_raw, s_raw = evaluate_analogies(W, word2idx, idx2word, categories)
    Wp = postprocess_embeddings(W.copy(), n_components=2)
    r_pp, s_pp = evaluate_analogies(Wp, word2idx, idx2word, categories)
    log(f"  {label:50s}  raw: sem={s_raw['semantic']['accuracy']*100:5.1f}% "
        f"syn={s_raw['syntactic']['accuracy']*100:5.1f}%  |  "
        f"pp: sem={s_pp['semantic']['accuracy']*100:5.1f}% "
        f"syn={s_pp['syntactic']['accuracy']*100:5.1f}% "
        f"all={s_pp['overall']['accuracy']*100:5.1f}%")
    return r_raw, s_raw, r_pp, s_pp


def print_category_breakdown(r, s, log):
    """Full per-category evaluation."""
    log(f"\n    {'Category':<35} {'Corr':>6} {'Tot':>6} {'Skip':>6} {'Acc':>8}")
    log(f"    {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for c, res in sorted(r.items()):
        if res['is_semantic']:
            log(f"    {c:<35} {res['correct']:>6} {res['total']:>6} "
                f"{res['skipped']:>6} {res['accuracy']*100:>7.1f}%")
    ss = s['semantic']
    log(f"    {'SEMANTIC TOTAL':<35} {ss['correct']:>6} {ss['total']:>6} "
        f"{ss['skipped']:>6} {ss['accuracy']*100:>7.1f}%")
    log("")
    for c, res in sorted(r.items()):
        if not res['is_semantic']:
            log(f"    {c:<35} {res['correct']:>6} {res['total']:>6} "
                f"{res['skipped']:>6} {res['accuracy']*100:>7.1f}%")
    ss = s['syntactic']
    log(f"    {'SYNTACTIC TOTAL':<35} {ss['correct']:>6} {ss['total']:>6} "
        f"{ss['skipped']:>6} {ss['accuracy']*100:>7.1f}%")
    ss = s['overall']
    log(f"    {'=== OVERALL ===':<35} {ss['correct']:>6} {ss['total']:>6} "
        f"{ss['skipped']:>6} {ss['accuracy']*100:>7.1f}%")


def failure_analysis(W_best, W_base, word2idx, idx2word, categories, log):
    """Show examples where best model differs from baseline."""
    norms1 = np.linalg.norm(W_best, axis=1, keepdims=True) + 1e-8
    Wn1 = (W_best / norms1).astype(np.float32)
    norms2 = np.linalg.norm(W_base, axis=1, keepdims=True) + 1e-8
    Wn2 = (W_base / norms2).astype(np.float32)

    improved, regressed = [], []

    for cat, qs in categories.items():
        answerable = [q for q in qs if all(w in word2idx for w in q)]
        for q in answerable:
            ai, bi, ci, di = [word2idx[w] for w in q]
            vec1 = Wn1[bi] - Wn1[ai] + Wn1[ci]
            vec1 /= np.linalg.norm(vec1) + 1e-8
            sims1 = Wn1 @ vec1
            for idx in [ai, bi, ci]: sims1[idx] = -np.inf
            pred1 = np.argmax(sims1)

            vec2 = Wn2[bi] - Wn2[ai] + Wn2[ci]
            vec2 /= np.linalg.norm(vec2) + 1e-8
            sims2 = Wn2 @ vec2
            for idx in [ai, bi, ci]: sims2[idx] = -np.inf
            pred2 = np.argmax(sims2)

            if pred1 == di and pred2 != di:
                improved.append((cat, q, idx2word[pred2]))
            elif pred2 == di and pred1 != di:
                regressed.append((cat, q, idx2word[pred1]))

    log(f"\n  Failure Analysis:")
    log(f"  Improved (best succeeds, baseline fails): {len(improved)}")
    log(f"  Regressed (baseline succeeds, best fails): {len(regressed)}")
    log(f"\n  10 Improved examples:")
    for cat, (a, b, c, d), base_pred in improved[:10]:
        log(f"    [{cat}] {a}:{b} :: {c}:? → {d} (baseline: {base_pred})")
    log(f"\n  10 Regressed examples:")
    for cat, (a, b, c, d), best_pred in regressed[:10]:
        log(f"    [{cat}] {a}:{b} :: {c}:? → {d} (best: {best_pred})")


def main():
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg); log_lines.append(msg)

    log("=" * 80)
    log("  WORD2VEC ADVANCED TECHNIQUES — Streamlined Experiments")
    log("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────
    log("\nLoading corpus & vocab...")
    tokens = load_text8(cfg.data_dir)
    word2idx, idx2word, freqs = build_vocab(tokens, cfg.min_count)
    V = len(word2idx)
    corpus_ids = tokens_to_ids(tokens, word2idx)
    keep_probs = compute_subsample_probs(freqs, cfg.subsample_t)
    neg_table = build_neg_table(freqs, cfg.neg_table_size)
    categories = load_analogy_questions(cfg.data_dir)
    log(f"  V={V:,}, corpus={len(corpus_ids):,} tokens")

    # ── Gradient checks ───────────────────────────────────────────────
    log("\nGradient checks...")
    gradient_check(os.path.join(cfg.results_dir, 'gradient_check.txt'))
    gradient_check_weighted(os.path.join(cfg.results_dir,
                                         'gradient_check_weighted.txt'))
    log("  All passed.")

    # ── Load baseline ──────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  BASELINE (Phase 7, K=15)")
    log("=" * 80)
    ft_path = os.path.join(cfg.results_dir, 'model_k15_ft.npz')
    data = np.load(ft_path)
    Wi_base, Wo_base = data['W_in'], data['W_out']
    W_base = Wi_base + Wo_base

    r_base_raw, s_base_raw, r_base_pp, s_base_pp = eval_full(
        W_base, word2idx, idx2word, categories, "Baseline (W_in+W_out)", log)

    comparison = []
    comparison.append(("Baseline (W_in+W_out)", s_base_raw, s_base_pp))
    W_base_pp = postprocess_embeddings(W_base.copy())

    # ══════════════════════════════════════════════════════════════════
    #  TECHNIQUE 2: SVD-BASED REFINEMENT
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  TECHNIQUE 2: SVD-Based Refinement (SPPMI + Truncated SVD)")
    log("=" * 80)

    # Use reduced vocabulary (top 30K) to fit in 18 GB RAM
    # 30K x 30K x 4B = 3.4 GB
    MAX_VOCAB_SVD = 30000
    log(f"  Using reduced vocabulary: top {MAX_VOCAB_SVD:,} words")

    sppmi_path = os.path.join(cfg.results_dir, f'sppmi_{MAX_VOCAB_SVD}.npz')
    if os.path.exists(sppmi_path):
        log("  Loading cached SPPMI matrix...")
        SPPMI = np.load(sppmi_path)['SPPMI']
        V_eff = SPPMI.shape[0]
    else:
        C, V_eff = build_cooccurrence_matrix(
            corpus_ids, V, window=10, weighted=True, max_vocab=MAX_VOCAB_SVD)
        SPPMI = compute_sppmi_inplace(C, neg_k=15)
        np.savez_compressed(sppmi_path, SPPMI=SPPMI)

    # SVD with different power values
    log("\n  SVD embeddings (varying power):")
    best_svd_sem = 0.0
    best_svd_power = 0.5
    best_svd_W_full = None

    for power in [0.0, 0.25, 0.5, 0.75, 1.0]:
        W_svd_reduced = svd_embeddings(SPPMI, dim=300, power=power)
        # Extend to full vocabulary
        W_svd_full = extend_svd_to_full_vocab(W_svd_reduced, W_base, V_eff, V)
        label = f"SVD-only (p={power}, V={V_eff})"
        _, s_raw, _, s_pp = eval_full(
            W_svd_full, word2idx, idx2word, categories, label, log)
        comparison.append((label, s_raw, s_pp))
        if s_pp['semantic']['accuracy'] > best_svd_sem:
            best_svd_sem = s_pp['semantic']['accuracy']
            best_svd_power = power
            best_svd_W_full = W_svd_full.copy()

    log(f"\n  Best SVD-only: power={best_svd_power}, "
        f"sem={best_svd_sem*100:.1f}% (pp)")

    # Blending SGNS + SVD (various alphas)
    log("\n  Blended SGNS+SVD embeddings:")
    best_blend_sem = s_base_pp['semantic']['accuracy']
    best_blend_config = "Baseline"
    best_blend_W = W_base.copy()

    for power in [0.0, 0.25, 0.5, 0.75, 1.0]:
        W_svd_reduced = svd_embeddings(SPPMI, dim=300, power=power)
        W_svd_full = extend_svd_to_full_vocab(W_svd_reduced, W_base, V_eff, V)
        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            W_blend = blend_embeddings(W_base, W_svd_full, alpha=alpha)
            label = f"Blend(α={alpha}, p={power})"
            _, s_raw, _, s_pp = eval_full(
                W_blend, word2idx, idx2word, categories, label, log)
            comparison.append((label, s_raw, s_pp))
            if s_pp['semantic']['accuracy'] > best_blend_sem:
                best_blend_sem = s_pp['semantic']['accuracy']
                best_blend_config = label
                best_blend_W = W_blend.copy()

    log(f"\n  *** Best blend: {best_blend_config} — "
        f"Semantic: {best_blend_sem*100:.1f}% (post-proc) ***")

    # ══════════════════════════════════════════════════════════════════
    #  TECHNIQUE 1: POSITION-WEIGHTED TRAINING (revised: 1/sqrt(d))
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  TECHNIQUE 1: Position-Dependent Context Weighting (1/√d)")
    log("=" * 80)

    pw_path = os.path.join(cfg.results_dir, 'model_posweight_sqrt.npz')
    if os.path.exists(pw_path):
        log("  Loading position-weighted (√d) model from checkpoint...")
        data = np.load(pw_path)
        Wi_pw, Wo_pw = data['W_in'], data['W_out']
    else:
        cfg_pw = Config()
        cfg_pw.use_position_weights = True
        cfg_pw.position_weight_power = 0.5  # 1/d^0.5 = 1/sqrt(d)
        cfg_pw.use_afws = False
        cfg_pw.neg_samples = 15
        cfg_pw.epochs = 20
        cfg_pw.seed = 1

        log(f"  Config: window={cfg_pw.window_size}, K={cfg_pw.neg_samples}, "
            f"epochs={cfg_pw.epochs}, weight=1/d^0.5")

        Wi_pw, Wo_pw, lh_pw, th_pw, lrh_pw, tt_pw = train(
            cfg_pw, corpus_ids, keep_probs, neg_table, freqs, V,
            word2idx, idx2word, categories=categories, label="pw-sqrt")
        log(f"  Time: {tt_pw:.0f}s ({tt_pw/60:.1f}m)")
        np.savez_compressed(pw_path, W_in=Wi_pw, W_out=Wo_pw)

    W_pw = Wi_pw + Wo_pw
    _, s_pw_raw, _, s_pw_pp = eval_full(
        W_pw, word2idx, idx2word, categories, "PosWeight-√d (W_in+W_out)", log)
    comparison.append(("PosWeight-√d", s_pw_raw, s_pw_pp))

    # Blend PW with SVD
    log("\n  Blending PW + SVD:")
    for power in [best_svd_power]:
        W_svd_reduced = svd_embeddings(SPPMI, dim=300, power=power)
        W_svd_full = extend_svd_to_full_vocab(W_svd_reduced, W_pw, V_eff, V)
        for alpha in [0.3, 0.5, 0.7]:
            W_blend_pw = blend_embeddings(W_pw, W_svd_full, alpha=alpha)
            label = f"Blend-PW+SVD(α={alpha}, p={power})"
            _, s_raw, _, s_pp = eval_full(
                W_blend_pw, word2idx, idx2word, categories, label, log)
            comparison.append((label, s_raw, s_pp))
            if s_pp['semantic']['accuracy'] > best_blend_sem:
                best_blend_sem = s_pp['semantic']['accuracy']
                best_blend_config = label
                best_blend_W = W_blend_pw.copy()

    # ══════════════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("  RESULTS COMPARISON TABLE")
    log("=" * 80)
    header = (f"  {'Configuration':50s} "
              f"{'Sem(raw)':>8s} {'Sem(pp)':>8s} "
              f"{'Syn(raw)':>8s} {'Syn(pp)':>8s} {'All(pp)':>8s}")
    log(header)
    log("  " + "-" * 92)
    for name, s_raw, s_pp in comparison:
        log(f"  {name:50s} "
            f"{s_raw['semantic']['accuracy']*100:7.1f}% "
            f"{s_pp['semantic']['accuracy']*100:7.1f}% "
            f"{s_raw['syntactic']['accuracy']*100:7.1f}% "
            f"{s_pp['syntactic']['accuracy']*100:7.1f}% "
            f"{s_pp['overall']['accuracy']*100:7.1f}%")

    # ══════════════════════════════════════════════════════════════════
    #  BEST MODEL — Per-category breakdown + failure analysis
    # ══════════════════════════════════════════════════════════════════
    log(f"\n  *** BEST: {best_blend_config} — "
        f"Semantic: {best_blend_sem*100:.1f}% ***")

    log("\n" + "=" * 80)
    log(f"  PER-CATEGORY BREAKDOWN: {best_blend_config}")
    log("=" * 80)
    Wp_best = postprocess_embeddings(best_blend_W.copy())
    r_best, s_best = evaluate_analogies(Wp_best, word2idx, idx2word, categories)
    print_category_breakdown(r_best, s_best, log)

    log("\n" + "=" * 80)
    log("  FAILURE ANALYSIS")
    log("=" * 80)
    failure_analysis(Wp_best, W_base_pp, word2idx, idx2word, categories, log)

    # ── Save ──────────────────────────────────────────────────────────
    with open(os.path.join(cfg.results_dir, 'improved_results.txt'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')

    log("\nDONE.")


if __name__ == '__main__':
    main()
