#!/usr/bin/env python3
"""
Improved Word2Vec — runs all experiments and produces comparison table.

Techniques:
  1. Position-dependent context weighting (1/d)
  2. SVD-based refinement (SPPMI + truncated SVD)
  3. Combination of techniques 1 and 2
"""

import json
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
from evaluate import analogy_examples, evaluate_analogies, nearest_neighbors
from io_utils import print_eval, save_all
from plot import plot_training
from svd_embeddings import (
    blend_embeddings,
    build_cooccurrence_matrix,
    compute_sppmi,
    postprocess_embeddings,
    svd_embeddings,
)
from train import gradient_check, gradient_check_weighted, train


def evaluate_with_postproc(W, word2idx, idx2word, categories, label, log,
                           n_components=2):
    """Evaluate raw and post-processed embeddings, return both results."""
    log(f"\n  -- {label} (raw) --")
    r_raw, s_raw = evaluate_analogies(W, word2idx, idx2word, categories)
    print_eval(r_raw, s_raw, log)

    Wp = postprocess_embeddings(W, n_components=n_components)
    log(f"\n  -- {label} (post-proc PCA-{n_components}) --")
    r_pp, s_pp = evaluate_analogies(Wp, word2idx, idx2word, categories)
    print_eval(r_pp, s_pp, log)

    return r_raw, s_raw, r_pp, s_pp


def results_row(label, s_raw, s_pp):
    """Format a single results row."""
    return (f"| {label:40s} "
            f"| {s_raw['semantic']['accuracy']*100:5.1f}% "
            f"| {s_pp['semantic']['accuracy']*100:5.1f}% "
            f"| {s_raw['syntactic']['accuracy']*100:5.1f}% "
            f"| {s_pp['syntactic']['accuracy']*100:5.1f}% "
            f"| {s_pp['overall']['accuracy']*100:5.1f}% |")


def print_comparison_table(rows, log):
    """Print the final comparison table."""
    log("\n" + "=" * 100)
    log("  RESULTS COMPARISON TABLE")
    log("=" * 100)
    header = (f"| {'Configuration':40s} "
              f"| {'Sem(raw)':>8s} "
              f"| {'Sem(pp)':>8s} "
              f"| {'Syn(raw)':>8s} "
              f"| {'Syn(pp)':>8s} "
              f"| {'Overall':>8s} |")
    sep = "|" + "-" * 41 + "|" + (("-" * 9 + "|") * 5)
    log(header)
    log(sep)
    for row in rows:
        log(row)
    log("")


def failure_analysis(W_best, W_base, word2idx, idx2word, categories, log):
    """Show 10 analogies where best model succeeds but baseline fails."""
    from evaluate import SEMANTIC_CATS
    norms1 = np.linalg.norm(W_best, axis=1, keepdims=True) + 1e-8
    Wn1 = (W_best / norms1).astype(np.float32)
    norms2 = np.linalg.norm(W_base, axis=1, keepdims=True) + 1e-8
    Wn2 = (W_base / norms2).astype(np.float32)

    improved = []
    regressed = []

    for cat, qs in categories.items():
        answerable = [q for q in qs if all(w in word2idx for w in q)]
        for q in answerable:
            ai, bi, ci, di = [word2idx[w] for w in q]
            # Best model prediction
            vec1 = Wn1[bi] - Wn1[ai] + Wn1[ci]
            vec1 /= np.linalg.norm(vec1) + 1e-8
            sims1 = Wn1 @ vec1
            for idx in [ai, bi, ci]:
                sims1[idx] = -np.inf
            pred1 = np.argmax(sims1)

            # Baseline prediction
            vec2 = Wn2[bi] - Wn2[ai] + Wn2[ci]
            vec2 /= np.linalg.norm(vec2) + 1e-8
            sims2 = Wn2 @ vec2
            for idx in [ai, bi, ci]:
                sims2[idx] = -np.inf
            pred2 = np.argmax(sims2)

            if pred1 == di and pred2 != di:
                improved.append((cat, q, idx2word[pred2]))
            elif pred2 == di and pred1 != di:
                regressed.append((cat, q, idx2word[pred1]))

    log(f"\n  Failure Analysis:")
    log(f"  Improved (best succeeds, baseline fails): {len(improved)} examples")
    log(f"  Regressed (baseline succeeds, best fails): {len(regressed)} examples")

    log(f"\n  10 Improved examples:")
    for cat, (a, b, c, d), base_pred in improved[:10]:
        log(f"    [{cat}] {a}:{b} :: {c}:? → {d} "
            f"(baseline predicted: {base_pred})")

    log(f"\n  10 Regressed examples:")
    for cat, (a, b, c, d), best_pred in regressed[:10]:
        log(f"    [{cat}] {a}:{b} :: {c}:? → {d} "
            f"(best predicted: {best_pred})")


def main():
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg); log_lines.append(msg)

    log("=" * 70)
    log("  WORD2VEC IMPROVED — Advanced Techniques Evaluation")
    log("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    log("\nLoading corpus...")
    tokens = load_text8(cfg.data_dir)
    log(f"  text8: {len(tokens):,} tokens")

    log("\nBuilding vocabulary...")
    word2idx, idx2word, freqs = build_vocab(tokens, cfg.min_count)
    V = len(word2idx)
    log(f"  Vocab: {V:,}  (min_count={cfg.min_count})")

    log("\nPreprocessing...")
    corpus_ids = tokens_to_ids(tokens, word2idx)
    keep_probs = compute_subsample_probs(freqs, cfg.subsample_t)
    neg_table = build_neg_table(freqs, cfg.neg_table_size)
    categories = load_analogy_questions(cfg.data_dir)

    # ── Gradient checks ───────────────────────────────────────────────
    log("\nGradient check (standard)...")
    gradient_check(os.path.join(cfg.results_dir, 'gradient_check.txt'))
    log("Gradient check (position-weighted)...")
    gradient_check_weighted(os.path.join(cfg.results_dir,
                                         'gradient_check_weighted.txt'))

    comparison_rows = []

    # ── Load or train baseline ─────────────────────────────────────────
    baseline_path = os.path.join(cfg.results_dir, 'model_base.npz')
    if os.path.exists(baseline_path):
        log("\nLoading baseline model from checkpoint...")
        data = np.load(baseline_path)
        Wi_base = data['W_in']
        Wo_base = data['W_out']
    else:
        log("\nTraining baseline (K=15, 20 epochs)...")
        cfg_base = Config()
        cfg_base.use_afws = False
        cfg_base.use_position_weights = False
        cfg_base.neg_samples = 15
        cfg_base.epochs = 20
        Wi_base, Wo_base, lh, th, lrh, tt = train(
            cfg_base, corpus_ids, keep_probs, neg_table, freqs, V,
            word2idx, idx2word, categories=categories, label="base")
        np.savez_compressed(baseline_path, W_in=Wi_base, W_out=Wo_base)
        plot_training(lh, th, lrh, cfg.results_dir, prefix='base_')

    # Evaluate baseline
    log("\n" + "=" * 70)
    log("  BASELINE EVALUATION")
    log("=" * 70)

    # Try W_in, W_in+W_out, pick best
    W_base_combined = Wi_base + Wo_base
    r1_raw, s1_raw, r1_pp, s1_pp = evaluate_with_postproc(
        Wi_base, word2idx, idx2word, categories, "Baseline W_in", log)
    r2_raw, s2_raw, r2_pp, s2_pp = evaluate_with_postproc(
        W_base_combined, word2idx, idx2word, categories,
        "Baseline W_in+W_out", log)

    if s2_pp['semantic']['accuracy'] > s1_pp['semantic']['accuracy']:
        W_base_best = W_base_combined
        s_base_raw, s_base_pp = s2_raw, s2_pp
        r_base_pp = r2_pp
        base_label = "Baseline (W_in+W_out)"
    else:
        W_base_best = Wi_base
        s_base_raw, s_base_pp = s1_raw, s1_pp
        r_base_pp = r1_pp
        base_label = "Baseline (W_in)"

    comparison_rows.append(results_row(base_label, s_base_raw, s_base_pp))
    W_base_pp = postprocess_embeddings(W_base_best.copy())

    log(f"\n  Best baseline: {base_label} — "
        f"Semantic: {s_base_pp['semantic']['accuracy']*100:.1f}% (post-proc)")

    # ── Technique 1: Position-Dependent Context Weighting ──────────────
    log("\n" + "=" * 70)
    log("  TECHNIQUE 1: Position-Dependent Context Weighting (1/d)")
    log("=" * 70)

    pw_path = os.path.join(cfg.results_dir, 'model_posweight.npz')
    if os.path.exists(pw_path):
        log("  Loading position-weighted model from checkpoint...")
        data = np.load(pw_path)
        Wi_pw = data['W_in']
        Wo_pw = data['W_out']
    else:
        cfg_pw = Config()
        cfg_pw.use_position_weights = True
        cfg_pw.use_afws = False
        cfg_pw.neg_samples = 15
        cfg_pw.epochs = 20
        cfg_pw.seed = 1

        log(f"  Config: window={cfg_pw.window_size}, K={cfg_pw.neg_samples}, "
            f"epochs={cfg_pw.epochs}, position_weights=True")

        Wi_pw, Wo_pw, lh_pw, th_pw, lrh_pw, tt_pw = train(
            cfg_pw, corpus_ids, keep_probs, neg_table, freqs, V,
            word2idx, idx2word, categories=categories, label="pos-wt")
        log(f"  Time: {tt_pw:.0f}s ({tt_pw/60:.1f}m)")
        np.savez_compressed(pw_path, W_in=Wi_pw, W_out=Wo_pw)
        plot_training(lh_pw, th_pw, lrh_pw, cfg.results_dir, prefix='pw_')

    W_pw_combined = Wi_pw + Wo_pw
    r_pw_raw, s_pw_raw, r_pw_pp, s_pw_pp = evaluate_with_postproc(
        W_pw_combined, word2idx, idx2word, categories,
        "PosWeight W_in+W_out", log)
    comparison_rows.append(results_row("Technique 1: Pos-Weight 1/d",
                                       s_pw_raw, s_pw_pp))

    # ── Technique 2: SVD-Based Refinement ──────────────────────────────
    log("\n" + "=" * 70)
    log("  TECHNIQUE 2: SVD-Based Refinement (SPPMI)")
    log("=" * 70)

    sppmi_path = os.path.join(cfg.results_dir, 'sppmi_matrix.npz')
    if os.path.exists(sppmi_path):
        log("  Loading SPPMI matrix from cache...")
        SPPMI = np.load(sppmi_path)['SPPMI']
    else:
        C = build_cooccurrence_matrix(corpus_ids, V, window=10, weighted=True)
        SPPMI = compute_sppmi(C, neg_k=15)
        np.savez_compressed(sppmi_path, SPPMI=SPPMI)
        del C

    # Try different power values
    for power in [0.0, 0.5, 1.0]:
        W_svd = svd_embeddings(SPPMI, dim=300, power=power)
        label = f"SVD (power={power})"
        r_raw, s_raw, r_pp, s_pp = evaluate_with_postproc(
            W_svd, word2idx, idx2word, categories, label, log)
        comparison_rows.append(results_row(f"Technique 2: {label}",
                                           s_raw, s_pp))

    # ── Blending: SGNS + SVD ───────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  BLENDING: SGNS + SVD")
    log("=" * 70)

    # Use best SVD power (0.5 typically works)
    W_svd_best = svd_embeddings(SPPMI, dim=300, power=0.5)

    # Try blending with baseline SGNS
    for alpha in [0.3, 0.5, 0.7]:
        W_blend = blend_embeddings(W_base_best, W_svd_best, alpha=alpha)
        label = f"Blend base+SVD (α={alpha})"
        r_raw, s_raw, r_pp, s_pp = evaluate_with_postproc(
            W_blend, word2idx, idx2word, categories, label, log)
        comparison_rows.append(results_row(label, s_raw, s_pp))

    # ── Combination: Technique 1 + Technique 2 ────────────────────────
    log("\n" + "=" * 70)
    log("  COMBINATION: Pos-Weight + SVD Blending")
    log("=" * 70)

    for alpha in [0.3, 0.5, 0.7]:
        W_blend_pw = blend_embeddings(W_pw_combined, W_svd_best, alpha=alpha)
        label = f"Blend PW+SVD (α={alpha})"
        r_raw, s_raw, r_pp, s_pp = evaluate_with_postproc(
            W_blend_pw, word2idx, idx2word, categories, label, log)
        comparison_rows.append(results_row(label, s_raw, s_pp))

    # ── Final comparison table ─────────────────────────────────────────
    print_comparison_table(comparison_rows, log)

    # ── Find the best overall configuration ────────────────────────────
    # Re-evaluate all configurations to find the best one
    best_sem = 0.0
    best_config = None
    best_W = None
    best_r_pp = None
    best_s_pp = None
    best_s_raw = None

    configs = [
        ("Baseline", W_base_best),
        ("Pos-Weight", W_pw_combined),
        ("SVD-0.5", W_svd_best),
    ]
    for alpha in [0.3, 0.5, 0.7]:
        configs.append((f"Blend-base+SVD-{alpha}",
                        blend_embeddings(W_base_best, W_svd_best, alpha)))
        configs.append((f"Blend-PW+SVD-{alpha}",
                        blend_embeddings(W_pw_combined, W_svd_best, alpha)))

    for name, W in configs:
        Wp = postprocess_embeddings(W.copy())
        _, s = evaluate_analogies(Wp, word2idx, idx2word, categories)
        sem = s['semantic']['accuracy']
        if sem > best_sem:
            best_sem = sem
            best_config = name
            best_W = W
            # Re-evaluate for full results
            best_r_pp, best_s_pp = evaluate_analogies(
                Wp, word2idx, idx2word, categories)
            _, best_s_raw = evaluate_analogies(
                W, word2idx, idx2word, categories)

    log(f"\n  *** BEST CONFIG: {best_config} — "
        f"Semantic: {best_sem*100:.1f}% (post-proc) ***")

    # ── Per-category breakdown for best ────────────────────────────────
    log("\n" + "=" * 70)
    log(f"  PER-CATEGORY BREAKDOWN: {best_config}")
    log("=" * 70)
    Wp_best = postprocess_embeddings(best_W.copy())
    r_best, s_best = evaluate_analogies(Wp_best, word2idx, idx2word, categories)
    print_eval(r_best, s_best, log)

    # ── Failure analysis ───────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  FAILURE ANALYSIS")
    log("=" * 70)
    failure_analysis(Wp_best, W_base_pp, word2idx, idx2word, categories, log)

    # ── Save everything ────────────────────────────────────────────────
    with open(os.path.join(cfg.results_dir, 'improved_log.txt'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')

    log("\nDONE.")


if __name__ == '__main__':
    main()
