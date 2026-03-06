#!/usr/bin/env python3

import json
import os

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
from plot import plot_comparison, plot_training
from train import gradient_check, train


def main():
    cfg = Config()
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg); log_lines.append(msg)

    log("=" * 70)
    log("  WORD2VEC SKIP-GRAM WITH NEGATIVE SAMPLING — Pure NumPy")
    log("=" * 70)

    log("\nLoading corpus...")
    tokens = load_text8(cfg.data_dir)
    log(f"  text8: {len(tokens):,} tokens")

    log("\nBuilding vocabulary...")
    word2idx, idx2word, freqs = build_vocab(tokens, cfg.min_count)
    V = len(word2idx)
    log(f"  Vocab: {V:,}  (min_count={cfg.min_count})")
    log(f"  In-vocab tokens: {int(freqs.sum()):,}")

    log("\nPreprocessing...")
    corpus_ids = tokens_to_ids(tokens, word2idx)
    log(f"  Corpus IDs: {len(corpus_ids):,}")
    keep_probs = compute_subsample_probs(freqs, cfg.subsample_t)
    log(f"  Subsample: keep('the')={keep_probs[0]:.4f}")
    neg_table = build_neg_table(freqs, cfg.neg_table_size)

    log("\nGradient check...")
    gradient_check(os.path.join(cfg.results_dir, 'gradient_check.txt'))

    log("\nModel spec:")
    log(f"  dim={cfg.embed_dim}  window={cfg.window_size}  "
        f"neg={cfg.neg_samples}  V={V}")
    log(f"  params={2*V*cfg.embed_dim:,}  "
        f"lr={cfg.lr_start}→{cfg.lr_min}  epochs={cfg.epochs}")
    with open(os.path.join(cfg.results_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    categories = load_analogy_questions(cfg.data_dir)

    log("\nTraining baseline...")
    cfg.use_afws = False
    Wi, Wo, lh, th, lrh, tt = train(
        cfg, corpus_ids, keep_probs, neg_table, freqs, V,
        word2idx, idx2word, categories=categories, label="base")
    log(f"  Time: {tt:.0f}s ({tt/60:.1f}m)")

    np.savez_compressed(os.path.join(cfg.results_dir, 'model_base.npz'),
                        W_in=Wi, W_out=Wo)
    plot_training(lh, th, lrh, cfg.results_dir, prefix='base_')

    log("\nEvaluation...")

    log("\n  -- W_in --")
    r1, s1 = evaluate_analogies(Wi, word2idx, idx2word, categories)
    print_eval(r1, s1, log)

    log("\n  -- W_in + W_out --")
    Wc = Wi + Wo
    r2, s2 = evaluate_analogies(Wc, word2idx, idx2word, categories)
    print_eval(r2, s2, log)

    if s2['semantic']['accuracy'] > s1['semantic']['accuracy']:
        bestW, bestR, bestS, blab = Wc, r2, s2, "W_in+W_out"
    else:
        bestW, bestR, bestS, blab = Wi, r1, s1, "W_in"
    log(f"\n  Best: {blab}  Semantic: {bestS['semantic']['accuracy']*100:.1f}%")

    log("\nQualitative examples...")
    probes = ['king', 'queen', 'computer', 'france', 'good', 'university',
              'water', 'dog', 'music', 'science', 'war', 'love', 'money', 'fast']
    nn = nearest_neighbors(bestW, idx2word, word2idx, probes)
    log("\n  Nearest neighbours:")
    for w, ns in nn.items():
        log(f"    {w}: " + ', '.join(f"{n}({s:.3f})" for n, s in ns[:8]))

    aq = [('man','king','woman'), ('france','paris','germany'),
          ('japan','tokyo','china'), ('big','bigger','small'),
          ('good','best','bad'), ('man','woman','boy'),
          ('slow','slowly','quick'), ('france','french','spain'),
          ('king','queen','man'), ('going','went','playing')]
    ae = analogy_examples(bestW, idx2word, word2idx, aq)
    log("\n  Analogies (a : b :: c : ?):")
    for a, b, c, ps in ae:
        log(f"    {a}:{b} :: {c}:?  →  " + ', '.join(f"{w}({s:.3f})" for w, s in ps))

    save_all(cfg.results_dir, bestR, bestS, nn, ae, log_lines)

    # Adaptive Frequency-Based Window Sizing
    log("\n" + "=" * 70)
    log("  NOVEL: Adaptive Frequency-Based Window Sizing (AFWS)")
    log("=" * 70)
    log(f"  Hypothesis: rare words benefit from larger context windows.")
    log(f"  Window range: [{cfg.afws_min_window}, {cfg.afws_max_window}], "
        f"alpha={cfg.afws_alpha}")

    cfg2 = Config()
    cfg2.use_afws = True
    cfg2.epochs = 10 
    Wi2, Wo2, lh2, th2, lrh2, tt2 = train(
        cfg2, corpus_ids, keep_probs, neg_table, freqs, V,
        word2idx, idx2word, categories=categories, label="AFWS")

    np.savez_compressed(os.path.join(cfg.results_dir, 'model_afws.npz'),
                        W_in=Wi2, W_out=Wo2)

    log("\n  AFWS W_in:")
    ra1, sa1 = evaluate_analogies(Wi2, word2idx, idx2word, categories)
    print_eval(ra1, sa1, log)
    log("\n  AFWS W_in+W_out:")
    Wc2 = Wi2 + Wo2
    ra2, sa2 = evaluate_analogies(Wc2, word2idx, idx2word, categories)
    print_eval(ra2, sa2, log)

    afws_best = max(sa1['semantic']['accuracy'], sa2['semantic']['accuracy'])
    base_best = bestS['semantic']['accuracy']
    log(f"\n  COMPARISON: baseline={base_best*100:.1f}%  "
        f"AFWS={afws_best*100:.1f}%  "
        f"diff={((afws_best - base_best)*100):+.1f}%")

    plot_comparison(lh, th, lh2, th2, cfg.results_dir)

    with open(os.path.join(cfg.results_dir, 'full_log.txt'), 'w') as f:
        f.write('\n'.join(log_lines) + '\n')

    log("\nDONE.")


if __name__ == '__main__':
    main()
