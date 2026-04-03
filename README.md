# Word2Vec from Scratch — Pure NumPy

A complete implementation of [Word2Vec](https://arxiv.org/abs/1301.3781) Skip-Gram with Negative Sampling, built from the ground up using only NumPy. No frameworks, no shortcuts — every component from gradient derivation to batched SGD is implemented and verified by hand.

Trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus (17M tokens, 71K vocabulary), the model reaches **55.1% semantic accuracy** on the [Google Word Analogy](https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)) benchmark — near the practical ceiling for this corpus size.

## What This Project Demonstrates

- **Mathematical foundations** — loss function, gradient derivation, and finite-difference verification (max relative error < 1e-7)
- **Efficient NumPy engineering** — vectorized pair generation, batched updates, float32 throughout, 0.12M pairs/sec on CPU
- **Systematic experimentation** — 7 training phases across K=5/10/15 negative samples, with ablation analysis
- **Post-processing** — "All-but-the-Top" (Mu et al. 2018) removes frequency bias from the embedding space (+5.4% accuracy, zero retraining)
- **SVD baseline** — PPMI matrix factorization (Levy & Goldberg 2014) as a count-based alternative to neural training
- **Novel experiment** — Adaptive Frequency-Based Window Sizing (AFWS), a negative result documented with analysis of why it fails

## Results

### Word Analogy Benchmark

| | Accuracy |
|---|---|
| Semantic (capitals, nationalities, ...) | **55.1%** |
| Syntactic (plurals, tenses, ...) | 23.8% |
| Overall | 36.8% |

Top-performing categories:

| Category | Accuracy |
|---|---|
| capital-common-countries | 81.6% |
| nationality-adjective | 70.3% |
| capital-world | 64.5% |
| city-in-state | 48.3% |

### Analogies

```
man : king  :: woman : ?  →  queen  ✓
france : paris  :: germany : ?  →  berlin  ✓
japan : tokyo  :: china : ?  →  beijing  ✓
big : bigger  :: small : ?  →  smaller  ✓
france : french  :: spain : ?  →  spanish  ✓
going : went  :: playing : ?  →  played  ✓
slow : slowly  :: quick : ?  →  surrendered  ✗
good : best  :: bad : ?  →  award  ✗
```

### Nearest Neighbors

```
king     → son, kings, queen, prince, throne
computer → computers, software, hardware, computing, programming
france   → germany, italy, french, netherlands, belgium
dog      → pictus, keeshond, catahoula, poodle, dogs
```

### Training Progression

| Phase | Neg. Samples | Epochs | Semantic Accuracy |
|---|---|---|---|
| 1 | K=5, from scratch | 20 | 47.9% |
| 2 | K=10, fine-tuned | 20 | 52.0% |
| 6 | K=15, from scratch | 20 | 54.2% |
| **7** | **K=15, fine-tuned** | **10** | **55.1%** |

The jump from K=5 to K=15 (+6.7%) was the single largest gain. Post-processing added another +5.4% on top. Full analysis in [REPORT.md](REPORT.md).

## Model Architecture

```
Corpus (17M tokens)
  → Subsampling (frequent word downweighting)
  → Pair generation (vectorized, symmetric, distance-weighted)
  → Skip-Gram with Negative Sampling
      W_in:  71,290 × 300  (center embeddings)
      W_out: 71,290 × 300  (context embeddings)
  → SGD with linear LR decay (0.025 → 0.0001)
  → Post-processing: L2-norm → mean-center → PCA-2 removal
```

Total parameters: 42.8M (float32, 171 MB)

## Quick Start

```bash
pip install -r requirements.txt

# Full baseline pipeline (~30-60 min on CPU)
python main.py

# Advanced techniques: position weighting + SVD refinement
python main_improved.py

# Streamlined experiment comparison
python run_experiments.py
```

The corpus downloads automatically on first run.

## Using Pre-trained Embeddings

```python
import numpy as np

# Best post-processed embeddings
data = np.load('results/model_final_best.npz')
W = data['W']  # (71290, 300)

# Raw model weights
data = np.load('results/model_k15_ft.npz')
W_in  = data['W_in']   # center embeddings
W_out = data['W_out']  # context embeddings
```

## Project Structure

```
main.py              Baseline training pipeline (gradient check → train → evaluate)
main_improved.py     Position weighting, SVD refinement, blending experiments
run_experiments.py   Streamlined experiment runner with comparison tables

train.py             Training loop, batched SGD, gradient checking
config.py            All hyperparameters in one place
data.py              Corpus loading, vocabulary, subsampling, negative sampling table
pairs.py             Context-pair generation (standard, weighted, AFWS)
evaluate.py          Google analogy benchmark, nearest-neighbor search
svd_embeddings.py    Co-occurrence matrix, SPPMI, truncated SVD, embedding blending
math_utils.py        Numerically stable sigmoid / log-sigmoid
io_utils.py          Logging and result serialization
plot.py              Training curve visualization

REPORT.md            Full technical report (gradients, training phases, analysis)
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `embed_dim` | 300 | Embedding dimensionality |
| `window_size` | 10 | Context window radius |
| `neg_samples` | 5 | Negative samples per positive pair |
| `min_count` | 5 | Minimum word frequency for vocabulary |
| `subsample_t` | 1e-5 | Subsampling threshold for frequent words |
| `lr_start` / `lr_min` | 0.025 / 0.0001 | Linear LR decay bounds |
| `epochs` | 20 | Training epochs |
| `batch_size` | 4096 | SGD mini-batch size |

## Technical Report

[REPORT.md](REPORT.md) covers:
- Full gradient derivation and verification methodology
- Comparison of negative sampling vs. full softmax vs. hierarchical softmax
- Per-phase training analysis with lessons learned (e.g., why high LR destroys fine-tuned embeddings)
- Post-processing theory and ablation (why removing top-2 PCA components helps)
- AFWS experiment design, results, and failure analysis
- Compute budget (~16 hours, ~6.9B training pairs across all phases)

## License

[MIT](LICENSE)
