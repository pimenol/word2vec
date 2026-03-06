# Word2Vec — Skip-Gram with Negative Sampling (Pure NumPy)

A from-scratch implementation of Word2Vec SGNS trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus

## Project Structure

```
word2vec/
├── main.py          # Full training pipeline (baseline + AFWS)
├── train.py         # Training loop & gradient check
├── config.py        # Hyperparameters
├── data.py          # Corpus loading, vocab, subsampling, neg-table
├── evaluate.py      # Analogy evaluation & nearest-neighbour search
├── pairs.py         # Context-pair generation (standard & AFWS)
├── math_utils.py    # sigmoid / log-sigmoid helpers
├── io_utils.py      # Logging & result saving
├── plot.py          # Training curve plots
├── demo.ipynb       # Interactive demo notebook
├── data/
│   ├── text8                  # Raw corpus (auto-downloaded if missing)
│   └── questions-words.txt    # Google analogy benchmark
└── results/                   # Saved models, plots, logs
```

## Requirements

Python 3.9+ with NumPy (and Matplotlib for plots):

```bash
pip install numpy matplotlib
```

> No deep-learning framework required — everything runs on pure NumPy.

## Running the Full Training Pipeline

```bash
python main.py
```

This will:
1. Download the **text8** corpus automatically if `data/text8` is missing
2. Run a **gradient check** to verify analytic gradients
3. Train the **baseline** Skip-Gram model (20 epochs, dim=300, window=10, neg=5)
4. Evaluate on the Google Word Analogy benchmark (semantic + syntactic)
5. Train the **AFWS** variant (adaptive window sizing for rare words)
6. Save all models, logs, and plots to `results/`

Expected runtime: ~30–60 min on a modern CPU.

### Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `embed_dim` | 300 | Embedding dimensionality |
| `window_size` | 10 | Context window radius |
| `neg_samples` | 5 | Negative samples per pair |
| `min_count` | 5 | Min token frequency for vocab |
| `subsample_t` | 1e-5 | Subsampling threshold |
| `lr_start` | 0.025 | Initial learning rate |
| `lr_min` | 0.0001 | Minimum learning rate |
| `epochs` | 20 | Training epochs |
| `batch_size` | 4096 | Mini-batch size |

## Using Pre-trained Embeddings

Pre-trained models are saved in `results/` as `.npz` files:

```python
import numpy as np

# Best post-processed embeddings (W_in + W_out combined)
data = np.load('results/model_final_best.npz')
W = data['W']  # shape: (71290, 300)

# Raw model weights
data = np.load('results/model_k15_ft.npz')
W_in  = data['W_in']   # center embeddings
W_out = data['W_out']  # context embeddings
```

## Output Files

After training, `results/` will contain:

| File | Description |
|---|---|
| `model_base.npz` | Baseline model weights (`W_in`, `W_out`) |
| `model_afws.npz` | AFWS model weights |
| `model_final_best.npz` | Best combined embeddings (`W`) |
| `config.json` | Saved hyperparameter config |
| `gradient_check.txt` | Finite-difference gradient verification |
| `analogy_results.txt` | Per-category analogy scores |
| `nearest_neighbors.txt` | Nearest-neighbour word lists |
| `base_training_curves.png` | Loss & LR curves (baseline) |
| `phase2_training_curves.png` | Loss & LR curves (AFWS) |
| `full_log.txt` | Complete training log |
