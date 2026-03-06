# Word2Vec Skip-Gram with Negative Sampling - Pure NumPy Implementation

## Model Specification

| Parameter | Value |
|---|---|
| Architecture | Skip-Gram with Negative Sampling (SGNS) |
| Embedding dimension | 300 |
| Vocabulary size | 71,290 |
| Context window | Dynamic, sampled uniformly from [1, 10] |
| Negative samples K | 15 |
| Subsampling threshold | 1e-5 |
| Total parameters | 42,774,000 (W_in + W_out, each 71,290×300) |
| Parameter memory | 171 MB (float32) |

---

## Data Pipeline

**Corpus.** text8 - first 100 MB of cleaned English Wikipedia. 17,005,207 tokens, single continuous sequence, pre-tokenized and lowercased.

**Vocabulary.** Words appearing ≥5 times, sorted by descending frequency: 71,290 words covering 16,718,844 token occurrences. The remaining ~286K tokens are OOV hapax legomena.

**Subsampling.** Each token is kept with probability:

    P(keep | w) = min(1, sqrt(t / f(w)) + t / f(w))

where f(w) is relative frequency and t = 1e-5. "The" is kept only 1.27% of the time; rare words are always kept. After subsampling, ~10M tokens remain per epoch. Without this, frequent words dominate the gradient without contributing proportional signal.

**Negative sampling distribution.**

    P_neg(w) ∝ count(w)^(3/4)

**Pair generation.** For each offset d ∈ [1, W], the pair (token[i], token[i+d]) is included with probability (W − d + 1) / W, both directions. This weights closer context more heavily without a Python loop over individual tokens.

---

## Gradient Derivation

### Loss Function

For center word c, positive context o, and K=15 negative samples n_1, ..., n_K:

    L = −log σ(v_c · u_o) − Σ_k log σ(−v_c · u_{n_k})

where v_c ∈ W_in and u_o, u_{n_k} ∈ W_out.

### Gradients

**Center word v_c:**

    ∂L/∂v_c = (σ(v_c · u_o) − 1) · u_o + Σ_k σ(v_c · u_{n_k}) · u_{n_k}

**Positive context u_o:**

    ∂L/∂u_o = (σ(v_c · u_o) − 1) · v_c

**Negative sample u_{n_k}:**

    ∂L/∂u_{n_k} = σ(v_c · u_{n_k}) · v_c

---

## Gradient Verification

Verified against finite differences on a toy model (V=20, D=5, K=3, B=2):

| Parameter | Max Relative Error | Status |
|---|---|---|
| W_in (center) | 3.20e-08 | PASS |
| W_out (positive) | 2.78e-08 | PASS |
| W_out (negative) | 2.12e-08 | PASS |

All errors are well below 1e-5.

---

## Training

| Detail | Value |
|---|---|
| Optimizer | SGD with linear learning rate decay |
| Phase 1 LR | 0.025 → 0.0001, 20 epochs from scratch |
| Phase 2 LR | 0.003 → 0.0001, 10 epochs fine-tune |
| Batch size | 4,096 |
| Pairs per epoch | ~60M |
| Throughput | 0.12M pairs/s |
| Total wall-clock | ~4.2 hours (both phases) |

Training used two phases: an initial run from scratch followed by a conservative fine-tune with a low starting LR. The low LR in phase 2 is deliberate - a high LR applied to pre-trained embeddings disrupts learned geometry.

float32 embeddings throughout (halves memory vs float64). Regular fancy indexing for parameter updates instead of `np.add.at`: 6.5× faster, at the cost of losing ~2–7% of gradient accumulation for duplicate indices within a batch - negligible for SGD.

---

## Post-Processing

Following Mu et al. (2018) "All-but-the-Top":

1. Combine W_in + W_out
2. L2-normalize each word vector
3. Mean-center across the vocabulary
4. Remove top-2 principal components

The removed components encode word frequency rather than semantics. After removal the embedding space is more isotropic, which directly benefits cosine similarity evaluation. This step added +5.4% semantic accuracy without retraining.

---

## Results

### Analogy Benchmark

| Category | Accuracy |
|---|---|
| capital-common-countries | **81.6%** |
| gram6-nationality-adjective | **70.3%** |
| capital-world | **64.5%** |
| city-in-state | 48.3% |
| gram8-plural | 33.4% |
| family | 31.7% |
| gram3-comparative | 20.2% |
| currency | 19.6% |
| gram7-past-tense | 16.6% |
| gram5-present-participle | 14.8% |
| gram1-adjective-to-adverb | 8.1% |
| gram4-superlative | 5.6% |
| gram2-opposite | 5.0% |
| **SEMANTIC TOTAL** | **55.1%** |
| **SYNTACTIC TOTAL** | **23.8%** |
| **OVERALL** | **36.8%** |

1,717 questions (8.8%) skipped due to OOV - mostly in capital-world and currency, where rare country and currency names fall below min_count=5.

**Strong categories (>60%)** involve named entities with dense, consistent co-occurrence in Wikipedia. The model sees enough signal to learn them reliably.

**Syntactic categories (<25%)** fail because a window of 10 biases training toward semantic relationships. Tight morphological patterns (adverb formation, superlatives) require local context that large windows dilute. "bad" and "worst" also rarely co-occur within 10 tokens in Wikipedia text.

**Currency (19.6%)** fails because pairs like "japan yen" are simply rare in text8.

### Nearest Neighbors

| Probe | Top Neighbors (cosine similarity) |
|---|---|
| king | son(0.642), kings(0.610), queen(0.571), prince(0.563), throne(0.546) |
| computer | computers(0.813), software(0.796), hardware(0.766), computing(0.723) |
| france | germany(0.704), italy(0.678), french(0.675), netherlands(0.660) |
| music | musical(0.797), songs(0.716), musicians(0.712), dance(0.681) |
| dog | pictus(0.554), keeshond(0.554), catahoula(0.532), poodle(0.530) |

Semantic clustering is coherent throughout. "dog" returns breed names rather than generic animals - Wikipedia's detailed breed articles dominate local co-occurrence for this word.

### Analogy Examples (a:b :: c:?)

| Analogy | Top Prediction | Correct? |
|---|---|---|
| man:king :: woman:? | queen (0.509) | Yes |
| france:paris :: germany:? | berlin (0.654) | Yes |
| japan:tokyo :: china:? | beijing (0.489) | Yes |
| big:bigger :: small:? | smaller (0.624) | Yes |
| france:french :: spain:? | spanish (0.810) | Yes |
| going:went :: playing:? | played (0.637) | Yes |
| slow:slowly :: quick:? | surrendered | No |
| good:best :: bad:? | award | No |

---

## Adaptive Frequency-Based Window Sizing

### Hypothesis

Rare words have fewer training examples, so each occurrence should capture more context - larger window. Frequent words have abundant signal and benefit from smaller, more precise windows.

### Implementation

    w_max(f) = w_min + (w_max − w_min) · (1 − f/f_max)^α

with w_min=3, w_max=15, α=0.5. Gradients unchanged - only pair generation is affected.

### Results

| Metric | Baseline | AFWS |
|---|---|---|
| Semantic accuracy | 38.1% | 30.0% |
| Syntactic accuracy | 24.7% | 22.3% |

AFWS underperformed by 8.1 percentage points. Many semantic analogy words (country names, capitals) are relatively frequent - shrinking their window removes exactly the long-range co-occurrences needed to learn those relationships. The hypothesis is sound, but window-size adaptation is not the right mechanism. Frequency-based learning rate scaling would be a more targeted approach.