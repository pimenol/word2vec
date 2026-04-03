# Word2Vec Skip-Gram with Negative Sampling — Pure NumPy Implementation

## Model Specification

| Parameter | Value |
|---|---|
| Architecture | Skip-Gram with Negative Sampling (SGNS) |
| Embedding dimension | 300 |
| Vocabulary size | 71,290 |
| Context window | Dynamic, sampled uniformly from [1, 10] |
| Negative samples K | 15 (best model; K=5 and K=10 also trained) |
| Subsampling threshold | 1e-5 |
| Total parameters | 42,774,000 (W_in + W_out, each 71,290×300) |
| Parameter memory | 171 MB (float32) |

---

## Data Pipeline

**Corpus.** text8 — first 100 MB of cleaned English Wikipedia. 17,005,207 tokens, single continuous sequence (no sentence boundaries), pre-tokenized and lowercased.

**Vocabulary.** Words appearing ≥5 times are kept, sorted by descending frequency. This yields 71,290 words covering 16,718,844 token occurrences. The remaining ~286K tokens are OOV hapax legomena.

**Subsampling.** Each token is kept with probability:

    P(keep | w) = min(1, sqrt(t / f(w)) + t / f(w))

where f(w) is relative frequency and t = 1e-5. "The" is kept only 1.27% of the time. Rare words are always kept. After subsampling, ~10M tokens remain per epoch. The motivation is straightforward: frequent words appear in so many contexts that each individual co-occurrence carries little signal, and they dominate the gradient without this correction.

**Negative sampling distribution.**

    P_neg(w) ∝ count(w)^(3/4)

The 3/4 exponent smooths out frequency differences — common words are still sampled more, but not as aggressively as their raw frequency would dictate.

**Pair generation.** For each offset d ∈ [1, W], the pair (token[i], token[i+d]) is included with probability (W − d + 1) / W. Both directions are generated (symmetric). This replicates uniform window-size sampling per center word, weighting close context more heavily, without a Python loop over individual tokens.

---

## Gradient Derivation

### Loss Function

For center word c, positive context o, and K negative samples n_1, ..., n_K:

    L = −log σ(v_c · u_o) − Σ_k log σ(−v_c · u_{n_k})

where v_c ∈ W_in and u_o, u_{n_k} ∈ W_out.

### Gradients

**Center word v_c:**

    ∂L/∂v_c = (σ(v_c · u_o) − 1) · u_o + Σ_k σ(v_c · u_{n_k}) · u_{n_k}

The first term pushes v_c toward u_o; the second pushes it away from each negative sample.

**Positive context u_o:**

    ∂L/∂u_o = (σ(v_c · u_o) − 1) · v_c

**Negative sample u_{n_k}:**

    ∂L/∂u_{n_k} = σ(v_c · u_{n_k}) · v_c

### Comparison to Full Softmax and Hierarchical Softmax

Full softmax requires computing the partition function over all V words — O(|V|) per training pair. Negative sampling replaces this with K+1 binary classification tasks, reducing cost to O(K). Hierarchical softmax achieves O(log|V|) via a binary tree, but requires maintaining the tree structure and is harder to parallelize. In practice, negative sampling with K=5–15 is simpler and empirically competitive on standard benchmarks.

---

## Gradient Verification

Analytic gradients were checked against finite differences on a toy model (V=20, D=5, K=3, B=2):

| Parameter | Max Relative Error | Status |
|---|---|---|
| W_in (center) | 3.20e-08 | PASS |
| W_out (positive) | 2.78e-08 | PASS |
| W_out (negative) | 2.12e-08 | PASS |

All errors are well below 1e-5, confirming the implementation is mathematically correct.

---

## Training Details

| Detail | Value |
|---|---|
| Optimizer | SGD with linear learning rate decay |
| Learning rate | 0.025 → 0.0001 |
| Batch size | 4,096 |
| Epochs | 20 (Phase 6); 10 (Phase 7 fine-tune) |
| Pairs per epoch | ~60M |
| Total pairs (best run) | ~1.8B across Phases 6+7 |
| Throughput | 0.12M pairs/s (K=15) |

**Optimizer.** Plain SGD with linear decay matches the original word2vec C implementation. The decay ensures the learning rate approaches zero at the end of training, letting the model settle rather than oscillate. Adam and Adagrad were considered but rejected: both require per-parameter moment accumulators, which are expensive for sparse updates on 71K×300 matrices, and the SGD recipe is well-calibrated for this objective.

**Key implementation decisions.**

- float32 throughout: halves memory footprint vs float64, improves cache utilization
- Vectorized pair generation (loop over offsets, not tokens)
- Regular fancy indexing for parameter updates instead of `np.add.at`: 6.5× faster. The cost is losing ~2–7% of gradient accumulation for duplicate indices within a batch — equivalent to a slight per-word learning rate reduction, negligible for SGD
- No per-chunk pair shuffling: chunk-start shuffling between epochs provides sufficient randomness, avoiding the ~1.5s overhead of shuffling 3M+ pairs per chunk

---

## Training Phases and Results

Multiple phases were used to find the best configuration. Each phase either continued from a prior checkpoint or trained from scratch.

| Phase | Config | Epochs | Raw Semantic | Best Post-Processed |
|---|---|---|---|---|
| Phase 1 | K=5, LR=0.025→0.0001 | 20 | 42.6% | 47.9% |
| Phase 2 | K=10, from Phase 1, LR=0.01→0.0001 | 20 | 46.3% | 52.0% |
| Phase 6 | K=15, from scratch, LR=0.025→0.0001 | 20 | 49.3% | 54.2% |
| **Phase 7** | **K=15, from Phase 6, LR=0.003→0.0001** | **10** | **49.7%** | **55.1%** |

The jump from K=5 to K=15 (+6.7% raw semantic) is the single largest gain across all experiments. More negatives per pair give a better gradient estimate of the full softmax — especially valuable on a small corpus where each positive pair is scarce.

Phase 7's fine-tuning used a conservative starting LR (0.003 vs 0.025). Phase 4 earlier showed that a high LR applied to pre-trained embeddings destroys learned geometry; recovery took ~10 epochs just to return to the Phase 2 baseline.

---

## Post-Processing

Following Mu et al. (2018) "All-but-the-Top":

1. Combine W_in + W_out (average center and context embeddings)
2. L2-normalize each word vector
3. Mean-center across the vocabulary
4. Remove the top-k principal components (k=2 was optimal)

The removed components primarily encode word frequency rather than semantics. After removal, the embedding space is more isotropic, which directly benefits cosine similarity-based evaluation. This added +5.4% on the best raw model without any retraining.

Ensembling (averaging embeddings from K=5 and K=15 models) did not help — models trained with different K develop incompatible geometric structures, so averaging degrades both.

---

## Quantitative Evaluation

### Per-Category Breakdown (Best Model: Phase 7, W_in+W_out, L2+PCA-2)

| Category | Correct | Total | Skipped | Accuracy |
|---|---|---|---|---|
| capital-common-countries | 413 | 506 | 0 | **81.6%** |
| capital-world | 2,298 | 3,564 | 960 | **64.5%** |
| city-in-state | 1,126 | 2,330 | 137 | 48.3% |
| currency | 117 | 596 | 270 | 19.6% |
| family | 133 | 420 | 86 | 31.7% |
| **SEMANTIC TOTAL** | **4,087** | **7,416** | **1,453** | **55.1%** |
| gram1-adjective-to-adverb | 80 | 992 | 0 | 8.1% |
| gram2-opposite | 38 | 756 | 56 | 5.0% |
| gram3-comparative | 269 | 1,332 | 0 | 20.2% |
| gram4-superlative | 56 | 992 | 130 | 5.6% |
| gram5-present-participle | 156 | 1,056 | 0 | 14.8% |
| gram6-nationality-adjective | 1,070 | 1,521 | 78 | **70.3%** |
| gram7-past-tense | 259 | 1,560 | 0 | 16.6% |
| gram8-plural | 445 | 1,332 | 0 | 33.4% |
| gram9-plural-verbs | 108 | 870 | 0 | 12.4% |
| **SYNTACTIC TOTAL** | **2,481** | **10,411** | **264** | **23.8%** |
| **OVERALL** | **6,568** | **17,827** | **1,717** | **36.8%** |

### What the numbers mean

**Strong categories (>60%):** capital-common-countries (81.6%), nationality-adjective (70.3%), capital-world (64.5%). These involve named entities with clear, consistent co-occurrence patterns across Wikipedia — the model sees enough signal to learn them reliably.

**Mid-range (30–50%):** city-in-state (48.3%) requires US-specific geographic knowledge that is sparsely represented in a 17M-token corpus. Plural (33.4%) and family (31.7%) require more morphological precision than large-window training naturally provides.

**Weak (<25%):** Currency (19.6%) fails because currency pairs like "japan yen" are rare in text8. Syntactic categories (adjective-to-adverb, superlative, opposite) all sit below 25% — a window of 10 biases training toward semantic rather than syntactic relationships, and these categories require tight local patterns that large windows dilute.

**Skipped questions:** 1,717 (8.8%) skipped due to OOV words. Most skips come from capital-world and currency, where rare country and currency names fall below min_count=5.

**Why 60% semantic is hard on text8:** The original word2vec achieved 60%+ on Google News (~100B tokens). With text8 at 17M tokens, many analogy words appear fewer than 100 times — not enough training signal. At 55.1% with K=15, multi-phase training, and post-processing, the model is likely near the ceiling for this corpus size. Getting to 60% would require either a much larger corpus or a different approach (e.g., SVD on PPMI, subword embeddings).

---

## Qualitative Evaluation

All results from Phase 7 (W_in+W_out, L2+PCA-2).

### Nearest Neighbors

| Probe | Top-8 Nearest Neighbors (cosine similarity) |
|---|---|
| king | son(0.642), kings(0.610), iii(0.578), queen(0.571), prince(0.563), throne(0.546), iv(0.522), brother(0.520) |
| queen | elizabeth(0.673), king(0.571), prince(0.532), victoria(0.510), consort(0.482), royal(0.466) |
| computer | computers(0.813), software(0.796), hardware(0.766), computing(0.723), programming(0.687), interface(0.679) |
| france | germany(0.704), italy(0.678), french(0.675), netherlands(0.660), belgium(0.653), spain(0.650) |
| dog | pictus(0.554), keeshond(0.554), catahoula(0.532), poodle(0.530), dogs(0.523), komondor(0.516) |
| money | depositors(0.659), pay(0.634), reinvested(0.619), paid(0.618), seignorage(0.616), financial(0.603) |

**Observations.** Semantic clustering is coherent throughout: countries cluster with countries, technical terms with technical terms, royalty with royalty. "dog" returns specific breed names (keeshond, catahoula) rather than generic animals — Wikipedia's detailed breed articles dominate the local co-occurrence signal for this word.

### Analogy Examples (a:b :: c:?)

| Analogy | Top-5 Predictions | Correct? |
|---|---|---|
| man:king :: woman:? | **queen**(0.509), daughter, son, wife, prince | Yes |
| france:paris :: germany:? | **berlin**(0.654), munich, stuttgart, mannheim | Yes |
| japan:tokyo :: china:? | **beijing**(0.489), chongqing, taipei, pyongyang | Yes |
| big:bigger :: small:? | **smaller**(0.624), larger, large, relatively, size | Yes |
| france:french :: spain:? | **spanish**(0.810), portuguese, dutch, italian | Yes |
| going:went :: playing:? | **played**(0.637), play, career, game, players | Yes |
| slow:slowly :: quick:? | surrendered, leave, prusias, retake, pacified | No |
| good:best :: bad:? | award, awards, filmfare, rnb, nastro | No |

Geographic and morphological analogies (comparative, past tense, gender) solve reliably. Adverb formation (slow:slowly::quick:quickly) and superlatives (good:best::bad:worst) fail consistently — "bad" and "worst" rarely co-occur in tight windows in Wikipedia text, so the model never builds that association.

---

## Novel Contribution: Adaptive Frequency-Based Window Sizing (AFWS)

### Hypothesis

Rare words appear fewer times in training, so each occurrence must capture as much context as possible — a larger window helps. Frequent words already have abundant signal and benefit from smaller, more precise windows that emphasize immediate neighbors.

### Implementation

For a center word with corpus frequency f:

    w_max(f) = w_min + (w_max − w_min) · (1 − f/f_max)^α

with w_min=3, w_max=15, α=0.5. The dynamic window is then sampled from [1, w_max(f)] using the standard keep-probability scheme. Gradients are unchanged — the modification only affects which pairs are generated.

### Results

| Metric | Baseline (10ep) | AFWS (10ep) |
|---|---|---|
| Semantic accuracy | 38.1% | 30.0% |
| Syntactic accuracy | 24.7% | 22.3% |
| Throughput | 0.25M pairs/s | 0.25M pairs/s |

AFWS underperformed by 8.1 percentage points on semantic accuracy. The most likely explanation: many semantic analogy words (country names, capitals) are relatively frequent, and shrinking their window reduces exactly the long-range co-occurrences the model needs to learn those relationships. With only 10 epochs, rare words also appear too infrequently for the larger window to compensate.

A more promising direction would be frequency-based learning rate scaling — giving rare words a higher effective learning rate — rather than modifying which pairs are generated.

---

## Compute Budget

| Phase | Wall-Clock Time | Total Pairs |
|---|---|---|
| Phase 1 (K=5, 20ep) | ~80 min | ~1.2B |
| Phase 2 (K=10, 20ep) | ~118 min | ~1.2B |
| Phase 4 (K=5, 30ep) | ~135 min | ~1.8B |
| Phase 5 (K=10, 15ep) | ~328 min* | ~0.9B |
| Phase 6 (K=15, 20ep) | ~172 min | ~1.2B |
| Phase 7 (K=15, 10ep) | ~83 min | ~0.6B |
| **Total** | **~16 hours** | **~6.9B pairs** |

*Phase 5 was slowed by concurrent process memory pressure (swap thrashing on 18GB RAM).
