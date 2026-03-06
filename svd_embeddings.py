"""SVD-based embeddings from explicit PPMI matrix (Levy & Goldberg 2014).

Builds a word-word co-occurrence matrix from the corpus, converts it to
Shifted Positive PMI (SPPMI), and applies truncated SVD to obtain embeddings.
All operations use pure NumPy.
"""

import time

import numpy as np


def build_cooccurrence_matrix(corpus_ids, vocab_size, window, weighted=True):
    """
    Build a dense co-occurrence matrix from the corpus.

    If weighted=True, each co-occurrence at distance d contributes 1/d.
    Otherwise each contributes 1.

    Returns:
        C: (V, V) float64 co-occurrence matrix (symmetric)
    """
    print(f"    Building co-occurrence matrix (V={vocab_size}, "
          f"window={window}, weighted={weighted})...")
    t0 = time.time()
    n = len(corpus_ids)
    # Use float64 for accumulation precision
    C = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    # Process in chunks for efficiency
    chunk_size = 100_000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = corpus_ids[start:end]
        for d in range(1, window + 1):
            if d >= len(chunk):
                break
            centers = chunk[:len(chunk) - d]
            contexts = chunk[d:]
            weight = 1.0 / d if weighted else 1.0
            # Accumulate both directions
            np.add.at(C, (centers, contexts), weight)
            np.add.at(C, (contexts, centers), weight)

        if (start // chunk_size) % 20 == 0 and start > 0:
            elapsed = time.time() - t0
            pct = start / n * 100
            print(f"      {pct:.0f}% ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    nnz = np.count_nonzero(C)
    print(f"    Done. {nnz:,} non-zero entries ({elapsed:.0f}s)")
    return C


def compute_sppmi(C, neg_k=15):
    """
    Compute Shifted Positive PMI from co-occurrence matrix.

    SPPMI[i,j] = max(PMI[i,j] - log(k), 0)
    where PMI[i,j] = log(C[i,j] * |D| / (sum_j C[i,j] * sum_i C[i,j]))
    and k is the number of negative samples.

    Args:
        C: (V, V) co-occurrence matrix
        neg_k: negative sample count for the shift

    Returns:
        SPPMI: (V, V) shifted positive PMI matrix
    """
    print(f"    Computing SPPMI (k={neg_k})...")
    t0 = time.time()

    D = C.sum()  # total co-occurrences
    if D == 0:
        return np.zeros_like(C)

    row_sums = C.sum(axis=1)  # (V,)
    col_sums = C.sum(axis=0)  # (V,)

    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)

    # PMI = log(C[i,j] * D / (row_sums[i] * col_sums[j]))
    # SPPMI = max(PMI - log(k), 0)
    # Do this in chunks to manage memory for large V
    V = C.shape[0]
    SPPMI = np.zeros_like(C)

    log_shift = np.log(neg_k)

    # Process in row chunks to avoid V*V temporary arrays
    chunk = 1000
    for i in range(0, V, chunk):
        end = min(i + chunk, V)
        block = C[i:end]  # (chunk, V)
        # Only compute where C > 0
        mask = block > 0
        if not mask.any():
            continue
        # PMI for non-zero entries
        pmi = np.zeros_like(block)
        pmi[mask] = (np.log(block[mask] * D)
                     - np.log(row_sums[i:end, None] * np.ones((1, V)))[mask]
                     - np.log(col_sums[None, :] * np.ones((end - i, 1)))[mask])
        # Shift and clip
        SPPMI[i:end] = np.maximum(pmi - log_shift, 0.0)

    elapsed = time.time() - t0
    nnz = np.count_nonzero(SPPMI)
    print(f"    Done. {nnz:,} non-zero SPPMI entries ({elapsed:.0f}s)")
    return SPPMI


def svd_embeddings(M, dim=300, power=0.5):
    """
    Compute truncated SVD embeddings from a matrix M.

    Embeddings = U_d * Sigma_d^power

    For large V, we use the eigendecomposition trick:
    M M^T = U Sigma^2 U^T, so we compute eigh of M M^T.

    Args:
        M: (V, V) matrix (e.g. SPPMI)
        dim: embedding dimension
        power: exponent for singular values (0=U only, 0.5=balanced, 1=full)

    Returns:
        W: (V, dim) embedding matrix
    """
    print(f"    Computing SVD (dim={dim}, power={power})...")
    t0 = time.time()
    V = M.shape[0]

    # For symmetric matrices, eigendecomposition is equivalent and faster
    # M = U S U^T for symmetric M
    # Use: M M^T has same eigenvectors, eigenvalues are S^2
    # Since SPPMI is symmetric, we can use eigh directly
    print(f"      Using eigendecomposition on {V}x{V} symmetric matrix...")
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # eigh returns eigenvalues in ascending order; we want the largest
    idx = np.argsort(-np.abs(eigenvalues))[:dim]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Eigenvalues of SPPMI should be non-negative (it's PSD after clipping)
    # but numerical issues can make some slightly negative
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Embeddings: eigenvectors * eigenvalues^power
    # (analogous to U * Sigma^power from SVD)
    W = eigenvectors * np.power(eigenvalues + 1e-10, power)[None, :]

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.0f}s). Embedding shape: {W.shape}")
    return W.astype(np.float32)


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

    # PCA: compute top-k components
    # For efficiency, use covariance matrix approach
    cov = Wn.T @ Wn / Wn.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Top components (largest eigenvalues, returned in ascending order)
    top_components = eigenvectors[:, -n_components:]  # (D, k)

    # Remove projections onto top components
    projections = Wn @ top_components  # (V, k)
    Wn -= projections @ top_components.T

    return Wn.astype(np.float32)
