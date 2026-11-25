"""Cosine similarity, threshold selection and Top-K retrieval.

cos(a, b) = (a . b) / (|a| |b|). For L2-normalised vectors it is just the dot product.
Scores are SIMILARITIES, not probabilities: 0.7 does not mean "70% chance of duplicate".
"""
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score


def normalize_rows(m):
    if sparse.issparse(m):
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        return sparse.diags(1 / np.maximum(norms, 1e-12)) @ m
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def cosine_scores(query, matrix) -> np.ndarray:
    """Similarity of one query vector (1 x d) with every row of matrix (n x d)."""
    q, m = normalize_rows(query), normalize_rows(matrix)
    s = m @ q.T
    return np.asarray(s.todense() if sparse.issparse(s) else s).ravel()


def paired_cosine(a, b) -> np.ndarray:
    """Similarity of row i of a with row i of b."""
    a, b = normalize_rows(a), normalize_rows(b)
    if sparse.issparse(a):
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return np.sum(a * b, axis=1)


def top_k(scores: np.ndarray, k: int = 5, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Indices and scores of the k highest scores. mask=False rows are skipped."""
    s = np.asarray(scores, dtype=float).copy()
    if mask is not None:
        s[~mask] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx], kind="stable")]
    return [(int(i), float(s[i])) for i in idx]


def predict(scores, threshold: float) -> np.ndarray:
    return (np.asarray(scores) >= threshold).astype(int)


def select_threshold(scores, labels) -> tuple[float, float]:
    """Threshold with the best F1. Use VALIDATION data only.

    Candidates are midpoints between neighbouring distinct scores. Ties go to the higher threshold.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    u = np.unique(scores)
    cands = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    if len(cands) > 2000:
        cands = np.quantile(scores, np.linspace(0, 1, 2001))
    best_t, best_f1 = float(cands[0]), -1.0
    for t in cands:
        f1 = f1_score(labels, predict(scores, t), zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and t > best_t):
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1
