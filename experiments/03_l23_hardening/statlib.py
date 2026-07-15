"""
Experiment 03 — self-contained statistics (numpy only).

Everything analyze.py needs beyond the LMM (which uses statsmodels): rank
correlations, partial correlation, Cliff's delta, bootstrap CIs, Benjamini-
Hochberg FDR, and the inter-rater trio (quadratic-weighted Cohen's kappa, Gwet
AC2, percent agreement). Kept dependency-light and matching Exp 01's loop.py
conventions so the numbers are comparable.
"""

from __future__ import annotations

import numpy as np


# ------------------------------ rank correlation -------------------------------


def ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def spearman_perm_p(x: np.ndarray, y: np.ndarray, n: int = 5000, seed: int = 0) -> float:
    obs = abs(spearman(x, y))
    rng = np.random.default_rng(seed)
    yc = np.asarray(y, float).copy()
    hits = 0
    for _ in range(n):
        rng.shuffle(yc)
        if abs(spearman(x, yc)) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x,y controlling for z (rank-based)."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom else 0.0


def partial_spearman_ci(x, y, z, n: int = 10000, seed: int = 0):
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    rng = np.random.default_rng(seed)
    m = len(x)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(partial_spearman(x[idx], y[idx], z[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------- effect sizes --------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_delta_ci(a, b, n: int = 5000, seed: int = 0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [cliffs_delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------- multiple comparisons ------------------------------


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Return BH-adjusted q-values, order-aligned with the input p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = p.argsort()
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


# ------------------------------ inter-rater ------------------------------------


def _confusion(a, b, q: int) -> np.ndarray:
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((q, q))
    for i, j in zip(a, b):
        O[i, j] += 1
    n = O.sum()
    return O / n if n else O


def _quadratic_weights(q: int, agreement: bool) -> np.ndarray:
    idx = np.arange(q)
    d = (idx[:, None] - idx[None, :]) ** 2 / (max(q - 1, 1) ** 2)
    return (1 - d) if agreement else d


def weighted_cohens_kappa(a, b, q: int) -> float:
    """Quadratic-weighted Cohen's kappa. q = number of ordinal categories."""
    if len(a) == 0:
        return float("nan")
    O = _confusion(a, b, q)
    d = _quadratic_weights(q, agreement=False)
    row, col = O.sum(1), O.sum(0)
    E = np.outer(row, col)
    num, den = (d * O).sum(), (d * E).sum()
    return float(1 - num / den) if den else float("nan")


def gwet_ac2(a, b, q: int) -> float:
    """Gwet's AC2 with quadratic (agreement) weights; robust to skewed marginals."""
    if len(a) == 0:
        return float("nan")
    O = _confusion(a, b, q)
    w = _quadratic_weights(q, agreement=True)
    pa = (w * O).sum()
    pi = (O.sum(1) + O.sum(0)) / 2  # mean marginal per category
    Tw = w.sum()
    if q < 2:
        return float("nan")
    pe = (Tw / (q * (q - 1))) * (pi * (1 - pi)).sum()
    return float((pa - pe) / (1 - pe)) if pe != 1 else float("nan")


def percent_agreement(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if len(a) else float("nan")
