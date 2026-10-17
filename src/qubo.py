r"""QUBO formulation of cardinality-constrained portfolio selection.

Decision variables
    x_i in {0,1}   -- 1 if stock i is selected, 0 otherwise.

Objective (minimise)
    H(x) = -mu.x + (gamma/2) x.T Sigma x  +  P (sum_i x_i - k)^2
           \______ return ______/ \___ risk ___/   \____ constraint ____/

Because x_i^2 = x_i for binaries, H is exactly a quadratic form
    H(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j  +  const

with (derived by expanding the square and using x_i^2 = x_i):
    Q_ii = -mu_i + (gamma/2) Sigma_ii + P (1 - 2k)
    Q_ij = gamma Sigma_ij + 2P            (i < j, each pair counted once)
    const = P k^2

The golden unit test (tests/test_qubo.py) checks evaluate_qubo against a
direct evaluation of the objective for random x, and that the optimum of a
toy instance is feasible.
"""
from __future__ import annotations

import numpy as np


def build_qubo(
    mu: np.ndarray, sigma: np.ndarray, k: int, gamma_risk: float, penalty: float
) -> tuple[np.ndarray, float]:
    """Return (Q, const) with H(x) = x^T-style QUBO as documented above."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -mu[i] + 0.5 * gamma_risk * sigma[i, i] + penalty * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = gamma_risk * sigma[i, j] + 2.0 * penalty
    const = penalty * k * k
    return Q, const


def evaluate_qubo(x: np.ndarray, Q: np.ndarray, const: float) -> float:
    """H(x) for a binary vector x (list/array of 0/1)."""
    x = np.asarray(x, dtype=float)
    lin = np.sum(np.diag(Q) * x)
    iu = np.triu_indices_from(Q, k=1)
    quad = np.sum(Q[iu] * x[iu[0]] * x[iu[1]])
    return float(lin + quad + const)


def portfolio_metrics(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, gamma_risk: float
) -> dict:
    """Finance-level reporting (no penalty term here!).

    The selected stocks are treated as an EQUAL-WEIGHTED portfolio
    (w_i = 1/k for selected names), so return and volatility are the usual
    annualised per-unit-invested quantities a finance reader expects.
    """
    x = np.asarray(x, dtype=float)
    k = x.sum()
    if k == 0:
        return {"return": 0.0, "variance": 0.0, "vol": 0.0,
                "objective": 0.0, "n_selected": 0}
    w = x / k
    ret = float(mu @ w)
    risk = float(w @ sigma @ w)
    return {
        "return": ret,
        "variance": risk,
        "vol": float(np.sqrt(risk)),
        "objective": ret - 0.5 * gamma_risk * risk,   # maximise this
        "n_selected": int(round(k)),
    }


def scale_qubo(Q: np.ndarray, const: float, target: float = 1.0):
    """Rescale so max|Q| -> target (default 1.0).

    QAOA's angle search behaves far better when QUBO coefficients are O(1).
    Scaling is linear, so argmin is unchanged and every optimality ratio is
    invariant. Keep ONE scaled (Q, const) and use it for ising, brute force
    and QAOA alike.
    """
    m = float(np.max(np.abs(Q)))
    if m == 0:
        return Q.copy(), const, 1.0
    s = target / m
    return Q * s, const * s, s
