"""Classical baselines -- the honest comparison (20% of the score).

Baselines:
  1. exact enumeration       -- already the ground truth (brute_force.py)
  2. greedy top-k Sharpe     -- pick k assets with best mu_i / sqrt(Sigma_ii)
  3. local search            -- iterative single-swap improvements on the QUBO
  4. simulated annealing     -- same moves, stochastic acceptance
  5. continuous Markowitz    -- SLSQP on continuous weights, then round to k
                                assets (the classic 'rounding hack' baseline)

All return an integer mask x; scoring is done by the SAME evaluate_qubo()
used for QAOA, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np

from . import brute_force


def greedy_topk_sharpe(mu: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
    sharpe = mu / np.sqrt(np.diag(sigma))
    x = np.zeros(len(mu), dtype=int)
    x[np.argsort(-sharpe)[:k]] = 1
    return x


def _energy(x, Q, const) -> float:
    x = np.asarray(x, dtype=float)
    iu = np.triu_indices_from(Q, k=1)
    return float(np.sum(np.diag(Q) * x) + np.sum(Q[iu] * x[iu[0]] * x[iu[1]]) + const)


def local_search(Q: np.ndarray, const: float, k: int, x0: np.ndarray,
                 max_iter: int = 200) -> np.ndarray:
    """Best-improvement single swaps: drop one selected, add one unselected."""
    n = len(Q)
    x = np.asarray(x0, dtype=int).copy()
    if x.sum() != k:
        raise ValueError("x0 must select exactly k assets")
    e = _energy(x, Q, const)
    for _ in range(max_iter):
        best_move, best_e = None, e
        sel = np.flatnonzero(x)
        unsel = np.flatnonzero(1 - x)
        for a in sel:
            for b in unsel:
                y = x.copy()
                y[a], y[b] = 0, 1
                ey = _energy(y, Q, const)
                if ey < best_e - 1e-12:
                    best_move, best_e = (a, b), ey
        if best_move is None:
            break
        x[best_move[0]], x[best_move[1]] = 0, 1
        e = best_e
    return x


def simulated_annealing(Q: np.ndarray, const: float, k: int, seed: int = 0,
                        iters: int = 4000, t0: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(Q)
    x = np.zeros(n, dtype=int)
    x[rng.choice(n, size=k, replace=False)] = 1
    e = _energy(x, Q, const)
    best_x, best_e = x.copy(), e
    for t in range(iters):
        temp = t0 * (1 - t / iters) + 1e-4
        a = rng.choice(np.flatnonzero(x))
        b = rng.choice(np.flatnonzero(1 - x))
        y = x.copy()
        y[a], y[b] = 0, 1
        ey = _energy(y, Q, const)
        if ey < e or rng.random() < np.exp(-(ey - e) / temp):
            x, e = y, ey
            if e < best_e:
                best_x, best_e = x.copy(), e
    return best_x


def markowitz_rounded(mu: np.ndarray, sigma: np.ndarray, k: int,
                      gamma_risk: float) -> np.ndarray:
    """Continuous mean-variance (no cardinality) -> round to k names.

    Note for the write-up: this is NOT the same problem -- the continuous
    relaxation is convex and easy; rounding gives no optimality guarantee,
    which is precisely the gap the discrete (QUBO) formulation addresses.
    """
    from scipy.optimize import minimize

    n = len(mu)

    def obj(w):
        return -(mu @ w) + 0.5 * gamma_risk * (w @ sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, np.ones(n) / n, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 500})
    w = res.x
    x = np.zeros(n, dtype=int)
    x[np.argsort(-w)[:k]] = 1
    return x


def run_all_baselines(Q, const, mu, sigma, k, gamma_risk, energies, e_opt,
                      seed=42) -> dict:
    """Evaluate every baseline with the same ruler as QAOA.

    `energies` is the full 2^N array from brute_force.enumerate_energies
    (needed for the robust optimality score).
    """
    out = {}
    x = greedy_topk_sharpe(mu, sigma, k)
    out["greedy"] = {"x": x.tolist(), "e": _energy(x, Q, const)}
    x = local_search(Q, const, k, x)
    out["local_search"] = {"x": x.tolist(), "e": _energy(x, Q, const)}
    x = simulated_annealing(Q, const, k, seed=seed)
    out["annealing"] = {"x": x.tolist(), "e": _energy(x, Q, const)}
    x = markowitz_rounded(mu, sigma, k, gamma_risk)
    out["markowitz_rounded"] = {"x": x.tolist(), "e": _energy(x, Q, const)}

    for name, d in out.items():
        d["optimality_ratio"] = brute_force.optimality_ratio(energies, d["e"])
        d["selected"] = [int(i) for i in np.flatnonzero(d["x"])]
    out["enumeration"] = {"e": float(e_opt), "optimality_ratio": 1.0,
                          "note": "exact 2^N search"}
    return out
