"""QAOA optimisation loop: angle search + evaluation of solution quality.

Design notes (also write-up material):
  * The optimisation cost is the EXPECTED energy <H> computed from exact
    statevector probabilities. This is a "statistical shortcut" only in the
    sense that we use the ideal probability distribution instead of finite
    shots -- the circuit itself is the real p-layer QAOA circuit. (Finite-shot
    runs happen in noise.py / hardware.py.)
  * All energies live in a precomputed 2^N lookup (brute_force.enumerate_energies),
    so cost evaluation is a dot product. Index convention: integer m <-> x_i = (m>>i)&1,
    matching Qiskit's little-endian probabilities ordering.
  * Optimisers: COBYLA (scipy) and a hand-rolled SPSA, both run over
    `n_restarts` random initial points; the best run wins.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector

from . import brute_force
from .qaoa_circuit import build_qaoa_circuit


# ---------------------------------------------------------------- helpers --
def expected_energy(probs: np.ndarray, energies: np.ndarray) -> float:
    return float(probs @ energies)


def probs_from_circuit(qc, theta, gammas, betas) -> np.ndarray:
    """Exact ideal distribution of the (measurement-free) circuit.

    IMPORTANT: parameters are bound through an EXPLICIT dict. Relying on
    list order would silently map values to the wrong angles because
    qc.parameters sorts alphabetically ('beta[0]' < 'gamma[0]').
    """
    mapping = {**{g: float(v) for g, v in zip(gammas, theta[: len(gammas)])},
               **{b: float(v) for b, v in zip(betas, theta[len(gammas):])}}
    bound = qc.assign_parameters(mapping)
    return np.asarray(Statevector(bound).probabilities())


def sample_from_probs(probs: np.ndarray, shots: int, seed: int) -> tuple[dict[int, int], np.ndarray]:
    """Draw `shots` samples from an ideal distribution (documented shortcut)."""
    rng = np.random.default_rng(seed)
    counts_arr = rng.multinomial(shots, probs)
    nz = np.flatnonzero(counts_arr)
    return {int(i): int(counts_arr[i]) for i in nz}, counts_arr


# ---------------------------------------------------------------- SPSA -----
def spsa(f, x0, steps: int, a: float = 0.25, c: float = 0.15,
         A: float = 10.0, alpha: float = 0.602, gamma_: float = 0.101, seed: int = 0):
    """Minimal SPSA. Returns (x_best, history)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    history = [float(f(x))]
    for t in range(steps):
        at = a / (t + 1 + A) ** alpha
        ct = c / (t + 1) ** gamma_
        delta = rng.choice([-1.0, 1.0], size=len(x))
        fp, fm = f(x + ct * delta), f(x - ct * delta)
        grad = (fp - fm) / (2.0 * ct) * delta
        x = x - at * grad
        history.append(float(f(x)))
    return x, history


# ------------------------------------------------------------- core API ----
def optimize_qaoa(
    h: np.ndarray,
    J: dict[tuple[int, int], float],
    p: int,
    energies: np.ndarray,
    optimizer: str = "COBYLA",
    n_restarts: int = 5,
    maxiter: int = 300,
    seed: int = 42,
    polish: bool = False,
) -> dict:
    """Optimise QAOA angles.

    Returns dict with:
      theta, energy, probs          -- best restart
      history                       -- its convergence trace (running min)
      all_restarts                  -- [{theta, energy, history}, ...]
      qc, gammas, betas             -- parameterised circuit

    `polish=True` runs a final Powell refinement from COBYLA's best point;
    measured on the N=8 instance it cut the expected energy by ~3-4x and
    raised P(optimal) by ~10x, at the cost of one extra local optimisation.
    """
    qc, gammas, betas = build_qaoa_circuit(h, J, p)
    rng = np.random.default_rng(seed)

    def cost(theta):
        probs = probs_from_circuit(qc, theta, gammas, betas)
        return expected_energy(probs, energies)

    all_restarts = []
    for r in range(n_restarts):
        theta0 = rng.uniform(0.0, 1.0, size=p).tolist() + \
                 rng.uniform(0.0, np.pi / 2, size=p).tolist()
        trace: list[float] = []
        if optimizer.upper() == "COBYLA":
            def cost_tracked(theta_, _trace=trace):
                val = cost(theta_)
                _trace.append(val)
                return val

            res = minimize(cost_tracked, theta0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.3, "tol": 1e-6})
            theta_best = np.array(res.x)
            trace = np.minimum.accumulate(trace).tolist()
        elif optimizer.upper() == "SPSA":
            def cost_spsa(theta_, _trace=trace):
                val = cost(theta_)
                _trace.append(val)
                return val

            theta_best, _ = spsa(cost_spsa, theta0, steps=maxiter, seed=seed + r)
            trace = np.minimum.accumulate(trace).tolist()
        else:
            raise ValueError(f"unknown optimizer {optimizer}")

        all_restarts.append({
            "theta": np.asarray(theta_best, dtype=float),
            "energy": float(cost(theta_best)),
            "history": trace,
        })

    best = min(all_restarts, key=lambda d: d["energy"])
    if polish:
        polish_trace: list[float] = []

        def cost_tracked_polish(theta_, _pt=polish_trace):
            val = cost(theta_)
            _pt.append(val)
            return val

        pol = minimize(cost_tracked_polish, best["theta"], method="Powell",
                       options={"maxiter": 2 * maxiter, "xtol": 1e-4,
                                "ftol": 1e-6})
        if pol.fun < best["energy"]:
            best = {"theta": np.asarray(pol.x, dtype=float),
                    "energy": float(pol.fun),
                    "history": best["history"]
                              + np.minimum.accumulate(polish_trace).tolist()}
    probs = probs_from_circuit(qc, best["theta"], gammas, betas)
    return {
        "theta": best["theta"],
        "energy": best["energy"],
        "history": best["history"],
        "all_restarts": all_restarts,
        "probs": probs,
        "qc": qc,
        "gammas": gammas,
        "betas": betas,
    }


# ------------------------------------------------------------ evaluation ---
def evaluate_distribution(probs: np.ndarray, energies: np.ndarray, k: int,
                          e_opt: float, opt_masks_int: list[int],
                          shots: int, seed: int = 0) -> dict:
    """All the numbers judges ask for, from a distribution over bitstrings."""
    n = int(np.log2(len(probs)))
    counts, _ = sample_from_probs(probs, shots, seed)

    e_best_sampled = min(energies[m] for m in counts)
    p_opt = float(sum(probs[m] for m in opt_masks_int))
    n = int(np.log2(len(probs)))
    sums = np.array([bin(m).count("1") for m in range(len(probs))])
    feasible_states = np.flatnonzero(sums == k)
    feasible_rate = float(np.sum(probs[feasible_states]))
    best_sampled_int = min(counts, key=lambda m: energies[m])

    return {
        "e_best_sampled": float(e_best_sampled),
        "best_x": brute_force.int_to_bits(best_sampled_int, n),
        "best_x_int": int(best_sampled_int),
        "optimality_ratio": brute_force.optimality_ratio(energies, float(e_best_sampled)),
        "prob_optimal": p_opt,
        "feasible_rate": feasible_rate,
        "n_unique_sampled": len(counts),
        "shots": shots,
    }
