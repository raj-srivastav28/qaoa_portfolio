"""End-to-end quantum smoke test on a tiny instance (N=5, k=2).

Asserts the hand-built QAOA, its optimiser, and the evaluation utilities
agree with brute force. Kept small so pytest stays under ~1 minute.
"""
import numpy as np

from src import brute_force, ising, optimize, qubo
from src.qaoa_circuit import build_qaoa_circuit


def _tiny_instance():
    rng = np.random.default_rng(3)
    n, k = 5, 2
    A = rng.normal(0.1, 0.02, size=(n, n))
    sigma = A @ A.T / n
    np.fill_diagonal(sigma, np.linspace(0.02, 0.06, n))
    mu = rng.uniform(0.0, 0.25, size=n)
    return mu, sigma, n, k


def test_circuit_structure():
    mu, sigma, n, k = _tiny_instance()
    Q, const = qubo.build_qubo(mu, sigma, k, 5.0, 10.0)
    Qs, consts, _ = qubo.scale_qubo(Q, const)
    h, J, _ = ising.qubo_to_ising(Qs, consts)
    qc, gammas, betas = build_qaoa_circuit(h, J, p=2)
    assert qc.num_qubits == n
    assert len(gammas) == 2 and len(betas) == 2
    # depth must grow with p
    qc1, _, _ = build_qaoa_circuit(h, J, p=1)
    assert qc1.depth() < qc.depth()


def test_qaoa_p1_beats_random_sampling():
    mu, sigma, n, k = _tiny_instance()
    Q, const = qubo.build_qubo(mu, sigma, k, 5.0, 10.0)
    Qs, consts, _ = qubo.scale_qubo(Q, const)
    h, J, _ = ising.qubo_to_ising(Qs, consts)
    bf = brute_force.solve_brute_force(Qs, consts, k)
    res = optimize.optimize_qaoa(h, J, p=1, energies=bf["energies"],
                                 optimizer="COBYLA", n_restarts=4,
                                 maxiter=120, seed=5)
    ev = optimize.evaluate_distribution(res["probs"], bf["energies"], k,
                                        bf["e_opt"], bf["opt_masks_int"],
                                        shots=2048, seed=5)
    # QAOA's best sampled solution must be meaningfully better than a random
    # feasible portfolio's mean energy (statistical margin kept generous).
    rng = np.random.default_rng(0)
    rand_e = []
    for _ in range(200):
        x = np.zeros(n, dtype=int)
        x[rng.choice(n, size=k, replace=False)] = 1
        rand_e.append(qubo.evaluate_qubo(x, Qs, consts))
    assert ev["e_best_sampled"] < np.mean(rand_e)
    assert 0 < ev["optimality_ratio"] <= 1.0 + 1e-9
    # distribution must be non-degenerate and constraint-aware
    assert 0.0 < ev["feasible_rate"] <= 1.0
