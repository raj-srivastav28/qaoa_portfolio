"""Baseline sanity tests: no classical heuristic may beat exact enumeration."""
import numpy as np

from src import baselines, brute_force, qubo

RNG = np.random.default_rng(19)


def _instance(n=9):
    A = RNG.normal(0.1, 0.02, size=(n, n))
    sigma = A @ A.T / n
    np.fill_diagonal(sigma, np.linspace(0.02, 0.08, n))
    mu = RNG.uniform(-0.02, 0.28, size=n)
    return mu, sigma, n


def test_baselines_feasible_and_never_beat_exact():
    mu, sigma, n = _instance()
    k = 3
    Q, const = qubo.build_qubo(mu, sigma, k, 5.0, 10.0)
    Qs, consts, _ = qubo.scale_qubo(Q, const)
    bf = brute_force.solve_brute_force(Qs, consts, k)
    res = baselines.run_all_baselines(Qs, consts, mu, sigma, k, 5.0,
                                      bf["energies"], bf["e_opt"], seed=1)
    for name in ("greedy", "local_search", "annealing", "markowitz_rounded"):
        d = res[name]
        x = np.asarray(d["x"])
        assert x.sum() == k, name
        assert 0 < d["optimality_ratio"] <= 1.0 + 1e-9, name


def test_local_search_improves_greedy():
    mu, sigma, n = _instance()
    k = 3
    Q, const = qubo.build_qubo(mu, sigma, k, 5.0, 10.0)
    Qs, consts, _ = qubo.scale_qubo(Q, const)
    x0 = baselines.greedy_topk_sharpe(mu, sigma, k)
    x1 = baselines.local_search(Qs, consts, k, x0)
    assert qubo.evaluate_qubo(x1, Qs, consts) <= qubo.evaluate_qubo(x0, Qs, consts) + 1e-12


def test_greedy_matches_manual_topk():
    mu = np.array([0.30, 0.10, 0.20, 0.05])
    sigma = np.diag([0.09, 0.04, 0.04, 0.01])
    x = baselines.greedy_topk_sharpe(mu, sigma, 2)
    # sharpe: 1.0, 0.5, 1.0, 0.5 -> tie between 0 and 2; argsort picks them
    assert x.sum() == 2 and x[1] == 0 and x[3] == 0
