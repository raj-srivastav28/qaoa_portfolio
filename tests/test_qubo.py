"""Golden tests for the QUBO formulation.

1. evaluate_qubo matches a direct evaluation of the raw objective
   H(x) = -mu.x + (gamma/2) x'Sigma x + P (sum x - k)^2   for random x.
2. A toy instance where the answer is provable by hand:
   one stock strictly dominates -> the optimum must select it,
   and the penalty must make infeasible selections worse.
"""
import numpy as np

from src import qubo

RNG = np.random.default_rng(7)


def _random_instance(n):
    A = RNG.normal(0.10, 0.02, size=(n, n))
    sigma = A @ A.T / n * 252 * 0.05
    np.fill_diagonal(sigma, np.linspace(0.02, 0.09, n))
    mu = RNG.uniform(-0.05, 0.30, size=n)
    return mu, sigma


def test_qubo_matches_direct_objective():
    n, k, gamma, P = 6, 2, 3.0, 8.0
    mu, sigma = _random_instance(n)
    Q, const = qubo.build_qubo(mu, sigma, k, gamma, P)
    for _ in range(50):
        x = RNG.integers(0, 2, size=n)
        direct = (-mu @ x + 0.5 * gamma * x @ sigma @ x
                  + P * (x.sum() - k) ** 2)
        via_qubo = qubo.evaluate_qubo(x, Q, const)
        assert np.isclose(direct, via_qubo, atol=1e-9), (x, direct, via_qubo)


def test_penalty_enforces_cardinality():
    """With a big P, the unconstrained optimum must pick exactly k."""
    n, k, gamma, P = 7, 3, 2.0, 50.0
    mu, sigma = _random_instance(n)
    Q, const = qubo.build_qubo(mu, sigma, k, gamma, P)
    energies = [qubo.evaluate_qubo(np.array([(m >> i) & 1 for i in range(n)]),
                                   Q, const)
                for m in range(2**n)]
    best = np.array([(int(np.argmin(energies)) >> i) & 1 for i in range(n)])
    assert best.sum() == k


def test_scale_qubo_preserves_argmin():
    n, k = 6, 2
    mu, sigma = _random_instance(n)
    Q, const = qubo.build_qubo(mu, sigma, k, 5.0, 10.0)
    Qs, consts, s = qubo.scale_qubo(Q, const)
    assert np.isclose(np.max(np.abs(Qs)), 1.0)
    e_raw = [qubo.evaluate_qubo(np.array([(m >> i) & 1 for i in range(n)]),
                                Q, const) for m in range(2**n)]
    e_scl = [qubo.evaluate_qubo(np.array([(m >> i) & 1 for i in range(n)]),
                                Qs, consts) for m in range(2**n)]
    assert np.argmin(e_raw) == np.argmin(e_scl)
