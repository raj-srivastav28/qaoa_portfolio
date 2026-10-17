"""GOLDEN TEST: Ising Hamiltonian ground state == brute-force optimum.

This is the single most important check in the repo: it proves the whole
encoding chain (portfolio objective -> QUBO -> Ising -> Pauli operator) is
exact. If this fails, every downstream quantum result is meaningless.
"""
import numpy as np

from src import brute_force, ising, qubo

RNG = np.random.default_rng(11)


def _random_qubo(n):
    Q = np.triu(RNG.normal(0, 1.0, size=(n, n)), k=1) \
        + np.diag(RNG.uniform(-1, 1, size=n))
    return Q, float(RNG.uniform(-2, 2))


def test_ising_ground_state_matches_brute_force():
    """eigmin(H) + offset == min H(x): the Pauli operator carries no constant
    term (QAOA is invariant to global energy shifts), so the offset must be
    added back before comparing with the brute-force energies."""
    for n in (4, 5, 6):
        Q, const = _random_qubo(n)
        h, J, offset = ising.qubo_to_ising(Q, const)
        H = ising.build_hamiltonian(h, J)
        eig_min = float(np.linalg.eigvalsh(H.to_matrix()).min()) + offset
        energies = brute_force.enumerate_energies(Q, const)
        assert np.isclose(eig_min, energies.min(), atol=1e-8), (n, eig_min, energies.min())


def test_every_state_matches():
    """E(z(x)) must equal H(x) for ALL 2^6 assignments, not just the minimum."""
    n = 6
    Q, const = _random_qubo(n)
    h, J, offset = ising.qubo_to_ising(Q, const)
    energies = brute_force.enumerate_energies(Q, const)
    for m in range(2**n):
        x = brute_force.int_to_bits(m, n)
        assert np.isclose(ising.ising_energy(x, h, J, offset), energies[m], atol=1e-9), m


def test_hamiltonian_terms_match_derivation():
    """Spot-check the Pauli coefficients against the closed-form derivation."""
    n = 4
    Q = np.array([
        [-1.0, 0.8, 0.0, 0.0],
        [0.0, 2.0, -0.4, 0.0],
        [0.0, 0.0, 0.5, 0.2],
        [0.0, 0.0, 0.0, -0.3],
    ])
    const = 1.5
    h, J, offset = ising.qubo_to_ising(Q, const)
    # J_ij = Q_ij/4
    assert np.isclose(J[(0, 1)], 0.8 / 4)
    assert np.isclose(J[(2, 3)], 0.2 / 4)
    # h_i = -Q_ii/2 - sum_{j!=i} Q_ij/4
    assert np.isclose(h[0], -(-1.0) / 2 - 0.8 / 4)
    assert np.isclose(h[2], -0.5 / 2 - (-0.4 / 4 + 0.2 / 4))
    # offset = sum Q_ii/2 + sum Q_ij/4 + const
    expected_off = np.trace(Q) / 2 + (0.8 - 0.4 + 0.2) / 4 + const
    assert np.isclose(offset, expected_off)
