"""QUBO -> Ising Hamiltonian mapping (hand-derived, no library shortcut).

Substitution
    x_i = (1 - z_i) / 2      z_i in {+1 (x=0), -1 (x=1)}

Starting from
    H(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + const

each substitution gives
    Q_ii x_i      = Q_ii (1 - z_i)/2
    Q_ij x_i x_j  = Q_ij (1 - z_i - z_j + z_i z_j)/4

Collecting terms yields the Ising form
    E(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset
with
    J_ij   = Q_ij / 4
    h_i    = -Q_ii/2 - (1/4) * sum_{j != i} Q_ij^(i<j)
    offset = sum_i Q_ii/2 + sum_{i<j} Q_ij/4 + const

GOLDEN INVARIANT (the most important test in this repo):
    min_z E(z)  ==  min_x H(x)          (exact equality, up to fp error)
checked in tests/test_ising_mapping.py by exact diagonalisation of the
SparsePauliOp matrix against brute-force enumeration.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubo_to_ising(Q: np.ndarray, const: float):
    """Return (h, J, offset): linear dict {i: h_i}, pair dict {(i,j): J_ij}."""
    n = Q.shape[0]
    h = np.zeros(n)
    J: dict[tuple[int, int], float] = {}
    offset = const
    for i in range(n):
        offset += Q[i, i] / 2.0
        h[i] -= Q[i, i] / 2.0
        for j in range(i + 1, n):
            Qij = Q[i, j]
            if Qij != 0.0:
                J[(i, j)] = J.get((i, j), 0.0) + Qij / 4.0
                offset += Qij / 4.0
                h[i] -= Qij / 4.0
                h[j] -= Qij / 4.0
    return h, J, offset


def _pauli_label_z(n: int, idx: tuple[int, ...]) -> str:
    """Pauli label with Z on the given qubits (Qiskit: leftmost = qubit n-1)."""
    chars = ["I"] * n
    for i in idx:
        chars[n - 1 - i] = "Z"
    return "".join(chars)


def build_hamiltonian(h: np.ndarray, J: dict[tuple[int, int], float]) -> SparsePauliOp:
    """H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j  as a SparsePauliOp."""
    n = len(h)
    labels, coeffs = [], []
    for i in range(n):
        if h[i] != 0.0:
            labels.append(_pauli_label_z(n, (i,)))
            coeffs.append(float(h[i]))
    for (i, j), Jij in J.items():
        if Jij != 0.0:
            labels.append(_pauli_label_z(n, (i, j)))
            coeffs.append(float(Jij))
    if not labels:  # degenerate: identity-only
        labels, coeffs = ["I" * n], [0.0]
    return SparsePauliOp(labels, np.array(coeffs))


def z_from_x(x: np.ndarray) -> np.ndarray:
    """Binary x (0/1) -> spins z (1 - 2x): bit 1 becomes spin -1."""
    return 1 - 2 * np.asarray(x, dtype=float)


def ising_energy(x: np.ndarray, h: np.ndarray, J: dict[tuple[int, int], float], offset: float) -> float:
    """E(z(x)) -- the Ising energy of the configuration encoded by binary x."""
    z = z_from_x(x)
    e = offset + float(h @ z)
    for (i, j), Jij in J.items():
        e += Jij * z[i] * z[j]
    return float(e)


def x_from_z(z: np.ndarray) -> np.ndarray:
    """Spins z (+1/-1) -> binary x (0/1): (1 - z)/2."""
    return ((1 - np.asarray(z, dtype=float)) // 2).astype(int)
