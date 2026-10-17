"""Hand-built QAOA ansatz (no ready-made solver).

Circuit, for depth p:
    |+>^N                                  (Hadamard on every qubit)
    repeat p times:
        cost layer   U_C(g) = exp(-i g H)  with H the Ising cost Hamiltonian
            - RZ(2 g h_i) on every qubit with h_i != 0
            - CX(i,j) RZ(2 g J_ij) CX(i,j)  for every pair term (Z_i Z_j)
        mixer layer  U_M(b) = exp(-i b sum_i X_i)
            - RX(2 b) on every qubit
    (measurement added separately so the same circuit can be used with
     Statevector during optimisation)

Why this implements the layers: RZ(t) = exp(-i t Z/2), and
CX(i,j) RZ(t on j) CX(i,j) = exp(-i t Z_i Z_j / 2). With the cost angle g
multiplied by 2 we get exactly exp(-i g H).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_qaoa_circuit(h: np.ndarray, J: dict[tuple[int, int], float], p: int):
    """Return (qc, gamma_params, beta_params); qc has NO measurements."""
    n = len(h)
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    qc = QuantumCircuit(n)

    qc.h(range(n))  # uniform superposition |+>^N

    for layer in range(p):
        # ---- cost layer: exp(-i * gamma * H) ----
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * gammas[layer] * float(h[i]), i)
        for (i, j), Jij in J.items():
            if Jij != 0.0:
                qc.cx(i, j)
                qc.rz(2.0 * gammas[layer] * float(Jij), j)
                qc.cx(i, j)
        # ---- mixer layer: exp(-i * beta * sum X) ----
        for i in range(n):
            qc.rx(2.0 * betas[layer], i)

    return qc, list(gammas), list(betas)


def add_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Copy with measure_all (for shot-based execution)."""
    measured = qc.copy()
    measured.measure_all()
    return measured
