"""Noisy simulation: QAOA under a hardware-like noise model.

Uses AerSimulator with a fake-backend noise model (or a generic
depolarising model as fallback). Goal: find the largest depth p that still
beats p=1 / stays useful -- a Track 3 stretch goal.

Nothing here touches real hardware; see hardware.py for the (optional,
bonus) IBM Quantum run.
"""
from __future__ import annotations

import numpy as np

from .qaoa_circuit import build_qaoa_circuit, add_measurements


def build_noise_model(n_qubits: int):
    """Fake-backend noise model if available, else generic depolarising."""
    try:  # preferred: IBM fake backend with calibrated noise
        from qiskit_ibm_runtime.fake_provider import (
            FakeGuadalupeV2, FakeMontrealV2, FakeManilaV2, FakeLimaV2,
        )
        for cand in (FakeGuadalupeV2, FakeMontrealV2, FakeManilaV2, FakeLimaV2):
            try:
                backend = cand()
                if backend.num_qubits >= n_qubits:
                    return backend, f"noise_model={backend.name}"
            except Exception:
                continue
    except Exception:
        pass

    # fallback: generic depolarising model, values typical of 2024-era devices
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    nm = NoiseModel()
    e1 = depolarizing_error(1e-3, 1)
    e2 = depolarizing_error(1e-2, 2)
    nm.add_all_qubit_quantum_error(e1, ["rz", "sx", "x", "h", "rx"])
    nm.add_all_qubit_quantum_error(e2, ["cx", "ecr"])
    nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))
    return nm, "noise_model=generic_depolarizing"


def run_noisy_qaoa(
    h: np.ndarray,
    J: dict[tuple[int, int], float],
    p: int,
    energies: np.ndarray,
    k: int,
    e_opt: float,
    opt_masks_int: list[int],
    theta: np.ndarray,
    shots: int = 4096,
    seed: int = 42,
) -> dict:
    """Execute the optimised QAOA circuit under noise; score the samples."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc, gammas, betas = build_qaoa_circuit(h, J, p)
    qc = add_measurements(qc).assign_parameters({**dict(zip(gammas, theta[:p])),
                                                 **dict(zip(betas, theta[p:]))})
    model, model_name = build_noise_model(len(h))
    backend = AerSimulator(noise_model=model) if not hasattr(model, "target") else AerSimulator.from_backend(model)
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=seed)
    result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
    counts = {int(key, 2): v for key, v in result.get_counts().items()}

    total = sum(counts.values())
    probs = np.zeros(len(energies))
    for m, c in counts.items():
        probs[m] = c / total

    e_best = min(energies[m] for m in counts)
    p_opt = float(sum(probs[m] for m in opt_masks_int))
    n = len(h)
    sums = np.array([bin(m).count("1") for m in range(len(probs))])
    feasible_rate = float(probs[sums == k].sum())
    return {
        "p": p,
        "e_best_sampled": float(e_best),
        "optimality_ratio": float(1.0 - (e_best - float(np.min(energies)))
                                  / (float(np.max(energies)) - float(np.min(energies)))),
        "prob_optimal": p_opt,
        "feasible_rate": feasible_rate,
        "model": model_name,
        "shots": shots,
    }
