"""OPTIONAL bonus: run the smallest QAOA instance on real IBM hardware.

Rules honoured here (from the organisers' guidelines):
  * the IBM token is read ONLY from the environment variable QISKIT_IBM_TOKEN
    (never hardcoded, never committed -- their check_submission.py greps for it);
  * a simulator-only path exists so the repo runs without any account;
  * the run records backend name, date, job ID for the README table.

Usage
    export QISKIT_IBM_TOKEN=...        # do NOT put this in any file
    python -c "from src.hardware import run_on_hardware; run_on_hardware()"

NOTE: the IBM Runtime API evolves quickly; if this script errors, adapt the
service/sampler calls to the current qiskit-ibm-runtime API, keep the
record-keeping, and document the change in your write-up.
"""
from __future__ import annotations

import datetime as _dt
import os

from . import config


def run_on_hardware(h, J, p: int, theta, shots: int = 1024) -> dict | None:
    token = os.environ.get(config.IBM_TOKEN_ENV, "").strip()
    if not token:
        print("[hardware] QISKIT_IBM_TOKEN not set -- skipping hardware run "
              "(simulator-only path is fine; the run is an optional bonus).")
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        from .qaoa_circuit import build_qaoa_circuit, add_measurements

        qc, gammas, betas = build_qaoa_circuit(h, J, p)
        qc = add_measurements(qc).assign_parameters(
            {**dict(zip(gammas, theta[:p])), **dict(zip(betas, theta[p:]))}
        )

        service = QiskitRuntimeService(token=token)
        backend = service.least_busy(operational=True, simulator=False)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_qc = pm.run(qc)

        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            job = sampler.run([isa_qc], shots=shots)
            result = job.result()

        counts = result[0].data.meas.get_counts()
        record = {
            "backend": backend.name,
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "job_id": job.job_id(),
            "shots": shots,
            "counts_top10": dict(
                sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            ),
        }
        print("[hardware] recorded:", record["backend"], record["job_id"])
        return record
    except Exception as exc:  # never break the pipeline on the bonus path
        print(f"[hardware] run failed ({exc!r}) -- simulator results stand alone.")
        return None
