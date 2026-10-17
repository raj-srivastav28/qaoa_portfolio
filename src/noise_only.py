"""Re-run ONLY the noisy-simulator study, reusing optimised angles.

Reads results.json (written by `python -m src.main`), rebuilds the instance
and the QAOA distributions from the stored angles, then executes them on the
noisy simulator. Saves figures/noise_study.png and merges
"noise_study" back into results.json.

Usage:  python -m src.noise_only
"""
from __future__ import annotations

import json

import numpy as np

from . import brute_force, config, data_pipeline, ising, main as pipeline, optimize, qubo


def main() -> None:
    if not config.RESULTS_PATH.exists():
        raise SystemExit("results.json not found -- run `python -m src.main` first.")

    with open(config.RESULTS_PATH) as f:
        results = json.load(f)
    thetas = {int(p): np.array(v) for p, v in results["qaoa_thetas"].items()}

    data = data_pipeline.load_data()
    mu, sigma = data["mu"], data["sigma"]
    k = config.K_ASSETS
    Q, const = qubo.build_qubo(mu, sigma, k, config.GAMMA_RISK, config.PENALTY)
    Qs, consts, _ = qubo.scale_qubo(Q, const)
    h, J, _off = ising.qubo_to_ising(Qs, consts)
    bf = brute_force.solve_brute_force(Qs, consts, k)

    rows = pipeline.step_noise(h, J, bf["energies"], k, bf["e_opt"],
                               bf["opt_masks_int"], thetas, config.SEED)
    results["noise_study"] = rows
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(pipeline.to_jsonable(results), f, indent=2)
    print("noise study merged into results.json")


if __name__ == "__main__":
    main()
