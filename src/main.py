"""End-to-end pipeline: data -> QUBO -> Ising -> QAOA vs classical baselines.

Usage
    python -m src.main              # full experiment (N = 12)
    python -m src.main --quick      # smoke test (N = 8, reduced sweeps)
    python -m src.main --noise      # + noisy-simulator depth study
    python -m src.main --hardware   # + optional real-IBM bonus run (token needed)

Outputs: figures/*.png + results.json + console summary.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import baselines, brute_force, config, data_pipeline, ising, noise, optimize, qubo
from .qaoa_circuit import build_qaoa_circuit


# ----------------------------------------------------------------- utils ---
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _fig_ax(w=9, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    ax.grid(alpha=0.3)
    return fig, ax


# ----------------------------------------------------------------- steps ---
def step_main_result(h, J, Qs, consts, energies, e_opt, opt_masks, mu, sigma,
                     k, seed):
    """Fig 1: optimality ratio vs depth p (mean +- std over restarts)."""
    per_p, best_theta = {}, {}
    for p in config.P_LAYERS:
        res = optimize.optimize_qaoa(h, J, p, energies, optimizer="COBYLA",
                                     n_restarts=config.N_RESTARTS,
                                     maxiter=config.MAX_ITER_COBYLA, seed=seed,
                                     polish=True)
        ratios = []
        for r in res["all_restarts"]:
            probs = optimize.probs_from_circuit(res["qc"], r["theta"],
                                                res["gammas"], res["betas"])
            ev = optimize.evaluate_distribution(probs, energies, k, e_opt,
                                                opt_masks, config.SHOTS,
                                                seed=seed + p)
            ratios.append(ev["optimality_ratio"])
        best_ev = optimize.evaluate_distribution(res["probs"], energies, k, e_opt,
                                                 opt_masks, config.SHOTS,
                                                 seed=seed + p)
        per_p[p] = {
            "ratios": ratios, "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)), "best": best_ev,
            "expected_energy": res["energy"],
        }
        best_theta[p] = res["theta"]
        print(f"  p={p}: ratio={per_p[p]['mean']:.4f} +- {per_p[p]['std']:.4f} "
              f"| P(optimal)={best_ev['prob_optimal']:.3f} "
              f"| feasible={best_ev['feasible_rate']:.3f}")

    greedy = baselines.greedy_topk_sharpe(mu, sigma, k)
    greedy_ratio = brute_force.optimality_ratio(
        energies, qubo.evaluate_qubo(greedy, Qs, consts))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ps = sorted(per_p)
    means = [per_p[p]["mean"] for p in ps]
    stds = [per_p[p]["std"] for p in ps]
    ax.errorbar(ps, means, yerr=stds, marker="o", capsize=4, lw=2,
                color="tab:blue", label="QAOA (COBYLA, restarts mean $\\pm$ std)")
    ax.axhline(1.0, color="green", ls="--", lw=1.5, label="Exact optimum (2^N)")
    ax.axhline(greedy_ratio, color="orange", ls=":", lw=1.8,
               label=f"Greedy top-k Sharpe ({greedy_ratio:.3f})")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Optimality ratio  (E_opt / E_best-sampled)")
    ax.set_ylim(0.9, 1.02)
    ax.set_title("Quality of best sampled portfolio")
    ax.legend(fontsize=8)

    ax2.plot(ps, [per_p[p]["best"]["prob_optimal"] for p in ps], "o-",
             lw=2, color="tab:purple", label="P(sample = optimal portfolio)")
    ax2.plot(ps, [per_p[p]["best"]["feasible_rate"] for p in ps], "s--",
             lw=1.6, color="tab:red", label="P(constraint satisfied)")
    ax2.set_xlabel("QAOA depth p")
    ax2.set_ylabel("Probability")
    ax2.set_title("Probability mass vs depth")
    ax2.legend(fontsize=8)
    fig.suptitle(f"QAOA solution quality vs depth  (N={config.N_ASSETS}, "
                 f"k={config.K_ASSETS}, {config.SHOTS} shots, "
                 f"{config.N_RESTARTS} restarts)")
    fig.savefig(config.FIGURES_DIR / "main_result.png", dpi=150)
    plt.close(fig)
    return per_p, best_theta, greedy_ratio


def step_baselines(Q, const, mu, sigma, k, gamma_risk, energies, e_opt, seed):
    res = baselines.run_all_baselines(Q, const, mu, sigma, k, gamma_risk,
                                      energies, e_opt, seed=seed)
    print("  classical baselines (same energy ruler as QAOA):")
    for name, d in res.items():
        print(f"    {name:<18} ratio={d['optimality_ratio']:.4f}")
    return res


def step_frontier(mu, sigma, k, seed):
    """Fig 2: cardinality-constrained efficient frontier: exact vs QAOA."""
    exact_pts, qaoa_pts = [], []
    greedy_x = baselines.greedy_topk_sharpe(mu, sigma, k)
    for gamma_risk in config.GAMMA_RISK_SWEEP:
        Q, const = qubo.build_qubo(mu, sigma, k, gamma_risk, config.PENALTY)
        Qs, consts, _ = qubo.scale_qubo(Q, const)
        h, J, off = ising.qubo_to_ising(Qs, consts)
        bf = brute_force.solve_brute_force(Qs, consts, k)
        x_best = brute_force.int_to_bits(bf["opt_masks_int"][0], len(mu))
        exact_pts.append(qubo.portfolio_metrics(x_best, mu, sigma, gamma_risk))

        energies = bf["energies"]
        res = optimize.optimize_qaoa(h, J, config.P_MAIN, energies,
                                     n_restarts=2, maxiter=120, seed=seed)
        ev = optimize.evaluate_distribution(res["probs"], energies, k,
                                            bf["e_opt"], bf["opt_masks_int"],
                                            config.SHOTS, seed=seed)
        qaoa_pts.append(qubo.portfolio_metrics(ev["best_x"], mu, sigma, gamma_risk))

    fig, ax = _fig_ax()
    ax.plot([p["vol"] for p in exact_pts], [p["return"] for p in exact_pts],
            "o-", lw=2, label="Exact (2^N search)")
    ax.plot([p["vol"] for p in qaoa_pts], [p["return"] for p in qaoa_pts],
            "s--", lw=1.6, label=f"QAOA p={config.P_MAIN}")
    gp = qubo.portfolio_metrics(greedy_x, mu, sigma, 0)
    ax.plot(gp["vol"], gp["return"], "^", ms=12, color="orange",
            label="Greedy top-k Sharpe")
    for gamma_risk, p in zip(config.GAMMA_RISK_SWEEP, exact_pts):
        ax.annotate(f"$\\gamma$={gamma_risk:g}", (p["vol"], p["return"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Portfolio volatility  sqrt(x' Sigma x)  (annualised)")
    ax.set_ylabel("Expected return  mu' x  (annualised)")
    ax.set_title(f"Cardinality-constrained efficient frontier "
                 f"(exactly k={k} of N={len(mu)} stocks)")
    ax.legend()
    fig.savefig(config.FIGURES_DIR / "frontier.png", dpi=150)
    plt.close(fig)
    return {"exact": exact_pts, "qaoa": qaoa_pts,
            "gamma_sweep": config.GAMMA_RISK_SWEEP}


def step_penalty_sweep(mu, sigma, k, seed):
    """Fig 3: constraint enforcement vs penalty weight P."""
    out = []
    for P in config.GAMMA_PENALTY_SWEEP:
        Q, const = qubo.build_qubo(mu, sigma, k, config.GAMMA_RISK, P)
        Qs, consts, _ = qubo.scale_qubo(Q, const)
        h, J, off = ising.qubo_to_ising(Qs, consts)
        bf_all = brute_force.solve_brute_force(Qs, consts, k=None)
        argmin_feasible = int(round(sum(brute_force.int_to_bits(
            int(np.argmin(bf_all["energies"])), len(mu)))))
        bf = brute_force.solve_brute_force(Qs, consts, k)
        energies = bf["energies"]
        res = optimize.optimize_qaoa(h, J, 1, energies, n_restarts=2,
                                     maxiter=100, seed=seed)
        ev = optimize.evaluate_distribution(res["probs"], energies, k,
                                            bf["e_opt"], bf["opt_masks_int"],
                                            config.SHOTS, seed=seed)
        out.append({"P": P, "argmin_feasible": bool(argmin_feasible == k),
                    "feasible_rate": ev["feasible_rate"],
                    "ratio": ev["optimality_ratio"]})
        print(f"  P={P:>6g}: unconstrained argmin feasible="
              f"{argmin_feasible == k} | QAOA feasible rate="
              f"{ev['feasible_rate']:.3f} | ratio={ev['optimality_ratio']:.3f}")

    fig, ax1 = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    Ps = [d["P"] for d in out]
    ax1.plot(Ps, [d["feasible_rate"] for d in out], "o-", lw=2,
             label="QAOA feasible-solution rate (p=1)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Penalty weight P")
    ax1.set_ylabel("Feasible rate (sum x = k)", color="tab:blue")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(Ps, [d["ratio"] for d in out], "s--", color="tab:red",
             label="Optimality ratio")
    ax2.set_ylabel("Optimality ratio", color="tab:red")
    ax1.axvline(config.PENALTY, color="gray", ls=":",
                label=f"P used in main run ({config.PENALTY:g})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("Effect of the penalty weight on feasibility and quality")
    fig.savefig(config.FIGURES_DIR / "penalty_sweep.png", dpi=150)
    plt.close(fig)
    return out


def step_optimizers(h, J, energies, k, e_opt, opt_masks, seed):
    """Fig 4: COBYLA vs SPSA convergence at p = P_MAIN."""
    cob = optimize.optimize_qaoa(h, J, config.P_MAIN, energies, "COBYLA",
                                 n_restarts=config.N_RESTARTS,
                                 maxiter=config.MAX_ITER_COBYLA, seed=seed,
                                 polish=True)
    spsa = optimize.optimize_qaoa(h, J, config.P_MAIN, energies, "SPSA",
                                  n_restarts=config.N_RESTARTS,
                                  maxiter=config.SPSA_STEPS, seed=seed)
    fig, ax = _fig_ax()
    ax.plot(cob["history"], lw=2, label=f"COBYLA (best restart, "
                                        f"E={cob['energy']:.4f})")
    ax.plot(spsa["history"], lw=2, label=f"SPSA (best restart, "
                                         f"E={spsa['energy']:.4f})")
    ax.axhline(e_opt, color="green", ls="--", lw=1.5,
               label="Exact optimum")
    ax.set_xlabel("Cost-function evaluations")
    ax.set_ylabel("Expected energy <H>  (running best)")
    ax.set_title(f"Optimiser comparison at p={config.P_MAIN}")
    ax.legend()
    fig.savefig(config.FIGURES_DIR / "optimizers.png", dpi=150)
    plt.close(fig)
    return {"cobyla_final": cob["energy"], "spsa_final": spsa["energy"],
            "e_opt": e_opt}


def step_noise(h, J, energies, k, e_opt, opt_masks, thetas, seed):
    """Fig 5 (optional): probability mass under noise vs ideal, per depth.

    Compares distributions with the SAME optimised angles: how much of the
    optimal-portfolio probability (and constraint feasibility) survives
    hardware-like noise.
    """
    rows = []
    for p in config.P_LAYERS:
        # ideal reference distribution with the SAME optimised angles
        qc, gammas, betas = build_qaoa_circuit(h, J, p)
        ideal_probs = optimize.probs_from_circuit(qc, thetas[p], gammas, betas)
        ideal_ev = optimize.evaluate_distribution(
            ideal_probs, energies, k, e_opt, opt_masks, config.SHOTS, seed=seed)
        noisy = noise.run_noisy_qaoa(h, J, p, energies, k, e_opt, opt_masks,
                                     thetas[p], shots=config.SHOTS, seed=seed)
        noisy["ideal_prob_optimal"] = ideal_ev["prob_optimal"]
        noisy["ideal_feasible_rate"] = ideal_ev["feasible_rate"]
        rows.append(noisy)
        print(f"  p={p}: noisy P(optimal)={noisy['prob_optimal']:.4f} "
              f"vs ideal {ideal_ev['prob_optimal']:.4f} "
              f"| feasible {noisy['feasible_rate']:.3f} vs "
              f"{ideal_ev['feasible_rate']:.3f} | model={noisy['model']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    x = np.arange(len(rows))
    ax1.bar(x - 0.18, [r["prob_optimal"] for r in rows], width=0.34,
            color="tab:red", label="Noisy simulator")
    ax1.bar(x + 0.18, [r["ideal_prob_optimal"] for r in rows], width=0.34,
            color="tab:purple", alpha=0.55, label="Ideal distribution")
    ax1.set_xticks(x, [f"p={r['p']}" for r in rows])
    ax1.set_ylabel("P(sample = optimal portfolio)")
    ax1.set_title("Optimal-portfolio probability")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.bar(x - 0.18, [r["feasible_rate"] for r in rows], width=0.34,
            color="tab:red", label="Noisy simulator")
    ax2.bar(x + 0.18, [r["ideal_feasible_rate"] for r in rows], width=0.34,
            color="tab:red", alpha=0.35, label="Ideal distribution")
    ax2.set_xticks(x, [f"p={r['p']}" for r in rows])
    ax2.set_ylabel("P(constraint satisfied)")
    ax2.set_title("Feasibility under noise")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle("Noise study: same optimised angles, noisy execution "
                 f"({rows[0]['model']})")
    fig.savefig(config.FIGURES_DIR / "noise_study.png", dpi=150)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------ main ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--noise", action="store_true", help="noisy depth study")
    ap.add_argument("--hardware", action="store_true",
                    help="optional real-IBM bonus run (needs QISKIT_IBM_TOKEN)")
    args = ap.parse_args()
    if args.quick:
        config.quick_adjust()

    t0 = time.time()
    seed = config.SEED
    np.random.seed(seed)
    config.FIGURES_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("QAOA Portfolio Constructor -- end-to-end pipeline")
    print("=" * 70)

    # 1. data ---------------------------------------------------------------
    data = data_pipeline.load_data()
    mu, sigma = data["mu"], data["sigma"]
    n, k = len(mu), config.K_ASSETS
    print(f"[1] data source: {data['source']} | N={n} assets, k={k}")
    print(f"    mu range   [{mu.min():+.3f}, {mu.max():+.3f}]")
    print(f"    vol range  [{np.sqrt(np.diag(sigma)).min():.3f}, "
          f"{np.sqrt(np.diag(sigma)).max():.3f}]")
    if data["source"] == "synthetic":
        print("    WARNING: synthetic fallback data (no internet/cache). "
              "State this clearly in the write-up if you keep it.")

    # 2. QUBO + Ising + golden invariant check ------------------------------
    Q, const = qubo.build_qubo(mu, sigma, k, config.GAMMA_RISK, config.PENALTY)
    Qs, consts, scale = qubo.scale_qubo(Q, const)
    h, J, offset = ising.qubo_to_ising(Qs, consts)
    H = ising.build_hamiltonian(h, J)
    bf = brute_force.solve_brute_force(Qs, consts, k)
    # NOTE: the Pauli operator carries no constant term, so add the Ising
    # offset back before comparing with brute-force energies.
    eig_min = float(np.linalg.eigvalsh(H.to_matrix()).min()) + offset
    golden_ok = np.isclose(eig_min, bf["e_opt"], atol=1e-8)
    print(f"[2] QUBO built and scaled by {scale:.3g} | Ising terms: "
          f"{sum(v != 0 for v in h)} linear + {len(J)} pair")
    print(f"    GOLDEN CHECK  Ising eigmin={eig_min:.10f} vs "
          f"brute-force {bf['e_opt']:.10f}  ->  "
          f"{'PASS' if golden_ok else 'FAIL'}")
    if not golden_ok:
        raise RuntimeError("Ising mapping disagrees with brute force -- "
                           "the pipeline is untrustworthy; fix before anything else.")

    # 3. QAOA main result ----------------------------------------------------
    print(f"[3] QAOA depth study (optimiser=COBYLA, "
          f"{config.N_RESTARTS} restarts/depth):")
    per_p, thetas, greedy_ratio = step_main_result(
        h, J, Qs, consts, bf["energies"], bf["e_opt"], bf["opt_masks_int"],
        mu, sigma, k, seed)

    # 4. baselines -----------------------------------------------------------
    print("[4] classical baselines:")
    bl = step_baselines(Qs, consts, mu, sigma, k, config.GAMMA_RISK,
                        bf["energies"], bf["e_opt"], seed)

    # 5. frontier ------------------------------------------------------------
    print(f"[5] cardinality-constrained frontier sweep over gamma:")
    frontier = step_frontier(mu, sigma, k, seed)

    # 6. penalty sweep -------------------------------------------------------
    print(f"[6] penalty-weight sweep:")
    penalty = step_penalty_sweep(mu, sigma, k, seed)

    # 7. optimiser comparison ------------------------------------------------
    print(f"[7] optimiser comparison at p={config.P_MAIN}:")
    optcmp = step_optimizers(h, J, bf["energies"], k, bf["e_opt"],
                             bf["opt_masks_int"], seed)
    print(f"    COBYLA E={optcmp['cobyla_final']:.4f} | "
          f"SPSA E={optcmp['spsa_final']:.4f} | optimum {bf['e_opt']:.4f}")

    # 8. optional: noise study ----------------------------------------------
    noise_rows = None
    if args.noise:
        print("[8] noisy-simulator depth study:")
        noise_rows = step_noise(h, J, bf["energies"], k, bf["e_opt"],
                                bf["opt_masks_int"], thetas, seed)

    # 9. optional: hardware bonus -------------------------------------------
    hw = None
    if args.hardware:
        from . import hardware
        print("[9] optional hardware run:")
        hw = hardware.run_on_hardware(h, J, min(config.P_LAYERS),
                                      thetas[min(config.P_LAYERS)])

    # 10. results -------------------------------------------------------------
    best_p = max(per_p, key=lambda p: per_p[p]["best"]["optimality_ratio"])
    summary = {
        "data_source": data["source"],
        "tickers": data["tickers"],
        "n_assets": n, "k": k,
        "gamma_risk": config.GAMMA_RISK, "penalty": config.PENALTY,
        "scale_factor": scale,
        "golden_check_pass": bool(golden_ok),
        "e_opt_scaled": bf["e_opt"],
        "n_optimal_portfolios": bf["n_optimal"],
        "optimal_portfolio": brute_force.int_to_bits(
            bf["opt_masks_int"][0], n).tolist(),
        "optimal_selected": [data["tickers"][i] for i in
                             np.flatnonzero(brute_force.int_to_bits(
                                 bf["opt_masks_int"][0], n))],
        "qaoa_by_depth": {str(p): {kk: vv for kk, vv in d.items()
                                   if kk != "best"} for p, d in per_p.items()},
        "qaoa_thetas": {str(p): thetas[p].tolist() for p in thetas},
        "qaoa_best_overall": {"p": best_p, **per_p[best_p]["best"]},
        "baselines": bl,
        "frontier": frontier,
        "penalty_sweep": penalty,
        "optimizer_comparison": optcmp,
        "noise_study": noise_rows,
        "hardware": hw,
        "runtime_seconds": round(time.time() - t0, 1),
        "seed": seed,
    }
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print("=" * 70)
    print(f"DONE in {summary['runtime_seconds']}s | seed={seed}")
    print(f"  optimal portfolio (exact): {summary['optimal_selected']}")
    print(f"  QAOA best: p={best_p}, ratio="
          f"{per_p[best_p]['best']['optimality_ratio']:.4f}, "
          f"P(optimal)={per_p[best_p]['best']['prob_optimal']:.3f}")
    print(f"  figures -> {config.FIGURES_DIR} | results -> "
          f"{config.RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
