"""Central configuration: seeds, universe, model parameters, paths.

Every experiment in this repo reads its knobs from here, so a judge can
change one file and re-run everything. Keep values reproducible (fixed
seeds) and document any value you change for the write-up.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_PATH = ROOT / "results.json"

# ---------------------------------------------------------------- seed -----
SEED = 42                      # master seed; propagated everywhere

# ------------------------------------------------------- universe / data ---
# NIFTY-50 large caps (Yahoo Finance symbols). Real-data path needs internet;
# if download fails we fall back to the cached CSV, then to synthetic data
# (clearly flagged in results and on figures).
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS",
]
PRICE_START = "2024-01-01"
PRICE_END = "2025-12-31"
TRADING_DAYS = 252             # annualisation factor

# ------------------------------------------------------------ the model ----
N_ASSETS = 12                  # N  -> N qubits (one per stock)
K_ASSETS = 4                   # k  -> exactly k stocks must be selected
GAMMA_RISK = 5.0               # risk-aversion coefficient in the objective
PENALTY = 10.0                 # P  -> weight of (sum x - k)^2 constraint term
GAMMA_PENALTY_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
GAMMA_RISK_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

# ------------------------------------------------------------- QAOA --------
P_LAYERS = [1, 2, 3]           # depths studied for the main result
P_MAIN = 2                     # depth used for the frontier / sweeps
SHOTS = 4096                   # sampled shots for reported probabilities
N_RESTARTS = 8                 # random restarts of the classical optimiser
MAX_ITER_COBYLA = 300
SPSA_STEPS = 120

# ------------------------------------------------------------- runtime -----
QUICK_MODE = False             # toggled by main.py --quick (smaller instance)
IBM_TOKEN_ENV = "QISKIT_IBM_TOKEN"   # NEVER hardcode tokens; env var only


def quick_adjust() -> None:
    """Shrink the instance for a fast smoke-test run (main.py --quick)."""
    global N_ASSETS, K_ASSETS, N_RESTARTS, MAX_ITER_COBYLA, SPSA_STEPS
    global P_LAYERS, GAMMA_RISK_SWEEP, GAMMA_PENALTY_SWEEP
    N_ASSETS = 8
    K_ASSETS = 3
    N_RESTARTS = 3
    MAX_ITER_COBYLA = 150
    SPSA_STEPS = 60
    P_LAYERS = [1, 2]
    GAMMA_RISK_SWEEP = [1.0, 5.0, 20.0]
    GAMMA_PENALTY_SWEEP = [1.0, 5.0, 20.0]


def asset_universe() -> list[str]:
    """First N_ASSETS tickers (order fixed for reproducibility)."""
    return TICKERS[:N_ASSETS]
