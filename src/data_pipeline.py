"""Data pipeline: prices -> annualised expected returns (mu) and covariance (Sigma).

Three data sources, tried in order:
  1. cached CSV at data/prices.csv   (works offline, keeps repo reproducible)
  2. Yahoo Finance via yfinance      (fresh data; writes the cache for next time)
  3. deterministic synthetic prices  (last resort so the repo always runs;
                                     flagged loudly - do not use for the
                                     final write-up without saying so)

Outputs
  prices   : DataFrame (T x N) adjusted close prices
  returns  : DataFrame (T-1 x N) daily log returns
  mu       : (N,) annualised expected returns
  sigma    : (N, N) annualised covariance
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CSV_PATH = config.DATA_DIR / "prices.csv"


# --------------------------------------------------------------------- IO --
def _load_csv() -> pd.DataFrame | None:
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
            if df.shape[1] >= config.N_ASSETS and len(df) > 100:
                return df.iloc[:, : config.N_ASSETS]
        except Exception:          # corrupted cache -> fall through
            return None
    return None


def _download_yahoo(tickers: list[str]) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=config.PRICE_START,
            end=config.PRICE_END,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return None
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        close = close.dropna(how="all").ffill().dropna(axis=1, how="any")
        # keep the configured order where possible
        cols = [c for c in tickers if c in close.columns]
        return close[cols]
    except Exception:
        return None


def _synthetic_prices(tickers: list[str], seed: int | None = None) -> pd.DataFrame:
    """Geometric-Brownian-motion prices from a one-factor correlation model.

    Deterministic given the seed. Distributions are chosen to resemble
    annualised equity data (mu in ~[0, 0.25], vols in ~[0.15, 0.45]) so the
    QUBO that reaches QAOA is statistically similar to the real one.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = len(tickers)
    t = config.TRADING_DAYS * 2                       # two years of dailies
    dates = pd.bdate_range("2024-01-02", periods=t)

    # factor model: shared market factor + idiosyncratic noise
    beta = rng.uniform(0.6, 1.3, size=n)              # market betas
    mkt = rng.normal(0.0003, 0.009, size=t)           # market daily returns
    idio_vol = rng.uniform(0.010, 0.020, size=n)
    idio = rng.normal(0.0, 1.0, size=(t, n)) * idio_vol
    drift = rng.uniform(-0.0002, 0.0012, size=n)      # per-stock drift
    daily = drift + beta * mkt[:, None] + idio

    prices = 100.0 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ----------------------------------------------------------------- stats ---
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def mu_sigma(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualise: mu = 252 * mean(daily), Sigma = 252 * cov(daily)."""
    mu = returns.mean().to_numpy() * config.TRADING_DAYS
    sigma = returns.cov().to_numpy() * config.TRADING_DAYS
    return mu, sigma


# ---------------------------------------------------------------- public ---
def load_data() -> dict:
    """Returns dict with prices, returns, mu, sigma, source flag."""
    tickers = config.asset_universe()
    source = None
    prices = _load_csv()
    if prices is not None:
        source = "cache"
    if prices is None:
        prices = _download_yahoo(tickers)
        if prices is not None:
            source = "yahoo"
            config.DATA_DIR.mkdir(exist_ok=True)
            prices.to_csv(CSV_PATH)
    if prices is None:
        prices = _synthetic_prices(tickers)
        source = "synthetic"
    prices = prices[tickers] if all(c in prices.columns for c in tickers) else prices.iloc[:, : config.N_ASSETS]

    rets = log_returns(prices)
    mu, sigma = mu_sigma(rets)
    return {
        "prices": prices,
        "returns": rets,
        "mu": mu,
        "sigma": sigma,
        "tickers": list(prices.columns),
        "source": source,
    }
