"""
Data layer — fetches OHLCV for forex pairs via yfinance.
Supports synthetic generation for offline testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/AUD": "GBPAUD=X",
}

TIMEFRAMES = {
    "1 Day":   {"interval": "1d",  "period": None},
    "4 Hour":  {"interval": "1h",  "period": None},
    "1 Hour":  {"interval": "1h",  "period": None},
    "30 Min":  {"interval": "30m", "period": None},
    "15 Min":  {"interval": "15m", "period": None},
}


def fetch_forex_data(pair: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV for a forex pair.
    Falls back to synthetic data if yfinance fails.
    """
    ticker = FOREX_PAIRS.get(pair, pair)

    if YFINANCE_AVAILABLE:
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, progress=False, auto_adjust=True)
            if df is not None and len(df) > 30:
                df.index = pd.to_datetime(df.index)
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                               for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                return df
        except Exception as e:
            pass

    # Synthetic fallback with realistic forex-like properties
    return _generate_synthetic(pair, start, end, interval)


def _generate_synthetic(pair: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """GBM + mean-reversion synthetic forex data."""
    np.random.seed(abs(hash(pair)) % 2**31)

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    interval_map = {"1d": 1, "1h": 1/24, "30m": 1/48, "15m": 1/96}
    step_days = interval_map.get(interval, 1)
    n_steps = max(100, int((end_dt - start_dt).days / step_days))

    # Pair-specific base prices
    base_prices = {
        "EUR/USD": 1.0850, "GBP/USD": 1.2700, "USD/JPY": 149.50,
        "AUD/USD": 0.6550, "USD/CAD": 1.3600, "USD/CHF": 0.8900,
        "NZD/USD": 0.6100, "EUR/GBP": 0.8550, "EUR/JPY": 162.0,
        "GBP/JPY": 189.0,  "AUD/JPY": 97.50,  "CHF/JPY": 168.0,
        "EUR/AUD": 1.6550, "EUR/CAD": 1.4750, "GBP/AUD": 1.9350,
    }
    S0 = base_prices.get(pair, 1.0)

    mu    = 0.0001
    sigma = 0.0060 if "JPY" not in pair else 0.0040
    theta = 0.01   # mean reversion speed

    prices = np.zeros(n_steps)
    prices[0] = S0
    for t in range(1, n_steps):
        drift = mu - theta * (prices[t-1] - S0)
        shock = sigma * np.random.randn()
        prices[t] = prices[t-1] * np.exp(drift + shock)

    # Build OHLCV
    dates = pd.date_range(start_dt, periods=n_steps, freq=f"{max(1, int(step_days * 1440))}min")
    bar_vol = sigma * prices * 0.5
    opens   = prices * (1 + np.random.randn(n_steps) * 0.0005)
    highs   = prices + np.abs(np.random.randn(n_steps)) * bar_vol
    lows    = prices - np.abs(np.random.randn(n_steps)) * bar_vol

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": np.random.randint(1000, 50000, n_steps).astype(float),
    }, index=dates)
    return df
