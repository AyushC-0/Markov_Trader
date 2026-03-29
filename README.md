# 🎲 Markov Chain Forex Trader

**Stochastic state-machine strategy with iterative parameter optimization and a live Streamlit dashboard.**

---

## Architecture

```
markov_forex_trader/
├── markov_engine.py    ← Core: MarkovChain, SignalGenerator, Backtester, Optimizer
├── data_feed.py        ← Data: yfinance fetcher + GBM synthetic fallback
├── dashboard.py        ← UI:   Streamlit dashboard (5 tabs)
├── requirements.txt
└── README.md
```

---

## How the Markov Chain Works

```
Price Returns ──► Discretize into N States (quantile / equal-width)
                       │
                       ▼
          Build Transition Matrix T[i, j]
          T[i,j] = P(next_state=j | current_state=i)
          Estimated by counting state→state transitions
          Laplace smoothing applied to avoid zero rows
                       │
                       ▼
   At each bar: observe current return → lookup current state
   → read row i of T → get P(each next state)
                       │
                       ▼
   Bullish Prob  = sum(T[i, mid+1:])   ← upper half states
   Bearish Prob  = sum(T[i, :mid])     ← lower half states
                       │
                       ▼
   if Bullish >= threshold → LONG
   if Bearish >= threshold → SHORT
   else                    → FLAT
```

---

## Parameter Optimizer

Three-phase iterative grid search:

| Phase | What | Variables |
|-------|------|-----------|
| 1 - Coarse Grid | All combinations of structural params | n_states, lookback, holding_period, entry_threshold |
| 2 - Fine Tune   | Fix best structural, sweep risk params | stop_loss, take_profit, position_size, state_method |
| 3 - Micro Tune  | Continuous perturbation around local best | ±Δ on threshold, SL, TP |

**Composite Score** (higher = better):
```python
score = sharpe * (1 - 2*abs(max_drawdown)) * (win_rate + 0.1) + log_bonus(n_trades)
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run dashboard.py

# 3. Open browser at http://localhost:8501
```

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 📊 Overview       | Equity curve, drawdown, key metrics, best params |
| 📈 Price & Signals | Candlestick with entry/exit markers, volume |
| 📋 Trade Log      | Every trade with direction, PnL, exit reason |
| 🔬 Markov Matrix  | Transition matrix heatmap + stationary distribution |
| 🏆 Optimizer      | Ranked top-10 configs, Sharpe vs Score scatter |

---

## Supported Forex Pairs

EUR/USD · GBP/USD · USD/JPY · AUD/USD · USD/CAD · USD/CHF
NZD/USD · EUR/GBP · EUR/JPY · GBP/JPY · AUD/JPY · CHF/JPY
EUR/AUD · EUR/CAD · GBP/AUD

---

## Programmatic Usage

```python
from data_feed import fetch_forex_data
from markov_engine import MarkovParams, Backtester, MarkovOptimizer

# Fetch data
df = fetch_forex_data("EUR/USD", start="2022-01-01", end="2024-01-01")
prices = df["close"].values

# Run with manual params
params = MarkovParams(n_states=5, lookback=20, entry_threshold=0.62)
result = Backtester(params).run(prices, initial_capital=10_000)
print(result.summary())

# Auto-optimize
optimizer = MarkovOptimizer(prices, initial_capital=10_000, max_iterations=2)
best = optimizer.optimize()
print(best.summary())
print(best.params.as_dict())
```

---

## Disclaimer
For research and educational purposes only. Not financial advice.
