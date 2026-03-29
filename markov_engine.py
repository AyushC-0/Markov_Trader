"""
╔══════════════════════════════════════════════════════════════╗
║        MARKOV CHAIN FOREX TRADING ENGINE                     ║
║        Markov State Predictor + Iterative Optimizer          ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class MarkovParams:
    n_states: int = 5           # Number of discretized return states
    lookback: int = 20          # Bars used to estimate transition matrix
    holding_period: int = 1     # Bars to hold a position
    entry_threshold: float = 0.60  # Min probability to enter a trade
    stop_loss_pct: float = 0.005   # 0.5% SL
    take_profit_pct: float = 0.010 # 1.0% TP
    position_size_pct: float = 0.02  # 2% risk per trade
    order_type: str = "market"  # market | limit
    state_method: str = "quantile"  # quantile | equal | kmeans

    def as_dict(self):
        return self.__dict__

@dataclass
class TradeRecord:
    entry_bar: int
    entry_price: float
    direction: str      # LONG | SHORT
    size: float
    exit_price: float = 0.0
    exit_bar: int = 0
    pnl: float = 0.0
    exit_reason: str = ""
    entry_state: int = 0
    predicted_prob: float = 0.0


@dataclass
class BacktestResult:
    params: MarkovParams
    trades: list
    equity_curve: np.ndarray
    returns: np.ndarray
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    n_trades: int
    score: float = 0.0   # composite optimization score

    def summary(self):
        return {
            "Sharpe Ratio": round(self.sharpe, 3),
            "Max Drawdown %": round(self.max_drawdown * 100, 2),
            "Win Rate %": round(self.win_rate * 100, 2),
            "Profit Factor": round(self.profit_factor, 3),
            "Total Return %": round(self.total_return * 100, 2),
            "N Trades": self.n_trades,
            "Score": round(self.score, 4),
        }


# ─────────────────────────────────────────────
#  MARKOV CHAIN CORE
# ─────────────────────────────────────────────

class MarkovChain:
    """
    Discrete-state Markov Chain built from price return quantiles.
    Transition matrix T[i,j] = P(state_j | state_i)
    """

    def __init__(self, params: MarkovParams):
        self.params = params
        self.transition_matrix = None
        self.state_edges = None
        self.n_states = params.n_states

    def _discretize(self, returns: np.ndarray, fit=True) -> np.ndarray:
        """Map continuous returns → discrete state indices."""
        if fit:
            if self.params.state_method == "quantile":
                quantiles = np.linspace(0, 100, self.n_states + 1)
                self.state_edges = np.percentile(returns, quantiles)
                self.state_edges[0] = -np.inf
                self.state_edges[-1] = np.inf
            elif self.params.state_method == "equal":
                lo, hi = returns.min(), returns.max()
                self.state_edges = np.linspace(lo, hi, self.n_states + 1)
                self.state_edges[0] = -np.inf
                self.state_edges[-1] = np.inf

        states = np.digitize(returns, self.state_edges[1:-1])
        return np.clip(states, 0, self.n_states - 1)

    def fit(self, returns: np.ndarray):
        """Estimate transition matrix from a return series."""
        states = self._discretize(returns, fit=True)
        n = self.n_states
        T = np.zeros((n, n))
        for t in range(len(states) - 1):
            T[states[t], states[t + 1]] += 1

        # Laplace smoothing to avoid zero rows
        T += 1e-6
        row_sums = T.sum(axis=1, keepdims=True)
        self.transition_matrix = T / row_sums
        return self

    def predict_next(self, current_return: float) -> np.ndarray:
        """Return probability distribution over next states."""
        if self.transition_matrix is None:
            raise RuntimeError("Call fit() first.")
        state = int(np.digitize(current_return, self.state_edges[1:-1]))
        state = np.clip(state, 0, self.n_states - 1)
        return self.transition_matrix[state].copy()

    def bullish_prob(self, probs: np.ndarray) -> float:
        """Sum probability mass in upper half of state space."""
        mid = self.n_states // 2
        return float(probs[mid + 1:].sum())

    def bearish_prob(self, probs: np.ndarray) -> float:
        """Sum probability mass in lower half of state space."""
        mid = self.n_states // 2
        return float(probs[:mid].sum())


# ─────────────────────────────────────────────
#  SIGNAL GENERATOR
# ─────────────────────────────────────────────

class SignalGenerator:
    def __init__(self, chain: MarkovChain, params: MarkovParams):
        self.chain = chain
        self.params = params

    def generate(self, returns_window: np.ndarray) -> tuple[str, float]:
        """
        Returns (signal, probability):
          signal ∈ {'LONG', 'SHORT', 'FLAT'}
          probability = confidence of the signal
        """
        if len(returns_window) < self.params.lookback:
            return "FLAT", 0.0

        # Rolling refit on window
        self.chain.fit(returns_window[-self.params.lookback:])
        last_ret = returns_window[-1]
        probs = self.chain.predict_next(last_ret)

        bull_p = self.chain.bullish_prob(probs)
        bear_p = self.chain.bearish_prob(probs)
        thresh = self.params.entry_threshold

        if bull_p >= thresh and bull_p > bear_p:
            return "LONG", bull_p
        elif bear_p >= thresh and bear_p > bull_p:
            return "SHORT", bear_p
        return "FLAT", max(bull_p, bear_p)


# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────

class Backtester:
    def __init__(self, params: MarkovParams):
        self.params = params

    def run(self, prices: np.ndarray, initial_capital: float = 10_000) -> BacktestResult:
        p = self.params
        chain = MarkovChain(p)
        sig_gen = SignalGenerator(chain, p)

        returns = np.diff(np.log(prices))
        n = len(prices)

        equity = initial_capital
        equity_curve = [equity]
        trades: list[TradeRecord] = []
        position: Optional[TradeRecord] = None

        min_bars = p.lookback + 5

        for i in range(min_bars, n):
            price = prices[i]
            ret_window = returns[:i]

            # ── Manage open position ──────────────────────
            if position is not None:
                move = (price - position.entry_price) / position.entry_price
                if position.direction == "SHORT":
                    move = -move

                sl_hit = move <= -p.stop_loss_pct
                tp_hit = move >= p.take_profit_pct
                hp_hit = (i - position.entry_bar) >= p.holding_period

                if sl_hit or tp_hit or hp_hit:
                    exit_price = price
                    raw_pnl = move * position.size
                    position.exit_price = exit_price
                    position.exit_bar = i
                    position.pnl = raw_pnl
                    position.exit_reason = ("SL" if sl_hit else "TP" if tp_hit else "HP")
                    equity += raw_pnl
                    trades.append(position)
                    equity_curve.append(equity)
                    position = None
                    continue

            # ── Generate signal ───────────────────────────
            if position is None:
                sig, prob = sig_gen.generate(ret_window)
                if sig != "FLAT":
                    size = equity * p.position_size_pct
                    position = TradeRecord(
                        entry_bar=i,
                        entry_price=price,
                        direction=sig,
                        size=size,
                        predicted_prob=prob,
                    )

            equity_curve.append(equity)

        # Force-close any open position at end
        if position is not None:
            price = prices[-1]
            move = (price - position.entry_price) / position.entry_price
            if position.direction == "SHORT":
                move = -move
            position.exit_price = price
            position.exit_bar = n - 1
            position.pnl = move * position.size
            position.exit_reason = "EOD"
            equity += position.pnl
            trades.append(position)

        equity_arr = np.array(equity_curve)
        ret_arr = np.diff(equity_arr) / equity_arr[:-1]

        # ── Metrics ───────────────────────────────────────
        sharpe = self._sharpe(ret_arr)
        mdd = self._max_drawdown(equity_arr)
        pnls = [t.pnl for t in trades]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1e-9
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        total_ret = (equity - initial_capital) / initial_capital

        score = self._score(sharpe, mdd, win_rate, pf, len(trades))

        return BacktestResult(
            params=p,
            trades=trades,
            equity_curve=equity_arr,
            returns=ret_arr,
            sharpe=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            profit_factor=pf,
            total_return=total_ret,
            n_trades=len(trades),
            score=score,
        )

    @staticmethod
    def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(np.sqrt(periods) * returns.mean() / returns.std())

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min())

    @staticmethod
    def _score(sharpe, mdd, win_rate, pf, n_trades) -> float:
        """
        Composite score balancing risk-adjusted return, drawdown, and activity.
        Higher is better.
        """
        if n_trades < 5:
            return -999.0
        dd_penalty = abs(mdd) * 2
        trade_bonus = min(np.log1p(n_trades) / 5, 0.5)
        return sharpe * (1 - dd_penalty) * (win_rate + 0.1) + trade_bonus


# ─────────────────────────────────────────────
#  PARAMETER OPTIMIZER
# ─────────────────────────────────────────────

class MarkovOptimizer:
    """
    Grid-search + iterative refinement over MarkovParams space.
    Pass a progress_callback(pct, best_score, best_params, iteration)
    for live UI updates.
    """

    SEARCH_SPACE = {
        "n_states":         [3, 5, 7, 9],
        "lookback":         [10, 20, 30, 50],
        "holding_period":   [1, 3, 5, 10],
        "entry_threshold":  [0.55, 0.60, 0.65, 0.70],
        "stop_loss_pct":    [0.003, 0.005, 0.008, 0.010],
        "take_profit_pct":  [0.006, 0.010, 0.015, 0.020],
        "position_size_pct":[0.01, 0.02, 0.03],
        "state_method":     ["quantile", "equal"],
    }

    def __init__(self, prices: np.ndarray, initial_capital: float = 10_000,
                 max_iterations: int = 3, progress_callback=None):
        self.prices = prices
        self.initial_capital = initial_capital
        self.max_iterations = max_iterations
        self.callback = progress_callback
        self.all_results: list[BacktestResult] = []
        self.best_result: Optional[BacktestResult] = None

    def _coarse_grid(self) -> list[MarkovParams]:
        """Generate coarse parameter combinations for first pass."""
        combos = []
        for n_states, lookback, hp, thresh in product(
            self.SEARCH_SPACE["n_states"],
            self.SEARCH_SPACE["lookback"],
            self.SEARCH_SPACE["holding_period"],
            self.SEARCH_SPACE["entry_threshold"],
        ):
            combos.append(MarkovParams(
                n_states=n_states,
                lookback=lookback,
                holding_period=hp,
                entry_threshold=thresh,
            ))
        return combos

    def _fine_grid(self, best: MarkovParams) -> list[MarkovParams]:
        """Refine around best params found so far."""
        combos = []
        for sl, tp, ps, sm in product(
            self.SEARCH_SPACE["stop_loss_pct"],
            self.SEARCH_SPACE["take_profit_pct"],
            self.SEARCH_SPACE["position_size_pct"],
            self.SEARCH_SPACE["state_method"],
        ):
            combos.append(MarkovParams(
                n_states=best.n_states,
                lookback=best.lookback,
                holding_period=best.holding_period,
                entry_threshold=best.entry_threshold,
                stop_loss_pct=sl,
                take_profit_pct=tp,
                position_size_pct=ps,
                state_method=sm,
            ))
        return combos

    def _micro_grid(self, best: MarkovParams) -> list[MarkovParams]:
        """Fine-tune numeric params around local best."""
        combos = []
        for delta_thresh in [-0.03, 0, 0.03]:
            for delta_sl in [-0.001, 0, 0.001]:
                for delta_tp in [-0.002, 0, 0.002]:
                    t = round(best.entry_threshold + delta_thresh, 3)
                    sl = round(best.stop_loss_pct + delta_sl, 4)
                    tp = round(best.take_profit_pct + delta_tp, 4)
                    if 0.50 < t < 0.90 and sl > 0.001 and tp > sl:
                        combos.append(MarkovParams(
                            n_states=best.n_states,
                            lookback=best.lookback,
                            holding_period=best.holding_period,
                            entry_threshold=t,
                            stop_loss_pct=sl,
                            take_profit_pct=tp,
                            position_size_pct=best.position_size_pct,
                            state_method=best.state_method,
                        ))
        return combos

    def optimize(self) -> BacktestResult:
        phases = [self._coarse_grid]
        for it in range(1, self.max_iterations):
            phases.append(self._fine_grid if it == 1 else self._micro_grid)

        total_phases = len(phases)
        bt = Backtester(MarkovParams())

        for phase_idx, phase_fn in enumerate(phases):
            if phase_idx == 0:
                combos = phase_fn()
            else:
                combos = phase_fn(self.best_result.params)

            n = len(combos)
            phase_results = []

            for i, params in enumerate(combos):
                bt.params = params
                try:
                    result = bt.run(self.prices, self.initial_capital)
                    phase_results.append(result)
                    self.all_results.append(result)

                    if self.best_result is None or result.score > self.best_result.score:
                        self.best_result = result
                except Exception:
                    pass

                # Progress callback
                if self.callback:
                    global_pct = (phase_idx / total_phases + (i + 1) / (n * total_phases)) * 100
                    self.callback(
                        pct=round(global_pct, 1),
                        best_score=self.best_result.score if self.best_result else 0,
                        best_params=self.best_result.params if self.best_result else params,
                        iteration=phase_idx + 1,
                        phase_name=["Coarse Grid", "Fine Tune", "Micro Tune"][min(phase_idx, 2)],
                        combo=i + 1,
                        total=n,
                    )

        return self.best_result

    def get_top_n(self, n: int = 10) -> list[BacktestResult]:
        return sorted(self.all_results, key=lambda r: r.score, reverse=True)[:n]
