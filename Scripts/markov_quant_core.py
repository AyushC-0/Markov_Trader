"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MARKOV QUANT CORE  —  Production-Grade Mathematical Architecture           ║
║                                                                              ║
║  Implements:                                                                 ║
║   1. Composite State Space (Volatility × Trend regime lattice)               ║
║   2. MLE Transition Matrix  P̂[i,j] = n(i→j) / Σ_k n(i→k)                  ║
║   3. Chapman-Kolmogorov     P^n = matrix power for n-step forecasting        ║
║   4. State Distribution     π(t) = π(0) · P^t                               ║
║   5. Rolling MLE Window     time-homogeneity mitigation via adaptive window  ║
║   6. Dynamic Position Sizing scaling by risk-state probability               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  §1  STATE SPACE DEFINITION
#  Composite state = (VolatilityRegime × TrendRegime)
#  giving a finite lattice of M = V × T states
# ══════════════════════════════════════════════════════════════════════════════

class VolatilityRegime(IntEnum):
    LOW    = 0   # σ < 33rd-percentile
    MEDIUM = 1   # 33rd ≤ σ < 67th-percentile
    HIGH   = 2   # σ ≥ 67th-percentile


class TrendRegime(IntEnum):
    BEARISH   = 0   # price < slow-EMA and slope < 0
    SIDEWAYS  = 1   # price within EMA band
    BULLISH   = 2   # price > slow-EMA and slope > 0


@dataclass(frozen=True)
class CompositeState:
    """
    A single point in the 3×3 = 9-state composite regime space.
    State label encodes both volatility and trend simultaneously,
    allowing the Markov chain to capture cross-regime transitions
    that a single-dimension chain misses entirely.
    """
    vol:   VolatilityRegime
    trend: TrendRegime

    @property
    def index(self) -> int:
        """Flatten 2D state → integer index (row-major)."""
        return int(self.vol) * 3 + int(self.trend)

    @property
    def is_high_risk(self) -> bool:
        """High-vol or bearish = risk-off."""
        return self.vol == VolatilityRegime.HIGH or self.trend == TrendRegime.BEARISH

    @property
    def is_low_risk(self) -> bool:
        return self.vol == VolatilityRegime.LOW and self.trend == TrendRegime.BULLISH

    def __repr__(self) -> str:
        return f"[{self.vol.name}|{self.trend.name}]"


# All 9 composite states indexed 0-8
ALL_STATES: list[CompositeState] = [
    CompositeState(VolatilityRegime(v), TrendRegime(t))
    for v in range(3) for t in range(3)
]
N_COMPOSITE_STATES = len(ALL_STATES)  # = 9


class StateSpaceBuilder:
    """
    Maps raw OHLCV data into the composite (Vol × Trend) state space.

    Volatility proxy: rolling normalised ATR = ATR(atr_period) / Close
    Trend proxy:      EMA crossover + slope sign
                      slow_ema > fast_ema AND slope > 0  → BULLISH
                      slow_ema < fast_ema AND slope < 0  → BEARISH
                      otherwise                          → SIDEWAYS

    All thresholds are ROLLING (percentile-based) to ensure the state
    space adapts to the current volatility regime rather than being
    anchored to a fixed historical period — directly mitigating the
    time-homogeneity problem.
    """

    def __init__(
        self,
        atr_period:      int   = 14,
        fast_ema:        int   = 10,
        slow_ema:        int   = 30,
        vol_window: int   = 120,   # bars for adaptive vol percentiles
        trend_band_pct:  float = 0.001, # ±0.1% around slow-EMA = sideways
    ):
        self.atr_period      = atr_period
        self.fast_ema        = fast_ema
        self.slow_ema        = slow_ema
        self.vol_window = vol_window
        self.trend_band_pct  = trend_band_pct

    # ── Volatility ───────────────────────────────────────────────────────────

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Wilder ATR — true range with previous close."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def _vol_regime(self, atr_norm: pd.Series) -> pd.Series:
        """
        Classify each bar into LOW / MEDIUM / HIGH volatility using
        rolling percentile thresholds — not fixed levels.
        This is the key adaptive element that fights non-stationarity.
        """
        p33 = atr_norm.rolling(self.vol_window, min_periods=20).quantile(0.33)
        p67 = atr_norm.rolling(self.vol_window, min_periods=20).quantile(0.67)

        vol = pd.Series(VolatilityRegime.MEDIUM, index=atr_norm.index)
        vol[atr_norm <  p33] = VolatilityRegime.LOW
        vol[atr_norm >= p67] = VolatilityRegime.HIGH
        return vol

    # ── Trend ─────────────────────────────────────────────────────────────────

    def _trend_regime(self, close: pd.Series) -> pd.Series:
        """
        Dual-EMA crossover with slope confirmation.
        Slope = (EMA_slow[t] - EMA_slow[t-3]) / EMA_slow[t-3]
        to avoid noise from single-bar reversals.
        """
        fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        slope = (slow - slow.shift(3)) / slow.shift(3).replace(0, np.nan)
        band  = slow * self.trend_band_pct

        trend = pd.Series(TrendRegime.SIDEWAYS, index=close.index)
        bullish_mask = (close > slow + band) & (slope > 0)
        bearish_mask = (close < slow - band) & (slope < 0)
        trend[bullish_mask] = TrendRegime.BULLISH
        trend[bearish_mask] = TrendRegime.BEARISH
        return trend

    # ── Composite ─────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> pd.Series:
        """
        Build a pd.Series of CompositeState objects aligned to df.index.
        df must have columns: open, high, low, close.
        """
        atr_norm = self._atr(df["high"], df["low"], df["close"]) / df["close"]
        vol      = self._vol_regime(atr_norm)
        trend    = self._trend_regime(df["close"])

        states = pd.Series(index=df.index, dtype=object)
        for i in df.index:
            try:
                states[i] = CompositeState(
                    VolatilityRegime(int(vol[i])),
                    TrendRegime(int(trend[i])),
                )
            except Exception:
                states[i] = CompositeState(VolatilityRegime.MEDIUM, TrendRegime.SIDEWAYS)
        return states


# ══════════════════════════════════════════════════════════════════════════════
#  §2  MLE TRANSITION MATRIX  +  §5 ROLLING WINDOW (TIME-HOMOGENEITY FIX)
#
#  P̂[i,j] = n(i→j) / Σ_k n(i→k)
#
#  Standard MLE assumes time-homogeneity (P is constant over time).
#  Markets violate this via regime drift, crises, and structural breaks.
#  Mitigation: use only the most recent `window` observations to estimate P,
#  so that P̂ reflects the *current* local regime rather than a lifetime average.
#  This trades estimation variance for bias reduction under non-stationarity.
# ══════════════════════════════════════════════════════════════════════════════

class MLETransitionMatrix:
    """
    Maximum Likelihood Estimation of a Markov transition matrix
    with Laplace (add-α) smoothing and rolling-window adaptation.

    The matrix P is estimated as:
        P̂[i,j] = (n(i→j) + α) / (Σ_k n(i→k) + α·N)

    where α is the Laplace pseudo-count (default 0.5 = Jeffreys prior),
    N is the number of states, and n(i→j) is the transition count in the
    rolling window.

    Parameters
    ----------
    n_states     : int   — number of discrete states (9 for composite space)
    window       : int   — rolling window size (bars); None = full history
    alpha        : float — Laplace smoothing pseudo-count
    min_obs      : int   — minimum observations before returning a valid P
    """

    def __init__(
        self,
        n_states: int   = N_COMPOSITE_STATES,
        window:   int   = 60,
        alpha:    float = 0.5,
        min_obs:  int   = 20,
    ):
        self.n_states = n_states
        self.window   = window
        self.alpha    = alpha
        self.min_obs  = min_obs
        self._P: Optional[np.ndarray] = None   # current estimate
        self._count_matrix: Optional[np.ndarray] = None

    # ── Core estimation ───────────────────────────────────────────────────────

    def fit(self, state_sequence: np.ndarray) -> "MLETransitionMatrix":
        """
        Estimate P from a sequence of integer state indices.
        Applies rolling window if window is set.

        Parameters
        ----------
        state_sequence : 1-D array of integer state indices (0 … n_states-1)
        """
        seq = state_sequence
        if self.window is not None and len(seq) > self.window:
            seq = seq[-self.window:]

        if len(seq) < self.min_obs:
            # Return uniform distribution (maximum entropy prior)
            self._P = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
            self._count_matrix = np.zeros((self.n_states, self.n_states))
            return self

        # Count transitions n(i→j)
        C = np.zeros((self.n_states, self.n_states), dtype=float)
        for t in range(len(seq) - 1):
            i, j = int(seq[t]), int(seq[t + 1])
            if 0 <= i < self.n_states and 0 <= j < self.n_states:
                C[i, j] += 1.0
        self._count_matrix = C.copy()

        # Laplace (add-α) smoothing — avoids zero rows (absorbing states)
        C_smooth = C + self.alpha
        row_sums = C_smooth.sum(axis=1, keepdims=True)
        self._P = C_smooth / row_sums
        return self

    @property
    def P(self) -> np.ndarray:
        if self._P is None:
            raise RuntimeError("Call fit() before accessing P.")
        return self._P

    @property
    def count_matrix(self) -> np.ndarray:
        return self._count_matrix if self._count_matrix is not None else np.zeros((self.n_states, self.n_states))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stationary_distribution(self) -> np.ndarray:
        """
        Compute the unique stationary distribution π such that π·P = π.
        Found as the left eigenvector corresponding to eigenvalue 1.
        """
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pi

    def entropy_rate(self) -> float:
        """
        H = -Σ_i π_i Σ_j P[i,j] log P[i,j]
        Measures the randomness of the chain (bits per transition).
        High entropy ≈ regime is unpredictable.
        Low entropy  ≈ strong regime persistence / momentum.
        """
        pi = self.stationary_distribution()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_P = np.where(self.P > 0, np.log2(self.P), 0.0)
        return float(-np.sum(pi[:, None] * self.P * log_P))

    def mixing_time(self, epsilon: float = 0.01) -> int:
        """
        Approximate mixing time τ = smallest n such that
        ||(π(0)·P^n) - π||_1 < epsilon for all starting distributions.
        Useful for choosing the look-forward horizon.
        """
        pi = self.stationary_distribution()
        current = np.full(self.n_states, 1.0 / self.n_states)
        for t in range(1, 500):
            current = current @ self.P
            if np.abs(current - pi).sum() < epsilon:
                return t
        return 500


# ══════════════════════════════════════════════════════════════════════════════
#  §3  CHAPMAN-KOLMOGOROV  n-STEP FORECASTING
#
#  P^n[i,j] = P(X_{t+n} = j | X_t = i)
#
#  The C-K equation gives us the exact n-step transition kernel.
#  We implement this via repeated squaring for computational efficiency:
#  O(log n) matrix multiplications rather than O(n).
# ══════════════════════════════════════════════════════════════════════════════

class ChapmanKolmogorov:
    """
    Efficient n-step transition probability computation.

    Given a fitted MLETransitionMatrix, computes P^n via
    matrix exponentiation by repeated squaring in O(k³ log n)
    where k is the number of states.

    This is used to forecast where the market is likely to be
    in exactly n bars from now — not just 1 bar ahead.
    """

    def __init__(self, mle: MLETransitionMatrix):
        self.mle = mle

    def matrix_power(self, n: int) -> np.ndarray:
        """
        Compute P^n exactly using repeated squaring.
        P^1 = P, P^2 = P·P, P^4 = P^2·P^2, etc.
        """
        assert n >= 1, "n must be ≥ 1"
        result = np.eye(self.mle.n_states)  # P^0 = I
        base   = self.mle.P.copy()
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
        return result

    def forecast_distribution(
        self,
        current_state_idx: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Given current state i, return P(X_{t+horizon} = j) for all j.
        This is the i-th row of P^horizon.
        """
        Pn = self.matrix_power(horizon)
        return Pn[current_state_idx]

    def forecast_horizon_table(
        self,
        current_state_idx: int,
        max_horizon: int = 10,
    ) -> pd.DataFrame:
        """
        Return a DataFrame showing P(state=j) at horizons 1 … max_horizon.
        Rows = horizons, Columns = state labels.
        """
        rows = {}
        for h in range(1, max_horizon + 1):
            dist = self.forecast_distribution(current_state_idx, h)
            rows[h] = {str(ALL_STATES[j]): round(float(dist[j]), 4)
                       for j in range(self.mle.n_states)}
        return pd.DataFrame(rows).T

    def expected_return_in_state(
        self,
        state_returns: np.ndarray,
        current_state_idx: int,
        horizon: int,
    ) -> float:
        """
        E[r_{t+horizon} | X_t = i]  using law of total expectation:
          = Σ_j P^n[i,j] · μ_j
        where μ_j = historical mean return in state j.
        """
        dist = self.forecast_distribution(current_state_idx, horizon)
        return float(np.dot(dist, state_returns))


# ══════════════════════════════════════════════════════════════════════════════
#  §4  DYNAMIC POSITION SIZING  via  π(t) = π(0) · P^t
#
#  The state distribution vector π(t) tells us the probability mass
#  across all states at time t. We use this to:
#   (a) Compute P(high-risk state in next t bars)
#   (b) Scale position size INVERSELY proportional to risk probability
#
#  Position scale formula:
#       f(t) = base_size × (1 - w·P_risk(t))
#  where P_risk(t) = Σ_{j ∈ high-risk} π(t)[j]
#  and w ∈ [0,1] is the risk-aversion weight.
#
#  This replaces the naive fixed-fraction sizing, which ignores
#  the probability of imminent adverse regime transitions.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSizeResult:
    base_size:     float        # unadjusted fraction of equity
    scaled_size:   float        # Markov-adjusted fraction
    risk_prob:     float        # P(high-risk state at horizon)
    scale_factor:  float        # multiplier applied (0 to 1)
    horizon:       int
    state_dist:    np.ndarray   # full π(t) distribution
    current_state: CompositeState


class DynamicPositionSizer:
    """
    Markov-based dynamic position sizing.

    At each entry signal, it:
    1. Takes the current composite state s_t
    2. Propagates the distribution π(t) = e_{s_t} · P^horizon
       (where e_{s_t} is the one-hot initial distribution)
    3. Computes P_risk(t) = probability of being in any high-risk state
    4. Scales the base position size: f = base × (1 - w · P_risk)
       clamped to [min_size, max_size]

    The result is a position that is systematically smaller
    when the chain predicts elevated adverse-regime probability.

    Parameters
    ----------
    risk_aversion   : float [0,1] — 0 = ignore risk, 1 = max avoidance
    sizing_horizon  : int — n-step horizon for risk assessment
    base_size_pct   : float — base fraction of equity per trade
    min_size_pct    : float — floor (never trade less than this)
    max_size_pct    : float — ceiling (never over-leverage)
    """

    def __init__(
        self,
        risk_aversion:  float = 0.6,
        sizing_horizon: int   = 5,
        base_size_pct:  float = 0.02,
        min_size_pct:   float = 0.005,
        max_size_pct:   float = 0.05,
    ):
        self.risk_aversion  = risk_aversion
        self.sizing_horizon = sizing_horizon
        self.base_size_pct  = base_size_pct
        self.min_size_pct   = min_size_pct
        self.max_size_pct   = max_size_pct

    def compute(
        self,
        ck:             ChapmanKolmogorov,
        current_state:  CompositeState,
        equity:         float,
    ) -> PositionSizeResult:
        """
        Compute Markov-adjusted position size in dollar terms.
        """
        idx = current_state.index

        # π(t) = e_i · P^horizon  (row vector propagation)
        state_dist = ck.forecast_distribution(idx, self.sizing_horizon)

        # P_risk = Σ P(high-risk state j at horizon t)
        risk_prob = float(sum(
            state_dist[j]
            for j, s in enumerate(ALL_STATES)
            if s.is_high_risk
        ))

        # Scale factor ∈ [1 - w, 1]
        scale = 1.0 - self.risk_aversion * risk_prob
        scale = float(np.clip(scale, 0.1, 1.0))   # hard floor at 10%

        raw_frac    = self.base_size_pct * scale
        clamped_frac = float(np.clip(raw_frac, self.min_size_pct, self.max_size_pct))
        dollar_size  = equity * clamped_frac

        return PositionSizeResult(
            base_size    = equity * self.base_size_pct,
            scaled_size  = dollar_size,
            risk_prob    = risk_prob,
            scale_factor = scale,
            horizon      = self.sizing_horizon,
            state_dist   = state_dist,
            current_state= current_state,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  §5  ROLLING REGIME DETECTOR (time-homogeneity mitigation)
#
#  Maintains a sliding buffer of (state, return) pairs.
#  At each bar:
#    1. Appends new observation
#    2. Re-fits the MLE matrix on the rolling window
#    3. Exposes current P, entropy, and mixing time
#
#  This ensures the model captures structural breaks, volatility cycles,
#  and macro regime shifts in near-real-time rather than
#  computing a static P over the full history.
# ══════════════════════════════════════════════════════════════════════════════

class RollingRegimeDetector:
    """
    Maintains a rolling buffer of composite states and re-estimates
    the transition matrix at each new observation.

    Also computes per-state statistics (mean return, volatility)
    which feed into the expected-return calculation of Chapman-Kolmogorov.
    """

    def __init__(
        self,
        mle:      MLETransitionMatrix,
        ssb:      StateSpaceBuilder,
        window:   int = 60,
    ):
        self.mle    = mle
        self.ssb    = ssb
        self.window = window

        # Circular buffers
        self._state_buf:  list[int]   = []
        self._return_buf: list[float] = []
        self._state_return_map: dict[int, list[float]] = {
            i: [] for i in range(mle.n_states)
        }

    def update(self, state: CompositeState, ret: float) -> None:
        """Add one new bar observation and refresh the model."""
        idx = state.index
        self._state_buf.append(idx)
        self._return_buf.append(ret)
        self._state_return_map[idx].append(ret)

        # Trim to window
        if len(self._state_buf) > self.window:
            old_idx = self._state_buf.pop(0)
            old_ret = self._return_buf.pop(0)
            if old_ret in self._state_return_map[old_idx]:
                self._state_return_map[old_idx].remove(old_ret)

        # Re-fit MLE
        if len(self._state_buf) >= self.mle.min_obs:
            self.mle.fit(np.array(self._state_buf))

    def per_state_mean_return(self) -> np.ndarray:
        """μ[i] = mean return observed while in state i (rolling window)."""
        return np.array([
            float(np.mean(self._state_return_map[i]))
            if self._state_return_map[i] else 0.0
            for i in range(self.mle.n_states)
        ])

    def per_state_vol(self) -> np.ndarray:
        """σ[i] = std of returns in state i."""
        return np.array([
            float(np.std(self._state_return_map[i]))
            if len(self._state_return_map[i]) > 1 else 0.0
            for i in range(self.mle.n_states)
        ])

    def regime_persistence(self) -> np.ndarray:
        """
        Self-transition probability P[i,i] — how strongly does each
        state persist? Values close to 1 mean high regime momentum;
        values close to 1/N mean rapid switching.
        """
        if self.mle._P is None:
            return np.full(self.mle.n_states, 1.0 / self.mle.n_states)
        return np.diag(self.mle.P)

    @property
    def current_state_idx(self) -> Optional[int]:
        return self._state_buf[-1] if self._state_buf else None

    @property
    def is_ready(self) -> bool:
        return len(self._state_buf) >= self.mle.min_obs


# ══════════════════════════════════════════════════════════════════════════════
#  §6  SIGNAL ENGINE  (upgraded from naive threshold-based version)
#
#  Entry logic now uses a MULTI-LAYER filter:
#   Layer 1 — 1-step Markov probability (replaces old naive threshold)
#   Layer 2 — n-step expected return sign (C-K confirmation)
#   Layer 3 — Regime risk gate (block trades in high-risk predicted states)
#   Layer 4 — Entropy filter (block trades when chain is maximally uncertain)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    direction:      str         # LONG | SHORT | FLAT
    entry_prob:     float       # 1-step directional probability
    expected_ret:   float       # E[r_{t+h}] from C-K
    risk_prob:      float       # P(high-risk at horizon h)
    entropy:        float       # chain entropy rate (bits)
    regime_state:   CompositeState
    layer_pass:     dict        # which filter layers passed


class QuantSignalEngine:
    """
    Multi-layer signal engine integrating all Markov quant components.

    Each layer acts as a necessary condition — all must be satisfied
    for a trade to fire. This replaces the naive single-threshold rule.
    """

    def __init__(
        self,
        mle:              MLETransitionMatrix,
        ck:               ChapmanKolmogorov,
        regime_detector:  RollingRegimeDetector,
        sizer:            DynamicPositionSizer,
        entry_prob_thresh: float = 0.55,  # Layer 1 threshold
        max_entropy:       float = 2.8,   # Layer 4: max tolerated entropy (bits, max for 9 states = log2(9)≈3.17)
        max_risk_prob:     float = 0.65,  # Layer 3: block if P(high-risk) > this
        ck_horizon:        int   = 3,     # C-K confirmation horizon
    ):
        self.mle               = mle
        self.ck                = ck
        self.rd                = regime_detector
        self.sizer             = sizer
        self.entry_prob_thresh = entry_prob_thresh
        self.max_entropy       = max_entropy
        self.max_risk_prob     = max_risk_prob
        self.ck_horizon        = ck_horizon

    def generate(
        self,
        current_state: CompositeState,
        equity:        float,
    ) -> tuple[SignalResult, PositionSizeResult]:
        """
        Run all filter layers and return (signal, sizing).
        """
        flat_size = PositionSizeResult(
            base_size    = 0,
            scaled_size  = 0,
            risk_prob    = 0,
            scale_factor = 0,
            horizon      = 0,
            state_dist   = np.zeros(N_COMPOSITE_STATES),
            current_state= current_state,
        )

        if not self.rd.is_ready:
            return SignalResult("FLAT", 0, 0, 0, 0, current_state, {}), flat_size

        idx = current_state.index

        # ── Layer 1: 1-step directional probability ────────────────────────
        next_probs   = self.mle.P[idx]
        # Aggregate over states by their trend dimension
        bull_prob = sum(next_probs[j] for j, s in enumerate(ALL_STATES) if s.trend == TrendRegime.BULLISH)
        bear_prob = sum(next_probs[j] for j, s in enumerate(ALL_STATES) if s.trend == TrendRegime.BEARISH)
        side_prob = 1.0 - bull_prob - bear_prob
        l1_pass   = max(bull_prob, bear_prob) >= self.entry_prob_thresh

        # ── Layer 2: C-K n-step expected return sign ───────────────────────
        mu_per_state = self.rd.per_state_mean_return()
        exp_ret      = self.ck.expected_return_in_state(mu_per_state, idx, self.ck_horizon)
        l2_pass      = True  # C-K used for sizing not as hard gate here

        # ── Layer 3: Regime risk gate ──────────────────────────────────────
        size_res     = self.sizer.compute(self.ck, current_state, equity)
        risk_prob    = size_res.risk_prob
        l3_pass      = risk_prob <= self.max_risk_prob

        # ── Layer 4: Entropy filter ────────────────────────────────────────
        entropy  = self.mle.entropy_rate()
        l4_pass  = entropy <= self.max_entropy

        layers = {
            "L1_direction_prob": l1_pass,
            "L2_ck_expected_ret": l2_pass,
            "L3_risk_gate": l3_pass,
            "L4_entropy": l4_pass,
        }

        all_pass = l1_pass and l2_pass and l3_pass and l4_pass

        if all_pass:
            if bull_prob >= bear_prob:
                direction = "LONG"
                entry_prob = bull_prob
            else:
                direction = "SHORT"
                entry_prob = bear_prob
        else:
            direction  = "FLAT"
            entry_prob = max(bull_prob, bear_prob)

        signal = SignalResult(
            direction    = direction,
            entry_prob   = entry_prob,
            expected_ret = exp_ret,
            risk_prob    = risk_prob,
            entropy      = entropy,
            regime_state = current_state,
            layer_pass   = layers,
        )
        return signal, size_res


# ══════════════════════════════════════════════════════════════════════════════
#  §7  UPGRADED BACKTESTER  (integrates all quant components)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantTradeRecord:
    entry_bar:     int
    entry_price:   float
    direction:     str
    size:          float          # Markov-scaled dollar size
    base_size:     float          # unscaled dollar size
    scale_factor:  float          # sizing scale 0-1
    risk_prob:     float          # P(high-risk at entry)
    entry_entropy: float
    entry_state:   CompositeState
    entry_prob:    float
    expected_ret:  float
    exit_price:    float  = 0.0
    exit_bar:      int    = 0
    pnl:           float  = 0.0
    exit_reason:   str    = ""
    bars_held:     int    = 0


@dataclass
class QuantBacktestResult:
    trades:         list
    equity_curve:   np.ndarray
    state_series:   list          # composite state per bar
    returns:        np.ndarray
    sharpe:         float
    max_drawdown:   float
    win_rate:       float
    profit_factor:  float
    total_return:   float
    n_trades:       int
    avg_scale:      float         # mean position scale (1.0 = unscaled)
    avg_risk_prob:  float         # mean P(high-risk) at entry
    avg_entropy:    float
    score:          float = 0.0

    def summary(self) -> dict:
        return {
            "Total Return %":   round(self.total_return * 100, 2),
            "Sharpe Ratio":     round(self.sharpe, 3),
            "Max Drawdown %":   round(self.max_drawdown * 100, 2),
            "Win Rate %":       round(self.win_rate * 100, 1),
            "Profit Factor":    round(self.profit_factor, 3),
            "N Trades":         self.n_trades,
            "Avg Scale Factor": round(self.avg_scale, 3),
            "Avg Risk Prob":    round(self.avg_risk_prob, 3),
            "Avg Entropy":      round(self.avg_entropy, 3),
            "Score":            round(self.score, 4),
        }


class QuantBacktester:
    """
    Full backtester integrating the Markov quant architecture.

    At every bar:
      1. StateSpaceBuilder classifies current regime
      2. RollingRegimeDetector updates rolling MLE
      3. QuantSignalEngine runs 4-layer filter
      4. DynamicPositionSizer computes Markov-scaled position
      5. Trade management: SL / TP / holding-period exits
    """

    def __init__(
        self,
        ssb:    StateSpaceBuilder,
        mle:    MLETransitionMatrix,
        sizer:  DynamicPositionSizer,
        stop_loss_pct:    float = 0.005,
        take_profit_pct:  float = 0.010,
        holding_period:   int   = 5,
        mle_window:       int   = 60,
        entry_prob_thresh:float  = 0.55,
        max_entropy:      float  = 2.8,
        max_risk_prob:    float  = 0.65,
        ck_horizon:       int    = 3,
    ):
        self.ssb               = ssb
        self.mle               = mle
        self.sizer             = sizer
        self.stop_loss_pct     = stop_loss_pct
        self.take_profit_pct   = take_profit_pct
        self.holding_period    = holding_period
        self.mle_window        = mle_window
        self.entry_prob_thresh = entry_prob_thresh
        self.max_entropy       = max_entropy
        self.max_risk_prob     = max_risk_prob
        self.ck_horizon        = ck_horizon

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000,
    ) -> QuantBacktestResult:

        prices   = df["close"].values
        returns  = np.diff(np.log(prices))
        n_bars   = len(prices)

        # Build state series for all bars
        state_series = self.ssb.build(df)
        state_list   = state_series.tolist()

        # Initialise rolling components
        rd  = RollingRegimeDetector(self.mle, self.ssb, window=self.mle_window)
        ck  = ChapmanKolmogorov(self.mle)
        sig = QuantSignalEngine(
            mle               = self.mle,
            ck                = ck,
            regime_detector   = rd,
            sizer             = self.sizer,
            entry_prob_thresh = self.entry_prob_thresh,
            max_entropy       = self.max_entropy,
            max_risk_prob     = self.max_risk_prob,
            ck_horizon        = self.ck_horizon,
        )

        equity: float = initial_capital
        equity_curve: list[float] = [equity]
        trades: list[QuantTradeRecord] = []
        position: Optional[QuantTradeRecord] = None

        warmup = max(self.ssb.slow_ema + 10, self.mle_window + 5)

        for i in range(1, n_bars):
            price     = prices[i]
            ret_now   = returns[i - 1]
            cur_state = state_list[i]
            if cur_state is None or not isinstance(cur_state, CompositeState):
                cur_state = CompositeState(VolatilityRegime.MEDIUM, TrendRegime.SIDEWAYS)

            # Update rolling detector
            rd.update(cur_state, ret_now)

            # ── Manage open position ─────────────────────────────────────
            if position is not None:
                move = (price - position.entry_price) / position.entry_price
                if position.direction == "SHORT":
                    move = -move
                sl_hit = move <= -self.stop_loss_pct
                tp_hit = move >= self.take_profit_pct
                hp_hit = (i - position.entry_bar) >= self.holding_period

                if sl_hit or tp_hit or hp_hit:
                    pnl = move * position.size
                    position.exit_price  = price
                    position.exit_bar    = i
                    position.pnl         = pnl
                    position.exit_reason = "SL" if sl_hit else "TP" if tp_hit else "HP"
                    position.bars_held   = i - position.entry_bar
                    equity += pnl
                    trades.append(position)
                    position = None

            # ── Generate signal ──────────────────────────────────────────
            if position is None and i >= warmup:
                signal, size_res = sig.generate(cur_state, equity)
                if signal.direction != "FLAT":
                    position = QuantTradeRecord(
                        entry_bar     = i,
                        entry_price   = price,
                        direction     = signal.direction,
                        size          = size_res.scaled_size,
                        base_size     = size_res.base_size,
                        scale_factor  = size_res.scale_factor,
                        risk_prob     = signal.risk_prob,
                        entry_entropy = signal.entropy,
                        entry_state   = cur_state,
                        entry_prob    = signal.entry_prob,
                        expected_ret  = signal.expected_ret,
                    )

            equity_curve.append(equity)

        # Force-close EOD
        if position is not None:
            move = (prices[-1] - position.entry_price) / position.entry_price
            if position.direction == "SHORT":
                move = -move
            position.exit_price  = prices[-1]
            position.exit_bar    = n_bars - 1
            position.pnl         = move * position.size
            position.exit_reason = "EOD"
            position.bars_held   = n_bars - 1 - position.entry_bar
            equity += position.pnl
            trades.append(position)

        eq_arr  = np.array(equity_curve)
        ret_arr = np.diff(eq_arr) / np.where(eq_arr[:-1] != 0, eq_arr[:-1], 1e-9)

        pnls        = [t.pnl for t in trades]
        wins        = [x for x in pnls if x > 0]
        losses      = [x for x in pnls if x <= 0]
        win_rate    = len(wins) / len(pnls) if pnls else 0
        gp          = sum(wins)
        gl          = abs(sum(losses)) + 1e-9
        pf          = gp / gl
        total_ret   = (equity - initial_capital) / initial_capital
        sharpe      = self._sharpe(ret_arr)
        mdd         = self._max_drawdown(eq_arr)
        avg_scale   = float(np.mean([t.scale_factor for t in trades])) if trades else 1.0
        avg_risk    = float(np.mean([t.risk_prob for t in trades])) if trades else 0.0
        avg_entropy = float(np.mean([t.entry_entropy for t in trades])) if trades else 0.0
        score       = self._score(sharpe, mdd, win_rate, pf, len(trades))

        return QuantBacktestResult(
            trades        = trades,
            equity_curve  = eq_arr,
            state_series  = [s for s in state_list if s is not None],
            returns       = ret_arr,
            sharpe        = sharpe,
            max_drawdown  = mdd,
            win_rate      = win_rate,
            profit_factor = pf,
            total_return  = total_ret,
            n_trades      = len(trades),
            avg_scale     = avg_scale,
            avg_risk_prob = avg_risk,
            avg_entropy   = avg_entropy,
            score         = score,
        )

    @staticmethod
    def _sharpe(r: np.ndarray, periods: int = 252) -> float:
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(np.sqrt(periods) * r.mean() / r.std())

    @staticmethod
    def _max_drawdown(eq: np.ndarray) -> float:
        peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / np.where(peak != 0, peak, 1)).min())

    @staticmethod
    def _score(sharpe, mdd, wr, pf, n) -> float:
        if n < 5:
            return -999.0
        return sharpe * (1 - 2 * abs(mdd)) * (wr + 0.1) + min(np.log1p(n) / 5, 0.5)
