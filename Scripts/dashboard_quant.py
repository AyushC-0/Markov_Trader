"""
╔══════════════════════════════════════════════════════════════════════╗
║   MARKOV QUANT FOREX TRADER  ·  Upgraded Dashboard                  ║
║   Run: streamlit run dashboard_quant.py                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

from data_feed import fetch_forex_data, FOREX_PAIRS, TIMEFRAMES
from markov_quant_core import (
    StateSpaceBuilder, MLETransitionMatrix, ChapmanKolmogorov,
    DynamicPositionSizer, QuantBacktester, QuantBacktestResult,
    ALL_STATES, N_COMPOSITE_STATES, TrendRegime, VolatilityRegime,
    CompositeState,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Markov Quant Forex",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg:#080c18; --panel:#0d1425; --border:#172040;
  --cyan:#00e5ff; --violet:#7c3aed; --green:#00ff88;
  --red:#ff4466; --gold:#ffd700; --amber:#ffaa00;
  --text:#b8cce8; --muted:#3a5070;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border)!important;}
h1,h2,h3,h4{font-family:'JetBrains Mono',monospace!important;color:var(--cyan)!important;letter-spacing:.04em;}
.metric-card{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--cyan);border-radius:4px;padding:14px 18px;margin:3px 0;}
.metric-card .lbl{font-size:9px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;font-family:'JetBrains Mono',monospace;}
.metric-card .val{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--cyan);}
.metric-card.g .val{color:var(--green);}.metric-card.g{border-left-color:var(--green);}
.metric-card.r .val{color:var(--red);}.metric-card.r{border-left-color:var(--red);}
.metric-card.a .val{color:var(--amber);}.metric-card.a{border-left-color:var(--amber);}
.metric-card.v .val{color:#a78bfa;}.metric-card.v{border-left-color:#a78bfa;}
.stButton button{background:linear-gradient(135deg,var(--violet),var(--cyan))!important;color:#fff!important;border:none!important;border-radius:4px!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;letter-spacing:.08em!important;padding:10px 24px!important;}
[data-testid="stMetric"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:4px!important;padding:10px!important;}
.state-chip{display:inline-block;padding:3px 8px;border-radius:3px;font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:600;}
.state-bull{background:rgba(0,255,136,.12);color:var(--green);}
.state-bear{background:rgba(255,68,102,.12);color:var(--red);}
.state-side{background:rgba(255,215,0,.12);color:var(--gold);}
.state-hvol{background:rgba(255,100,0,.12);color:#ff6400;}
.state-mvol{background:rgba(0,229,255,.08);color:var(--cyan);}
.state-lvol{background:rgba(0,255,136,.08);color:var(--green);}
.critique-box{background:linear-gradient(135deg,rgba(124,58,237,.08),rgba(0,229,255,.05));border:1px solid rgba(124,58,237,.3);border-radius:8px;padding:20px 24px;margin:12px 0;}
.math-box{background:rgba(13,20,37,.8);border:1px solid var(--border);border-radius:6px;padding:16px 20px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#7eb8e0;margin:8px 0;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    for k, v in {
        "qresult": None, "df": None,
        "mle_obj": None, "ck_obj": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

def mc(label, val, cls=""):
    st.markdown(f'<div class="metric-card {cls}"><div class="lbl">{label}</div><div class="val">{val}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:18px 0 8px;border-bottom:1px solid #172040;margin-bottom:20px;">
  <h1 style="font-size:26px;margin:0;">🧮 MARKOV QUANT FOREX TRADER</h1>
  <p style="font-family:'JetBrains Mono';font-size:11px;color:#3a5070;margin:5px 0 0;">
    MLE Transition Matrix · Chapman-Kolmogorov Forecasting · Dynamic Position Sizing · Rolling Regime Detection
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ MARKET CONFIGURATION")
    st.markdown("---")
    pair      = st.selectbox("Forex Pair", list(FOREX_PAIRS.keys()))
    timeframe = st.selectbox("Timeframe",  list(TIMEFRAMES.keys()))
    c1, c2    = st.columns(2)
    with c1: start_date = st.date_input("Start", date.today() - timedelta(days=730))
    with c2: end_date   = st.date_input("End",   date.today())
    init_cap  = st.number_input("Initial Capital ($)", 100, 10_000_000, 10_000, 500)

    st.markdown("---")
    st.markdown("### 🧬 STATE SPACE")
    atr_period   = st.slider("ATR Period",     5,  30, 14)
    fast_ema_p   = st.slider("Fast EMA",       5,  30, 10)
    slow_ema_p   = st.slider("Slow EMA",      10, 100, 30)
    vol_window   = st.slider("Vol Roll Window",20, 200, 80)

    st.markdown("---")
    st.markdown("### 📐 MLE / ROLLING WINDOW")
    mle_window   = st.slider("MLE Window (bars)",   20, 200, 60,
                             help="Rolling window for time-homogeneity mitigation")
    alpha_smooth = st.slider("Laplace α (smoothing)", 0.01, 2.0, 0.5, 0.01,
                             help="Pseudo-count — higher = more regularisation")

    st.markdown("---")
    st.markdown("### 🔭 SIGNAL ENGINE")
    entry_thresh  = st.slider("Entry Prob Threshold", 0.40, 0.85, 0.55, 0.01)
    max_entropy   = st.slider("Max Chain Entropy (bits)", 1.5, 3.17, 2.8, 0.05,
                              help="Block entries when chain is too uncertain. Max for 9 states ≈ 3.17")
    max_risk_p    = st.slider("Max Risk Prob Gate",  0.30, 0.95, 0.65, 0.05,
                              help="Block if P(high-risk state) exceeds this")
    ck_horizon    = st.slider("C-K Forecast Horizon", 1, 20, 3,
                              help="n-step look-ahead for expected return & risk prob")

    st.markdown("---")
    st.markdown("### 💰 DYNAMIC SIZING")
    base_size_p    = st.slider("Base Position %",  0.5,  8.0, 2.0, 0.5) / 100
    risk_aversion  = st.slider("Risk Aversion",    0.0,  1.0, 0.6, 0.05,
                               help="0 = ignore risk, 1 = max position reduction")
    sizing_horizon = st.slider("Sizing Horizon",    1,   10,  5)
    min_size_p     = st.slider("Min Position %",   0.1,  2.0, 0.5, 0.1) / 100
    max_size_p     = st.slider("Max Position %",   1.0, 15.0, 5.0, 0.5) / 100

    st.markdown("---")
    st.markdown("### 📉 TRADE MANAGEMENT")
    stop_loss_p    = st.slider("Stop Loss %",   0.1, 3.0, 0.5, 0.05) / 100
    take_profit_p  = st.slider("Take Profit %", 0.2, 6.0, 1.0, 0.1)  / 100
    holding_p      = st.slider("Holding Period (bars)", 1, 30, 5)

    st.markdown("---")
    run_btn = st.button("▶  RUN QUANT BACKTEST", use_container_width=True)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {pair} data …"):
        df = fetch_forex_data(pair, str(start_date), str(end_date),
                              TIMEFRAMES[timeframe]["interval"])
        st.session_state["df"] = df

    if df is None or len(df) < 100:
        st.error("Not enough data. Try a wider date range.")
    else:
        progress = st.progress(0, text="Building state space …")

        ssb = StateSpaceBuilder(
            atr_period=atr_period, fast_ema=fast_ema_p, slow_ema=slow_ema_p,
            vol_window=vol_window,
        )
        progress.progress(15, text="Initialising MLE matrix …")

        mle = MLETransitionMatrix(
            n_states=N_COMPOSITE_STATES,
            window=mle_window,
            alpha=alpha_smooth,
        )
        progress.progress(30, text="Calibrating Chapman-Kolmogorov …")

        sizer = DynamicPositionSizer(
            risk_aversion=risk_aversion,
            sizing_horizon=sizing_horizon,
            base_size_pct=base_size_p,
            min_size_pct=min_size_p,
            max_size_pct=max_size_p,
        )
        progress.progress(45, text="Running backtest …")

        bt = QuantBacktester(
            ssb=ssb, mle=mle, sizer=sizer,
            stop_loss_pct=stop_loss_p, take_profit_pct=take_profit_p,
            holding_period=holding_p, mle_window=mle_window,
            entry_prob_thresh=entry_thresh, max_entropy=max_entropy,
            max_risk_prob=max_risk_p, ck_horizon=ck_horizon,
        )
        result = bt.run(df, initial_capital=init_cap)
        st.session_state["qresult"] = result
        st.session_state["mle_obj"] = mle
        st.session_state["ck_obj"]  = ChapmanKolmogorov(mle)

        progress.progress(100, text="✅ Complete")
        st.rerun()


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
result: QuantBacktestResult = st.session_state.get("qresult")
df: pd.DataFrame             = st.session_state.get("df")
mle_obj: MLETransitionMatrix = st.session_state.get("mle_obj")
ck_obj: ChapmanKolmogorov    = st.session_state.get("ck_obj")

if result is None:
    # ── Welcome / Critique screen ──────────────────────────────────────────
    st.markdown("## 📋 Model Critique & Mathematical Architecture")

    st.markdown("""
    <div class="critique-box">
    <h4 style="color:#a78bfa;font-family:'JetBrains Mono'">A) How the naive model fails — 5 critical gaps</h4>

    <p><b style="color:#ff4466">Gap 1 — Independence assumption:</b>
    The original model fires a LONG/SHORT based on single-bar probability mass in the upper/lower state half.
    This implicitly assumes P(X_{t+1} | X_t) = P(X_{t+1}) — i.e., the future state is independent of the current one.
    This is mathematically false for financial returns, which exhibit autocorrelation, volatility clustering (GARCH-type), and regime persistence.</p>

    <p><b style="color:#ff4466">Gap 2 — Flat position sizing:</b>
    <code>size = equity × position_size_pct</code> ignores the probability of transitioning into a high-volatility/bearish state
    in the next few bars. A trade entered when P(high-risk at t+5) = 0.80 should be half the size of one where P = 0.20.</p>

    <p><b style="color:#ff4466">Gap 3 — 1-dimensional state space:</b>
    Quantising only log-returns into N states collapses a 2D market reality (volatility × trend) into 1D.
    A HIGH-VOL BULLISH state has completely different Markov dynamics from a LOW-VOL BULLISH state, but the original
    model treats them identically because both produce a positive return observation.</p>

    <p><b style="color:#ff4466">Gap 4 — Static transition matrix:</b>
    The matrix is re-estimated on each lookback window but has no explicit time-homogeneity check.
    During macro regime shifts (e.g., 2020 vol spike, 2022 rate cycle), the historical P drifts badly.
    A rolling MLE window directly addresses this by discarding observations that predate the current regime.</p>

    <p><b style="color:#ff4466">Gap 5 — No n-step forecasting:</b>
    The model only forecasts 1-bar ahead. For position-holding strategies with multi-bar exits, the relevant question is
    "where will the market likely be in 3–5 bars?" — answered exactly by P^n via the Chapman-Kolmogorov equation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="critique-box">
    <h4 style="color:#a78bfa;font-family:'JetBrains Mono'">B) Mathematical architecture of the upgrade</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="math-box">
        <b style="color:#00e5ff">MLE Transition Matrix:</b><br><br>
          P̂[i,j] = (n_w(i→j) + α) / (Σ_k n_w(i→k) + α·N)<br><br>
        where:<br>
          n_w = transition count in rolling window w<br>
          α   = Laplace smoothing pseudo-count<br>
          N   = number of states (9)<br><br>
        <b style="color:#00e5ff">n-Step Forecasting (Chapman-Kolmogorov):</b><br><br>
          P^n = P · P · ... · P  (n times)<br>
          P^n[i,j] = P(X_{t+n}=j | X_t=i)<br><br>
          Implemented via repeated squaring: O(k³ log n)
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="math-box">
        <b style="color:#00e5ff">State Distribution Vector:</b><br><br>
          π(t) = π(0) · P^t<br><br>
        where π(0) = e_i (one-hot at current state i)<br><br>
        <b style="color:#00e5ff">Dynamic Position Sizing:</b><br><br>
          P_risk(t) = Σ_{j∈H} π(t)[j]<br>
          f(t)      = f_base × (1 − w · P_risk(t))<br>
          size      = clamp(f(t), f_min, f_max) · equity<br><br>
        where H = {high-volatility or bearish states}<br>
              w = risk-aversion ∈ [0,1]
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Configure parameters in the sidebar and click **▶ RUN QUANT BACKTEST**")
    st.stop()


# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance", "📈 Price & States", "📋 Trade Log",
    "🔬 Transition Matrix", "🔭 C-K Forecast", "📉 Regime Analysis"
])


# ══════════════════════════════════════════════
#  TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════
with tab1:
    s = result.summary()
    st.markdown("#### 📐 Performance Summary")

    r1c = [st.columns(6)[i] for i in range(6)]
    ret_c  = "g" if result.total_return >= 0 else "r"
    sh_c   = "g" if result.sharpe >= 1    else ("r" if result.sharpe < 0 else "")
    with st.columns(6)[0]: pass  # placeholder — use below

    cols = st.columns(6)
    with cols[0]: mc("Total Return",   f"{result.total_return*100:+.2f}%", ret_c)
    with cols[1]: mc("Sharpe Ratio",   f"{result.sharpe:.3f}", sh_c)
    with cols[2]: mc("Max Drawdown",   f"{result.max_drawdown*100:.2f}%", "r")
    with cols[3]: mc("Win Rate",       f"{result.win_rate*100:.1f}%", "g" if result.win_rate>=.5 else "r")
    with cols[4]: mc("Profit Factor",  f"{result.profit_factor:.3f}", "g" if result.profit_factor>=1.5 else "")
    with cols[5]: mc("N Trades",       str(result.n_trades), "a")

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    with cols2[0]: mc("Avg Scale Factor",  f"{result.avg_scale:.3f}", "v")
    with cols2[1]: mc("Avg Risk Prob",     f"{result.avg_risk_prob:.3f}", "a")
    with cols2[2]: mc("Avg Chain Entropy", f"{result.avg_entropy:.3f} bits", "v")
    with cols2[3]: mc("Sizing Benefit",    f"{(1-result.avg_scale)*100:.1f}% reduced", "a")

    # ── Equity + Drawdown ─────────────────────────────────────────────────
    st.markdown("#### 💹 Equity Curve & Drawdown")
    eq  = result.equity_curve
    idx = df.index[:len(eq)] if df is not None else np.arange(len(eq))
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak!=0, peak, 1) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=idx, y=eq, name="Equity",
        line=dict(color="#00e5ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=dd, name="Drawdown %",
        line=dict(color="#ff4466", width=1, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,68,102,0.08)"), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
        height=420, margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(title="Equity ($)", gridcolor="#172040"),
        yaxis2=dict(title="DD %", gridcolor="#172040"),
        xaxis2=dict(gridcolor="#172040"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scale factor over time
    if result.trades:
        st.markdown("#### ⚖️ Dynamic Position Scale Factor (per trade)")
        sf_x  = [t.entry_bar for t in result.trades]
        sf_y  = [t.scale_factor for t in result.trades]
        rp_y  = [t.risk_prob for t in result.trades]
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Scale Factor per Trade", "Risk Probability at Entry"])
        fig2.add_trace(go.Scatter(x=list(range(len(sf_y))), y=sf_y, mode="lines+markers",
            line=dict(color="#7c3aed", width=1.5),
            marker=dict(size=4, color=sf_y, colorscale=[[0,"#ff4466"],[1,"#00ff88"]])), row=1, col=1)
        fig2.add_trace(go.Scatter(x=list(range(len(rp_y))), y=rp_y, mode="lines+markers",
            line=dict(color="#ffaa00", width=1.5),
            marker=dict(size=4, color=rp_y, colorscale=[[0,"#00ff88"],[1,"#ff4466"]])), row=1, col=2)
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=280, margin=dict(l=10,r=10,t=30,b=10), showlegend=False,
            yaxis=dict(gridcolor="#172040"), yaxis2=dict(gridcolor="#172040"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — PRICE & STATES
# ══════════════════════════════════════════════
with tab2:
    if df is None:
        st.warning("No price data.")
    else:
        st.markdown("#### 📈 Candlestick + Regime States + Signals")

        state_series = result.state_series
        n_s = min(len(df), len(state_series))
        vol_colors = []
        trend_colors = []
        for i in range(n_s):
            s = state_series[i]
            if isinstance(s, CompositeState):
                vc = {"LOW": "rgba(0,255,136,0.25)", "MEDIUM": "rgba(0,229,255,0.15)", "HIGH": "rgba(255,100,0,0.25)"}
                tc = {"BULLISH": "rgba(0,255,136,0.4)", "SIDEWAYS": "rgba(255,215,0,0.3)", "BEARISH": "rgba(255,68,102,0.4)"}
                vol_colors.append(vc.get(state_series[i].vol.name, "rgba(100,100,100,0.1)"))
                trend_colors.append(tc.get(state_series[i].trend.name, "rgba(100,100,100,0.1)"))

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#00ff88", decreasing_line_color="#ff4466", name=pair,
        ), row=1, col=1)

        # Trade markers
        lx, ly, sx, sy, ex, ey = [], [], [], [], [], []
        for t in result.trades:
            b = min(t.entry_bar, len(df)-1)
            lx.append(df.index[b]) if t.direction=="LONG"  else sx.append(df.index[b])
            ly.append(t.entry_price) if t.direction=="LONG" else sy.append(t.entry_price)
            eb = min(t.exit_bar, len(df)-1)
            ex.append(df.index[eb]); ey.append(t.exit_price)

        if lx: fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#00ff88"), name="Long"), row=1, col=1)
        if sx: fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#ff4466"), name="Short"), row=1, col=1)
        if ex: fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers",
            marker=dict(symbol="x", size=7, color="#ffd700"), name="Exit"), row=1, col=1)

        # Regime bars
        if trend_colors:
            fig.add_trace(go.Bar(x=df.index[:n_s], y=[1]*n_s,
                marker_color=trend_colors, name="Trend Regime", showlegend=True), row=2, col=1)
        if vol_colors:
            fig.add_trace(go.Bar(x=df.index[:n_s], y=[1]*n_s,
                marker_color=vol_colors, name="Vol Regime", showlegend=True), row=3, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=600, margin=dict(l=10,r=10,t=10,b=10),
            xaxis_rangeslider_visible=False,
            yaxis=dict(gridcolor="#172040"), yaxis2=dict(showticklabels=False),
            yaxis3=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # State legend
        st.markdown("""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:11px;font-family:'JetBrains Mono'">
          <span class="state-chip state-bull">■ BULLISH trend</span>
          <span class="state-chip state-side">■ SIDEWAYS trend</span>
          <span class="state-chip state-bear">■ BEARISH trend</span>
          <span style="margin-left:16px;"></span>
          <span class="state-chip state-lvol">■ LOW vol</span>
          <span class="state-chip state-mvol">■ MED vol</span>
          <span class="state-chip state-hvol">■ HIGH vol</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — TRADE LOG
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### 📋 Trade-by-Trade Log")
    if not result.trades:
        st.info("No trades taken.")
    else:
        rows = []
        for i, t in enumerate(result.trades):
            trend_n = t.entry_state.trend.name if isinstance(t.entry_state, CompositeState) else "?"
            vol_n   = t.entry_state.vol.name   if isinstance(t.entry_state, CompositeState) else "?"
            rows.append({
                "#": i+1,
                "Dir": t.direction,
                "Entry": round(t.entry_price, 5),
                "Exit":  round(t.exit_price,  5),
                "Bars":  t.bars_held,
                "Size($)":  round(t.size, 2),
                "Scale":    round(t.scale_factor, 3),
                "RiskProb": round(t.risk_prob, 3),
                "P&L($)":   round(t.pnl, 2),
                "Ret%":     round(t.pnl/t.size*100 if t.size else 0, 2),
                "Prob":     round(t.entry_prob, 3),
                "Entropy":  round(t.entry_entropy, 3),
                "Trend":    trend_n,
                "Vol":      vol_n,
                "Exit":     t.exit_reason,
            })
        tdf = pd.DataFrame(rows)

        pnls = [t.pnl for t in result.trades]
        cols = st.columns(4)
        with cols[0]: mc("Gross Profit",  f"${sum(x for x in pnls if x>0):,.2f}", "g")
        with cols[1]: mc("Gross Loss",    f"${abs(sum(x for x in pnls if x<=0)):,.2f}", "r")
        with cols[2]: mc("Avg Win",       f"${np.mean([x for x in pnls if x>0]):.2f}" if any(x>0 for x in pnls) else "$0", "g")
        with cols[3]: mc("Avg Loss",      f"${abs(np.mean([x for x in pnls if x<=0])):.2f}" if any(x<=0 for x in pnls) else "$0", "r")

        st.markdown("<br>", unsafe_allow_html=True)

        def style_fn(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in df.iterrows():
                c = "color: #00ff88" if row["P&L($)"] >= 0 else "color: #ff4466"
                styles.at[i,"P&L($)"] = c
                styles.at[i,"Ret%"]   = c
                styles.at[i,"Dir"]    = "color:#00ff88" if row["Dir"]=="LONG" else "color:#ff4466"
                sc = row["Scale"]
                styles.at[i,"Scale"]  = f"color:{'#00ff88' if sc>0.8 else '#ffaa00' if sc>0.5 else '#ff4466'}"
            return styles

        st.dataframe(tdf.style.apply(style_fn, axis=None),
                     use_container_width=True, height=420, hide_index=True)

        # PnL distribution
        fig_d = make_subplots(rows=1, cols=2,
                              subplot_titles=["P&L Distribution", "Scale Factor vs P&L"])
        fig_d.add_trace(go.Histogram(x=pnls, nbinsx=25,
            marker_color=["#00ff88" if x>=0 else "#ff4466" for x in pnls]), row=1, col=1)
        sf = [t.scale_factor for t in result.trades]
        fig_d.add_trace(go.Scatter(x=sf, y=pnls, mode="markers",
            marker=dict(color=pnls, colorscale=[[0,"#ff4466"],[0.5,"#ffd700"],[1,"#00ff88"]],
                        size=6, showscale=False)), row=1, col=2)
        fig_d.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=300, margin=dict(l=10,r=10,t=30,b=10), showlegend=False,
            yaxis=dict(gridcolor="#172040"), yaxis2=dict(gridcolor="#172040"),
            xaxis2=dict(title="Scale Factor"),
        )
        st.plotly_chart(fig_d, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — TRANSITION MATRIX
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### 🔬 MLE Transition Matrix  P̂[i→j]")
    if mle_obj is None or mle_obj._P is None:
        st.warning("No fitted matrix — run the backtest first.")
    else:
        P = mle_obj.P
        state_labels = [str(s) for s in ALL_STATES]

        fig_hm = go.Figure(go.Heatmap(
            z=P, x=state_labels, y=state_labels,
            colorscale=[[0,"#080c18"],[0.3,"#0d2040"],[0.6,"#0e4080"],[1,"#00e5ff"]],
            text=np.round(P, 3), texttemplate="%{text}",
            textfont={"size": 10, "family": "JetBrains Mono"},
            zmin=0, zmax=1,
        ))
        fig_hm.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=480,
            title=dict(text="Transition Matrix — row i = current state, col j = next state",
                       font=dict(color="#00e5ff", family="JetBrains Mono", size=13)),
            xaxis=dict(title="Next State j", side="bottom"),
            yaxis=dict(title="Current State i", autorange="reversed"),
            margin=dict(l=10,r=10,t=50,b=10),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Stationary distribution + entropy + mixing time
        try:
            pi  = mle_obj.stationary_distribution()
            ent = mle_obj.entropy_rate()
            mt  = mle_obj.mixing_time()
            per = mle_obj.regime_persistence() if hasattr(mle_obj, 'regime_persistence') else np.diag(P)

            col1, col2, col3 = st.columns(3)
            with col1: mc("Chain Entropy", f"{ent:.3f} bits", "v")
            with col2: mc("Mixing Time τ", f"{mt} bars", "a")
            with col3: mc("Max Persistence", f"{np.diag(P).max():.3f}", "g")

            st.markdown("#### 🌀 Stationary Distribution π")
            fig_pi = go.Figure(go.Bar(
                x=state_labels, y=pi,
                marker_color=["#ff4466" if ALL_STATES[i].is_high_risk else "#00ff88"
                               for i in range(N_COMPOSITE_STATES)],
                text=[f"{v:.3f}" for v in pi], textposition="outside",
            ))
            fig_pi.update_layout(
                template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
                height=280, margin=dict(l=10,r=10,t=10,b=10),
                yaxis=dict(gridcolor="#172040"),
                xaxis=dict(title="State (🔴 = high-risk, 🟢 = low-risk)"),
            )
            st.plotly_chart(fig_pi, use_container_width=True)

            # Count matrix
            st.markdown("#### 📊 Transition Count Matrix  n(i→j)")
            C = mle_obj.count_matrix
            fig_cm = go.Figure(go.Heatmap(
                z=C, x=state_labels, y=state_labels,
                colorscale=[[0,"#0d1425"],[1,"#7c3aed"]],
                text=C.astype(int), texttemplate="%{text}",
                textfont={"size": 10},
            ))
            fig_cm.update_layout(
                template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
                height=400, margin=dict(l=10,r=10,t=10,b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.warning(f"Diagnostics failed: {e}")


# ══════════════════════════════════════════════
#  TAB 5 — CHAPMAN-KOLMOGOROV FORECAST
# ══════════════════════════════════════════════
with tab5:
    st.markdown("#### 🔭 Chapman-Kolmogorov n-Step Forecast  P^n[i→j]")

    if ck_obj is None:
        st.warning("Run the backtest first.")
    else:
        start_state_label = st.selectbox(
            "Current State (starting point for forecast)",
            [str(s) for s in ALL_STATES],
            index=5,
        )
        start_idx = next(i for i, s in enumerate(ALL_STATES) if str(s) == start_state_label)
        max_h     = st.slider("Max Forecast Horizon", 2, 30, 15)

        # Horizon table
        try:
            tbl = ck_obj.forecast_horizon_table(start_idx, max_h)
            st.markdown(f"**P(state=j at horizon n | start at {start_state_label})**")

            # Aggregate by trend & vol for readability
            bull_cols = [str(s) for s in ALL_STATES if s.trend == TrendRegime.BULLISH]
            bear_cols = [str(s) for s in ALL_STATES if s.trend == TrendRegime.BEARISH]
            hrisk_cols = [str(s) for s in ALL_STATES if s.is_high_risk]

            agg = pd.DataFrame({
                "P(Bullish)": tbl[bull_cols].sum(axis=1),
                "P(Bearish)": tbl[bear_cols].sum(axis=1),
                "P(High-Risk)": tbl[hrisk_cols].sum(axis=1),
                "P(Sideways)":  1 - tbl[bull_cols].sum(axis=1) - tbl[bear_cols].sum(axis=1),
            })
            agg.index.name = "Horizon (bars)"

            fig_ck = go.Figure()
            colors = {"P(Bullish)": "#00ff88", "P(Bearish)": "#ff4466",
                      "P(High-Risk)": "#ff6400", "P(Sideways)": "#ffd700"}
            for col in agg.columns:
                fig_ck.add_trace(go.Scatter(
                    x=agg.index, y=agg[col], mode="lines+markers",
                    name=col, line=dict(color=colors[col], width=2),
                    marker=dict(size=5),
                ))
            fig_ck.add_hline(y=0.5, line_dash="dot", line_color="#3a5070", line_width=1)
            fig_ck.update_layout(
                template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
                height=340, margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(title="Forecast Horizon n (bars)", gridcolor="#172040"),
                yaxis=dict(title="Probability", gridcolor="#172040", range=[0,1]),
                legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_ck, use_container_width=True)

            st.markdown("**Full state-by-state n-step table:**")
            st.dataframe(tbl, use_container_width=True, height=320)

            # Mixing time viz
            try:
                mt = mle_obj.mixing_time()
                st.info(f"📡 Estimated mixing time **τ = {mt} bars** — after this many steps the distribution is approximately stationary regardless of starting state.")
            except:
                pass

        except Exception as e:
            st.error(f"C-K computation failed: {e}")


# ══════════════════════════════════════════════
#  TAB 6 — REGIME ANALYSIS
# ══════════════════════════════════════════════
with tab6:
    st.markdown("#### 📉 Regime Distribution & Persistence")

    if not result.state_series:
        st.warning("No state data available.")
    else:
        states = [s for s in result.state_series if isinstance(s, CompositeState)]

        # State frequency
        freq = {}
        for s in states:
            k = str(s)
            freq[k] = freq.get(k, 0) + 1
        freq_df = pd.DataFrame({"State": list(freq.keys()), "Count": list(freq.values())})
        freq_df["Pct"] = freq_df["Count"] / freq_df["Count"].sum() * 100

        fig_freq = go.Figure(go.Bar(
            x=freq_df["State"], y=freq_df["Count"],
            marker_color=["#ff4466" if ALL_STATES[i].is_high_risk else "#00ff88"
                          for i, s in enumerate(ALL_STATES) if str(s) in freq_df["State"].values],
            text=freq_df["Pct"].round(1).astype(str)+"%", textposition="outside",
        ))
        fig_freq.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=300, margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(gridcolor="#172040"),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        # Regime persistence over time (rolling %)
        st.markdown("#### 🕐 Regime Persistence over Time")
        win = st.slider("Rolling window (bars)", 10, 100, 30)
        bull_roll, bear_roll, hrisk_roll = [], [], []
        for k in range(win, len(states)+1):
            chunk = states[k-win:k]
            bull_roll.append(sum(1 for s in chunk if s.trend==TrendRegime.BULLISH) / win)
            bear_roll.append(sum(1 for s in chunk if s.trend==TrendRegime.BEARISH) / win)
            hrisk_roll.append(sum(1 for s in chunk if s.is_high_risk) / win)

        fig_roll = go.Figure()
        x_r = list(range(win, len(states)+1))
        fig_roll.add_trace(go.Scatter(x=x_r, y=bull_roll, name="P(Bullish)",
            line=dict(color="#00ff88", width=1.5), fill="tozeroy", fillcolor="rgba(0,255,136,0.05)"))
        fig_roll.add_trace(go.Scatter(x=x_r, y=bear_roll, name="P(Bearish)",
            line=dict(color="#ff4466", width=1.5), fill="tozeroy", fillcolor="rgba(255,68,102,0.05)"))
        fig_roll.add_trace(go.Scatter(x=x_r, y=hrisk_roll, name="P(High-Risk)",
            line=dict(color="#ff6400", width=1.5, dash="dash")))
        fig_roll.update_layout(
            template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
            height=300, margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(title="Rolling Probability", gridcolor="#172040", range=[0,1]),
            xaxis=dict(gridcolor="#172040"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

        # Self-transition persistence (diagonal of P)
        if mle_obj and mle_obj._P is not None:
            diag = np.diag(mle_obj.P)
            fig_per = go.Figure(go.Bar(
                x=[str(s) for s in ALL_STATES], y=diag,
                marker_color=["#7c3aed" if d > 0.5 else "#3a5070" for d in diag],
                text=[f"{d:.3f}" for d in diag], textposition="outside",
            ))
            fig_per.add_hline(y=1/N_COMPOSITE_STATES, line_dash="dot",
                              line_color="#3a5070", annotation_text="random baseline")
            fig_per.update_layout(
                template="plotly_dark", paper_bgcolor="#080c18", plot_bgcolor="#080c18",
                height=300, margin=dict(l=10,r=10,t=10,b=10),
                title=dict(text="State Self-Persistence P[i,i]  (diagonal of transition matrix)",
                           font=dict(color="#00e5ff", size=12)),
                yaxis=dict(gridcolor="#172040", range=[0,1]),
            )
            st.plotly_chart(fig_per, use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px 0;color:#172040;font-family:'JetBrains Mono';font-size:10px;">
  ⚠️ Research and educational purposes only · Not financial advice · Past performance ≠ future results
</div>
""", unsafe_allow_html=True)
