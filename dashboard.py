"""
╔══════════════════════════════════════════════════════════════════════╗
║   MARKOV FOREX TRADER  ·  Streamlit Dashboard                       ║
║   Run: streamlit run dashboard.py                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import time
import threading

from data_feed import fetch_forex_data, FOREX_PAIRS, TIMEFRAMES
from markov_engine import (
    MarkovParams, MarkovChain, SignalGenerator,
    Backtester, BacktestResult, MarkovOptimizer
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Markov Forex Trader",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark terminal-quant aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --panel: #0f1628;
    --border: #1e2d4a;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #00ff88;
    --red: #ff4466;
    --gold: #ffd700;
    --text: #c8d8f0;
    --muted: #4a6080;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4 {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent) !important;
    letter-spacing: 0.05em;
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 14px 18px;
    margin: 4px 0;
}
.metric-card .label {
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 22px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
}
.metric-card.green .value { color: var(--green); border-color: var(--green); }
.metric-card.red    .value { color: var(--red);   border-color: var(--red);   }
.metric-card.gold   .value { color: var(--gold);  border-color: var(--gold);  }

.trade-row-long  { background: rgba(0,255,136,0.04); }
.trade-row-short { background: rgba(255,68,102,0.04); }

.stButton button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 10px 24px !important;
}
.stButton button:hover { opacity: 0.85 !important; }

[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 12px !important;
}

.header-bar {
    background: linear-gradient(90deg, #0a0e1a 0%, #0f1628 50%, #0a0e1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 16px 0;
    margin-bottom: 24px;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.tag-long  { background: rgba(0,255,136,0.15); color: var(--green); }
.tag-short { background: rgba(255,68,102,0.15); color: var(--red); }
.tag-tp    { background: rgba(0,212,255,0.15);  color: var(--accent); }
.tag-sl    { background: rgba(255,68,102,0.15); color: var(--red); }
.tag-hp    { background: rgba(124,58,237,0.15); color: #a78bfa; }
.tag-eod   { background: rgba(255,215,0,0.15);  color: var(--gold); }

div[data-testid="stHorizontalBlock"] { gap: 12px; }

.stProgress > div > div { background: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "result": None,
        "df": None,
        "optimizer": None,
        "opt_running": False,
        "opt_progress": 0,
        "opt_best_score": 0,
        "opt_best_params": None,
        "opt_log": [],
        "opt_done": False,
        "all_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def fmt_pct(v): return f"{v*100:+.2f}%"
def fmt_price(v, pair): return f"{v:.5f}" if "JPY" not in pair else f"{v:.3f}"
def color_val(v, green_thresh=0): return "green" if v >= green_thresh else "red"

def metric_card(label, value, cls=""):
    st.markdown(f"""
    <div class="metric-card {cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>""", unsafe_allow_html=True)

def tag_html(text, cls):
    return f'<span class="tag {cls}">{text}</span>'


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
<h1 style="text-align:center; font-size:28px; margin:0; padding:0;">
  🎲 &nbsp; MARKOV CHAIN FOREX TRADER
</h1>
<p style="text-align:center; color:#4a6080; font-family:'JetBrains Mono'; font-size:12px; margin:6px 0 0;">
  Stochastic State Machine · Iterative Parameter Optimizer · Live Backtester
</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ CONFIGURATION")
    st.markdown("---")

    pair = st.selectbox("Forex Pair", list(FOREX_PAIRS.keys()), index=0)
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=0)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=730))
    with col2:
        end_date = st.date_input("End", value=date.today())

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=100, max_value=10_000_000,
        value=10_000, step=500
    )

    st.markdown("---")
    st.markdown("### 🧬 MARKOV PARAMS")

    use_optimizer = st.checkbox("🔄 Auto-Optimize", value=True,
                                help="Iteratively find best parameters")

    if not use_optimizer:
        n_states = st.slider("States", 3, 11, 5, step=2)
        lookback = st.slider("Lookback Bars", 5, 100, 20)
        holding_period = st.slider("Holding Period", 1, 20, 3)
        entry_threshold = st.slider("Entry Threshold", 0.50, 0.90, 0.60, step=0.01)
        stop_loss_pct = st.slider("Stop Loss %", 0.1, 2.0, 0.5, step=0.05) / 100
        take_profit_pct = st.slider("Take Profit %", 0.2, 4.0, 1.0, step=0.1) / 100
        position_size = st.slider("Position Size %", 0.5, 10.0, 2.0, step=0.5) / 100
        state_method = st.selectbox("State Method", ["quantile", "equal"])
    else:
        opt_iters = st.slider("Optimization Depth", 1, 3, 2,
                              help="1=Coarse, 2=Fine, 3=Micro-tune")

    st.markdown("---")
    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)


# ─────────────────────────────────────────────
#  RUN BACKTEST
# ─────────────────────────────────────────────
if run_btn:
    st.session_state["opt_log"] = []
    st.session_state["opt_progress"] = 0
    st.session_state["opt_done"] = False
    st.session_state["result"] = None
    st.session_state["all_results"] = []

    with st.spinner(f"Fetching {pair} data..."):
        df = fetch_forex_data(
            pair,
            start=str(start_date),
            end=str(end_date),
            interval=TIMEFRAMES[timeframe]["interval"],
        )
        st.session_state["df"] = df

    if df is None or len(df) < 50:
        st.error("⚠️ Not enough data. Try a wider date range.")
    else:
        prices = df["close"].values

        if not use_optimizer:
            params = MarkovParams(
                n_states=n_states,
                lookback=lookback,
                holding_period=holding_period,
                entry_threshold=entry_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_size_pct=position_size,
                state_method=state_method,
            )
            with st.spinner("Running backtest..."):
                bt = Backtester(params)
                result = bt.run(prices, initial_capital)
            st.session_state["result"] = result
            st.session_state["opt_done"] = True
        else:
            # ── Optimizer with live progress ──────────────────
            progress_bar = st.progress(0, text="Initializing optimizer…")
            status_text  = st.empty()
            log_expander = st.expander("📋 Optimization Log", expanded=False)
            log_area     = log_expander.empty()

            def progress_cb(pct, best_score, best_params, iteration,
                            phase_name, combo, total):
                st.session_state["opt_progress"] = pct
                st.session_state["opt_best_score"] = best_score
                st.session_state["opt_best_params"] = best_params
                log_entry = (
                    f"[{phase_name}] {combo}/{total} | "
                    f"Best Score: {best_score:.4f} | "
                    f"States={best_params.n_states} "
                    f"LB={best_params.lookback} "
                    f"HP={best_params.holding_period} "
                    f"Thresh={best_params.entry_threshold:.2f}"
                )
                st.session_state["opt_log"].append(log_entry)
                progress_bar.progress(
                    min(int(pct), 99),
                    text=f"[{phase_name}] Iteration {iteration} · {combo}/{total} combinations · Best Score: {best_score:.4f}"
                )
                if len(st.session_state["opt_log"]) % 10 == 0:
                    log_area.text("\n".join(st.session_state["opt_log"][-50:]))

            optimizer = MarkovOptimizer(
                prices=prices,
                initial_capital=initial_capital,
                max_iterations=opt_iters,
                progress_callback=progress_cb,
            )

            best = optimizer.optimize()
            st.session_state["result"] = best
            st.session_state["all_results"] = optimizer.get_top_n(10)
            st.session_state["opt_done"] = True

            progress_bar.progress(100, text="✅ Optimization complete!")
            log_area.text("\n".join(st.session_state["opt_log"][-50:]))
            status_text.success(f"🏆 Best score: {best.score:.4f} | "
                                f"Sharpe: {best.sharpe:.3f} | "
                                f"Return: {best.total_return*100:.2f}%")

    st.rerun()


# ─────────────────────────────────────────────
#  RESULTS DISPLAY
# ─────────────────────────────────────────────
result: BacktestResult = st.session_state.get("result")
df: pd.DataFrame = st.session_state.get("df")

if result is None:
    st.info("👈 Configure your strategy in the sidebar and click **▶ RUN BACKTEST**")

    # Illustrative diagram
    st.markdown("""
    <br>
    <div style="text-align:center; padding:40px; border:1px dashed #1e2d4a; border-radius:8px; color:#4a6080;">
        <h3 style="color:#00d4ff; font-family:'JetBrains Mono';">How It Works</h3>
        <pre style="text-align:left; display:inline-block; font-size:12px; color:#4a6080;">
  Price Returns ──► Discretize into N States
                         │
                         ▼
              Build Transition Matrix T[i,j]
              P(next_state=j | current_state=i)
                         │
                         ▼
         Predict: P(bullish) > threshold → LONG
                  P(bearish) > threshold → SHORT
                         │
                         ▼
           Iterate params → maximize Sharpe/Score
        </pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Price & Signals", "📋 Trade Log",
    "🔬 Markov Matrix", "🏆 Optimizer Results"
])


# ══════════════════════════════════════════════
#  TAB 1  —  OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    p = result.params
    s = result.summary()

    # ── Key Metrics ───────────────────────────
    st.markdown("#### 📐 Performance Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ret_cls = "green" if result.total_return >= 0 else "red"
    sh_cls  = "green" if result.sharpe >= 1 else ("" if result.sharpe >= 0 else "red")

    with c1: metric_card("Total Return", fmt_pct(result.total_return), ret_cls)
    with c2: metric_card("Sharpe Ratio", f"{result.sharpe:.3f}", sh_cls)
    with c3: metric_card("Max Drawdown", fmt_pct(result.max_drawdown), "red")
    with c4: metric_card("Win Rate", f"{result.win_rate*100:.1f}%",
                         "green" if result.win_rate >= 0.5 else "red")
    with c5: metric_card("Profit Factor", f"{result.profit_factor:.2f}",
                         "green" if result.profit_factor >= 1.5 else "")
    with c6: metric_card("N Trades", str(result.n_trades), "gold")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity Curve ──────────────────────────
    st.markdown("#### 💹 Equity Curve")
    eq = result.equity_curve
    dates_eq = df.index[:len(eq)] if df is not None else np.arange(len(eq))

    fig_eq = go.Figure()
    # Gradient fill under equity
    fig_eq.add_trace(go.Scatter(
        x=dates_eq, y=eq,
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(color="#00d4ff", width=2),
        name="Equity",
    ))
    # Drawdown overlay
    peak = np.maximum.accumulate(eq)
    dd_curve = (eq - peak) / peak * 100
    fig_eq.add_trace(go.Scatter(
        x=dates_eq, y=dd_curve,
        fill="tozeroy",
        fillcolor="rgba(255,68,102,0.10)",
        line=dict(color="#ff4466", width=1, dash="dot"),
        name="Drawdown %",
        yaxis="y2",
    ))
    fig_eq.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Equity ($)", gridcolor="#1e2d4a", titlefont_color="#00d4ff"),
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right",
                    gridcolor="#1e2d4a", titlefont_color="#ff4466"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e2d4a"),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Best Parameters ───────────────────────
    st.markdown("#### 🧬 Best Parameters Found")
    param_data = {
        "Parameter": ["States", "Lookback", "Holding Period", "Entry Threshold",
                       "Stop Loss", "Take Profit", "Position Size", "State Method"],
        "Value": [
            p.n_states, p.lookback, f"{p.holding_period} bars",
            f"{p.entry_threshold:.2f}",
            f"{p.stop_loss_pct*100:.2f}%", f"{p.take_profit_pct*100:.2f}%",
            f"{p.position_size_pct*100:.1f}%", p.state_method
        ]
    }
    st.dataframe(
        pd.DataFrame(param_data),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════
#  TAB 2  —  PRICE & SIGNALS
# ══════════════════════════════════════════════
with tab2:
    if df is None:
        st.warning("No price data available.")
    else:
        st.markdown("#### 📈 Price Chart with Trade Signals")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4466",
            name=pair,
        ), row=1, col=1)

        # Trade overlays
        long_x, long_y, short_x, short_y = [], [], [], []
        exit_x, exit_y = [], []

        for t in result.trades:
            if t.entry_bar < len(df):
                idx = df.index[min(t.entry_bar, len(df) - 1)]
                ep  = t.entry_price
                if t.direction == "LONG":
                    long_x.append(idx); long_y.append(ep)
                else:
                    short_x.append(idx); short_y.append(ep)
            if t.exit_bar < len(df):
                idx = df.index[min(t.exit_bar, len(df) - 1)]
                exit_x.append(idx); exit_y.append(t.exit_price)

        if long_x:
            fig.add_trace(go.Scatter(
                x=long_x, y=long_y, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00ff88"),
                name="Long Entry",
            ), row=1, col=1)

        if short_x:
            fig.add_trace(go.Scatter(
                x=short_x, y=short_y, mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ff4466"),
                name="Short Entry",
            ), row=1, col=1)

        if exit_x:
            fig.add_trace(go.Scatter(
                x=exit_x, y=exit_y, mode="markers",
                marker=dict(symbol="x", size=7, color="#ffd700"),
                name="Exit",
            ), row=1, col=1)

        # Volume
        colors = ["#00ff88" if c >= o else "#ff4466"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, opacity=0.5, name="Volume",
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            height=580,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
            xaxis2=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            yaxis2=dict(gridcolor="#1e2d4a"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3  —  TRADE LOG
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### 📋 Trade-by-Trade Log")

    trades = result.trades
    if not trades:
        st.info("No trades executed.")
    else:
        rows = []
        for i, t in enumerate(trades):
            rows.append({
                "#": i + 1,
                "Direction": t.direction,
                "Entry Bar": t.entry_bar,
                "Entry Price": round(t.entry_price, 6),
                "Exit Price": round(t.exit_price, 6),
                "Exit Bar": t.exit_bar,
                "Bars Held": t.exit_bar - t.entry_bar,
                "Size ($)": round(t.size, 2),
                "P&L ($)": round(t.pnl, 2),
                "Return %": round(t.pnl / t.size * 100 if t.size else 0, 2),
                "Exit Reason": t.exit_reason,
                "Signal Prob": round(t.predicted_prob, 3),
            })

        trade_df = pd.DataFrame(rows)

        # Summary stats
        pnls = [t.pnl for t in trades]
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Gross Profit",  f"${sum(x for x in pnls if x>0):,.2f}", "green")
        with c2: metric_card("Gross Loss",    f"${abs(sum(x for x in pnls if x<=0)):,.2f}", "red")
        with c3: metric_card("Avg Win",
                             f"${np.mean([x for x in pnls if x>0]):.2f}" if any(x>0 for x in pnls) else "$0",
                             "green")
        with c4: metric_card("Avg Loss",
                             f"${abs(np.mean([x for x in pnls if x<=0])):.2f}" if any(x<=0 for x in pnls) else "$0",
                             "red")

        st.markdown("<br>", unsafe_allow_html=True)

        # Color-coded table
        def style_trades(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in df.iterrows():
                pnl_col = "color: #00ff88" if row["P&L ($)"] >= 0 else "color: #ff4466"
                styles.at[i, "P&L ($)"] = pnl_col
                styles.at[i, "Return %"] = pnl_col
                dir_col = "color: #00ff88" if row["Direction"] == "LONG" else "color: #ff4466"
                styles.at[i, "Direction"] = dir_col
            return styles

        st.dataframe(
            trade_df.style.apply(style_trades, axis=None),
            use_container_width=True,
            height=420,
            hide_index=True,
        )

        # PnL Distribution
        st.markdown("#### 🎯 P&L Distribution")
        fig_dist = make_subplots(rows=1, cols=2,
                                 subplot_titles=["P&L per Trade ($)", "Cumulative P&L ($)"])

        fig_dist.add_trace(go.Histogram(
            x=pnls, nbinsx=30,
            marker_color=["#00ff88" if x >= 0 else "#ff4466" for x in pnls],
            name="P&L",
        ), row=1, col=1)

        cum_pnl = np.cumsum(pnls)
        fig_dist.add_trace(go.Scatter(
            y=cum_pnl, mode="lines",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
            name="Cum P&L",
        ), row=1, col=2)

        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4  —  MARKOV MATRIX
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### 🔬 Transition Matrix Visualization")
    if df is not None and len(df) > 50:
        prices_arr = df["close"].values
        returns    = np.diff(np.log(prices_arr))

        p = result.params
        chain = MarkovChain(p)
        chain.fit(returns[-max(p.lookback, 100):])

        T = chain.transition_matrix
        n = p.n_states
        state_labels = [f"S{i}" for i in range(n)]

        # Annotated heatmap
        fig_tm = go.Figure(go.Heatmap(
            z=T,
            x=state_labels,
            y=state_labels,
            colorscale=[
                [0,   "#0a0e1a"],
                [0.3, "#0f2040"],
                [0.6, "#0e4080"],
                [1,   "#00d4ff"],
            ],
            text=np.round(T, 3),
            texttemplate="%{text}",
            textfont={"size": 11, "family": "JetBrains Mono"},
            showscale=True,
        ))
        fig_tm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            height=420,
            title=dict(text="Markov Transition Matrix T[i→j]",
                       font=dict(color="#00d4ff", family="JetBrains Mono")),
            xaxis=dict(title="Next State", side="bottom"),
            yaxis=dict(title="Current State", autorange="reversed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_tm, use_container_width=True)

        # State stationary distribution
        st.markdown("#### 🌀 Stationary Distribution π")
        try:
            eigvals, eigvecs = np.linalg.eig(T.T)
            idx = np.argmin(np.abs(eigvals - 1))
            stationary = np.real(eigvecs[:, idx])
            stationary = stationary / stationary.sum()

            fig_sd = go.Figure(go.Bar(
                x=state_labels,
                y=stationary,
                marker_color=["#ff4466" if i < n//2 else "#00ff88" if i > n//2 else "#ffd700"
                               for i in range(n)],
                text=[f"{v:.3f}" for v in stationary],
                textposition="outside",
            ))
            fig_sd.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0a0e1a",
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(gridcolor="#1e2d4a"),
                xaxis=dict(title="Market State (low → high return)"),
            )
            st.plotly_chart(fig_sd, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                bull_states = stationary[n//2+1:].sum()
                bear_states = stationary[:n//2].sum()
                metric_card("Bullish Regime Prob", f"{bull_states*100:.1f}%", "green")
            with col2:
                metric_card("Bearish Regime Prob", f"{bear_states*100:.1f}%", "red")
        except Exception:
            st.info("Could not compute stationary distribution.")
    else:
        st.warning("Load data first (run backtest).")


# ══════════════════════════════════════════════
#  TAB 5  —  OPTIMIZER RESULTS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("#### 🏆 Top Parameter Configurations")
    all_results: list[BacktestResult] = st.session_state.get("all_results", [])

    if not all_results:
        if result:
            all_results = [result]
        else:
            st.info("Run the optimizer to see ranked configurations.")
            st.stop()

    rows = []
    for i, r in enumerate(all_results):
        rows.append({
            "Rank": i + 1,
            "Score": round(r.score, 4),
            "Sharpe": round(r.sharpe, 3),
            "Return %": round(r.total_return * 100, 2),
            "MaxDD %": round(r.max_drawdown * 100, 2),
            "Win Rate %": round(r.win_rate * 100, 1),
            "PF": round(r.profit_factor, 2),
            "Trades": r.n_trades,
            "States": r.params.n_states,
            "Lookback": r.params.lookback,
            "HP": r.params.holding_period,
            "Thresh": r.params.entry_threshold,
            "SL%": round(r.params.stop_loss_pct * 100, 2),
            "TP%": round(r.params.take_profit_pct * 100, 2),
            "Method": r.params.state_method,
        })

    opt_df = pd.DataFrame(rows)
    st.dataframe(opt_df, use_container_width=True, hide_index=True)

    # Score scatter
    if len(all_results) > 3:
        st.markdown("#### 📡 Score vs Sharpe (all configurations)")
        scores = [r.score for r in all_results]
        sharpes = [r.sharpe for r in all_results]
        rets = [r.total_return for r in all_results]

        fig_sc = go.Figure(go.Scatter(
            x=sharpes, y=scores,
            mode="markers",
            marker=dict(
                size=6,
                color=rets,
                colorscale=[[0, "#ff4466"], [0.5, "#ffd700"], [1, "#00ff88"]],
                colorbar=dict(title="Return", titlefont_color="#00d4ff"),
                showscale=True,
            ),
            text=[f"States={r.params.n_states}, LB={r.params.lookback}" for r in all_results],
        ))
        fig_sc.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            height=380,
            xaxis=dict(title="Sharpe Ratio", gridcolor="#1e2d4a"),
            yaxis=dict(title="Composite Score", gridcolor="#1e2d4a"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0; color:#1e2d4a; font-family:'JetBrains Mono'; font-size:10px;">
  ⚠️ For research and educational purposes only. Not financial advice. Past performance ≠ future results.
</div>
""", unsafe_allow_html=True)
