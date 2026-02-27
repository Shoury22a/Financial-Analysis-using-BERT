"""
FINSIGHT AI - Streamlit Application
AI-Powered Financial Sentiment Analysis & Stock Intelligence
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FINSIGHT AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark theme base */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0f1e 100%);
}

/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid rgba(0, 212, 170, 0.15);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(0, 212, 170, 0.2);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0, 212, 170, 0.5);
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.1);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
}
.metric-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.4);
    margin-top: 4px;
}

/* Sentiment result cards */
.sentiment-positive {
    background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.sentiment-negative {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(185,28,28,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.sentiment-neutral {
    background: linear-gradient(135deg, rgba(234,179,8,0.15) 0%, rgba(161,98,7,0.08) 100%);
    border: 1px solid rgba(234,179,8,0.4);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00d4aa;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0, 212, 170, 0.2);
}

/* Stock info badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    color: #00d4aa;
    margin-right: 6px;
}

/* Input styling */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,170,0.25) !important;
    border-radius: 10px !important;
    color: white !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0,212,170,0.6) !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #0ea5e9 100%);
    color: #0a0a1a;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 14px;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9;
    box-shadow: 0 0 20px rgba(0,212,170,0.4);
    transform: translateY(-1px);
}

/* Dividers */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 24px 0;
}

/* Main title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #0ea5e9 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.main-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.5);
    margin-top: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,170,0.2) 0%, rgba(14,165,233,0.2) 100%) !important;
    color: #00d4aa !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4aa, #0ea5e9) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,170,0.5); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load FinBERT model (PyTorch/safetensors) - tries fine-tuned first, falls back to base."""
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        MODEL_PATH = "financial_sentiment_model"
        if not os.path.exists(MODEL_PATH):
            MODEL_PATH = "ProsusAI/finbert"

        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

        # Read label mapping from config
        config_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label", {})
            label_map = {int(k): v.capitalize() for k, v in id2label.items()}
        else:
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        return model, tokenizer, label_map, MODEL_PATH
    except Exception as e:
        return None, None, None, str(e)


def predict_sentiment(text, model, tokenizer, label_map):
    """Run inference and return (sentiment_label, probabilities_dict)."""
    import torch
    encodings = tokenizer(
        text, truncation=True, padding=True,
        max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encodings)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

    max_idx = int(np.argmax(probs))
    sentiment = label_map[max_idx]

    # Build prob dict from label_map
    prob_dict = {v.capitalize(): round(float(probs[k]), 4) for k, v in label_map.items()}
    return sentiment, prob_dict


# ─────────────────────────────────────────────
# STOCK DATA HELPERS
# ─────────────────────────────────────────────
GLOBAL_STOCKS = {
    # US
    'AAPL': ('Apple Inc.', 'Technology'), 'MSFT': ('Microsoft Corp', 'Technology'),
    'GOOGL': ('Alphabet (Google)', 'Technology'), 'AMZN': ('Amazon', 'Consumer'),
    'NVDA': ('NVIDIA Corp', 'Technology'), 'META': ('Meta Platforms', 'Technology'),
    'TSLA': ('Tesla Inc.', 'Automotive'), 'JPM': ('JPMorgan Chase', 'Finance'),
    'V': ('Visa Inc.', 'Finance'), 'JNJ': ('Johnson & Johnson', 'Healthcare'),
    'WMT': ('Walmart', 'Retail'), 'MA': ('Mastercard', 'Finance'),
    'DIS': ('Walt Disney', 'Entertainment'), 'NFLX': ('Netflix', 'Entertainment'),
    'AMD': ('AMD', 'Technology'), 'INTC': ('Intel', 'Technology'),
    'BA': ('Boeing', 'Aerospace'), 'PYPL': ('PayPal', 'Finance'),
    # India
    'RELIANCE.NS': ('Reliance Industries', 'Energy'), 'TCS.NS': ('TCS', 'Technology'),
    'HDFCBANK.NS': ('HDFC Bank', 'Finance'), 'INFY.NS': ('Infosys', 'Technology'),
    'ICICIBANK.NS': ('ICICI Bank', 'Finance'), 'SBIN.NS': ('SBI', 'Finance'),
    'TATAMOTORS.NS': ('Tata Motors', 'Automotive'), 'WIPRO.NS': ('Wipro', 'Technology'),
    # Asia / Europe
    'BABA': ('Alibaba', 'Technology'), 'TSM': ('TSMC', 'Technology'),
    'ASML': ('ASML Holding', 'Technology'), 'NVO': ('Novo Nordisk', 'Healthcare'),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock(ticker: str, period: str = "1mo"):
    """Fetch stock OHLCV + info using yfinance with browser headers to avoid blocks."""
    import requests
    try:
        # Use a requests session with browser User-Agent to avoid Yahoo Finance blocks
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period=period)
        if hist.empty:
            return None, {}
        info = {}
        try:
            raw = stock.info
            info = {
                "name":             raw.get("longName", ticker),
                "sector":           raw.get("sector", "N/A"),
                "currentPrice":     raw.get("currentPrice") or (float(hist["Close"].iloc[-1]) if not hist.empty else None),
                "previousClose":    raw.get("previousClose"),
                "marketCap":        raw.get("marketCap"),
                "volume":           raw.get("volume"),
                "fiftyTwoWeekHigh": raw.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow":  raw.get("fiftyTwoWeekLow"),
                "peRatio":          raw.get("trailingPE"),
                "dividendYield":    raw.get("dividendYield"),
                "beta":             raw.get("beta"),
            }
        except Exception:
            info = {"name": ticker}
        return hist, info
    except Exception as e:
        return None, {"error": str(e)}


def format_large_num(n):
    if n is None:
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def make_candlestick(hist: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        increasing_fillcolor="rgba(16,185,129,0.8)",
        decreasing_fillcolor="rgba(239,68,68,0.8)",
        name="Price",
    ), row=1, col=1)
    # Volume bars
    colors = ["#10b981" if c >= o else "#ef4444"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color=colors, name="Volume", opacity=0.6,
    ), row=2, col=1)
    # 20-day SMA
    if len(hist) >= 20:
        sma = hist["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=sma, line=dict(color="#a78bfa", width=1.5),
            name="SMA 20",
        ), row=1, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(255,255,255,0.03)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=480,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True, zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True, zeroline=False,
    )
    return fig


def make_probability_gauge(probs: dict):
    labels = list(probs.keys())
    values = [round(v * 100, 1) for v in probs.values()]
    colors = ["#10b981", "#ef4444", "#f59e0b"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=14, family="Inter"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
        yaxis=dict(range=[0, 110], showgrid=False, showticklabels=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=10, r=10, t=20, b=10),
        height=220,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:42px; margin-bottom:8px;'>🔮</div>
        <div style='font-size:22px; font-weight:800; background:linear-gradient(135deg,#00d4aa,#0ea5e9);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;'>
            FINSIGHT AI
        </div>
        <div style='font-size:11px; color:rgba(255,255,255,0.4); letter-spacing:2px; margin-top:4px;'>
            FINANCIAL INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🔮 Sentiment Analysis", "📈 Stock Explorer", "📋 Batch Analysis", "📚 Glossary"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Model status
    model, tokenizer, label_map, model_path = load_model()
    if model is not None:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
             border-radius:10px; padding:12px; text-align:center;'>
            <div style='font-size:11px; color:#10b981; font-weight:700; letter-spacing:1px;'>
                ✅ MODEL LOADED
            </div>
            <div style='font-size:10px; color:rgba(255,255,255,0.4); margin-top:4px;'>
                FinBERT · 3-Class Classifier
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Model failed to load")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:rgba(255,255,255,0.3); text-align:center; line-height:1.8;'>
        Built with FinBERT · TensorFlow<br>
        yfinance · Plotly · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════
if page == "🔮 Sentiment Analysis":
    st.markdown("""
    <div class='main-title'>Sentiment Analysis</div>
    <div class='main-subtitle'>
        Powered by FinBERT — a financial-domain BERT model trained on thousands of news headlines.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-header">Analyze Text</div>', unsafe_allow_html=True)
        news_text = st.text_area(
            "Enter financial news headline or paragraph:",
            placeholder="e.g. Apple reports record quarterly profits driven by iPhone sales surge...",
            height=140,
            label_visibility="collapsed",
        )

        # Example prompts
        st.markdown('<div style="font-size:12px; color:rgba(255,255,255,0.4); margin-bottom:6px;">Try an example:</div>', unsafe_allow_html=True)
        examples = [
            "🟢 Apple beats earnings expectations with record revenue.",
            "🔴 Company reports massive losses amid rising debt concerns.",
            "🟡 The Federal Reserve holds interest rates steady this quarter.",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                news_text = ex[2:].strip()
                st.session_state["_example_text"] = news_text
                st.rerun()

        if "_example_text" in st.session_state and not news_text:
            news_text = st.session_state["_example_text"]

        analyze_btn = st.button("🔮 Analyze Sentiment", key="analyze")

    with col2:
        st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()

        if analyze_btn and news_text.strip():
            if model is None:
                result_placeholder.error("Model not loaded. Please check the logs.")
            else:
                with st.spinner("Running FinBERT inference..."):
                    sentiment, probs = predict_sentiment(news_text.strip(), model, tokenizer, label_map)

                ICONS = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                STYLES = {"Positive": "sentiment-positive", "Negative": "sentiment-negative", "Neutral": "sentiment-neutral"}
                COLORS = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}

                icon  = ICONS.get(sentiment, "⚪")
                style = STYLES.get(sentiment, "sentiment-neutral")
                color = COLORS.get(sentiment, "#ffffff")

                result_placeholder.markdown(f"""
                <div class="{style}">
                    <div style='font-size:40px; margin-bottom:8px;'>{icon}</div>
                    <div style='font-size:30px; font-weight:800; color:{color};'>{sentiment}</div>
                    <div style='font-size:12px; color:rgba(255,255,255,0.4); margin-top:6px;'>
                        Confidence: {probs[sentiment]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
                fig = make_probability_gauge(probs)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Confidence breakdown
                for label, val in probs.items():
                    c = COLORS.get(label, "#ffffff")
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; margin-bottom:8px;'>
                        <div style='width:80px; font-size:12px; color:rgba(255,255,255,0.6);'>{label}</div>
                        <div style='flex:1; height:6px; background:rgba(255,255,255,0.06);
                             border-radius:3px; overflow:hidden; margin: 0 10px;'>
                            <div style='height:100%; width:{val*100:.1f}%; background:{c};
                                 border-radius:3px; transition:width 0.5s ease;'></div>
                        </div>
                        <div style='width:50px; font-size:12px; color:{c}; text-align:right; font-weight:600;'>
                            {val*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        elif analyze_btn:
            result_placeholder.warning("Please enter some text to analyze.")
        else:
            result_placeholder.markdown("""
            <div style='background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.1);
                 border-radius:16px; padding:40px; text-align:center;'>
                <div style='font-size:36px; margin-bottom:10px;'>✍️</div>
                <div style='color:rgba(255,255,255,0.35); font-size:14px;'>
                    Enter text and click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: STOCK EXPLORER
# ═══════════════════════════════════════════════
elif page == "📈 Stock Explorer":
    st.markdown("""
    <div class='main-title'>Stock Explorer</div>
    <div class='main-subtitle'>Real-time price data, candlestick charts, and key financial metrics.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Session state for ticker so quick picks actually work ──
    if "stock_ticker" not in st.session_state:
        st.session_state["stock_ticker"] = "AAPL"

    # Quick picks
    st.markdown('<div style="font-size:12px; color:rgba(255,255,255,0.4); margin-bottom:8px;">Quick picks:</div>', unsafe_allow_html=True)
    quick_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "TCS.NS", "RELIANCE.NS", "INFY.NS"]
    qcols = st.columns(len(quick_tickers))
    for i, tk in enumerate(quick_tickers):
        if qcols[i].button(tk, key=f"qt_{tk}", use_container_width=True):
            st.session_state["stock_ticker"] = tk
            st.rerun()

    col_search, col_period = st.columns([3, 1])
    with col_search:
        ticker_input = st.text_input(
            "Stock Ticker",
            value=st.session_state["stock_ticker"],
            placeholder="e.g. AAPL, TSLA, RELIANCE.NS …",
            label_visibility="collapsed",
        )
        if ticker_input.strip().upper() != st.session_state["stock_ticker"]:
            st.session_state["stock_ticker"] = ticker_input.strip().upper()
    with col_period:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "5y"],
            label_visibility="collapsed",
        )

    st.markdown("---")

    active_ticker = st.session_state.get("stock_ticker", "AAPL").strip().upper()
    if active_ticker:
        with st.spinner(f"Fetching data for {active_ticker}..."):
            hist, info = fetch_stock(active_ticker, period)

        if hist is None or hist.empty:
            st.error(f"⚠️ Could not fetch data for **{active_ticker}**. Yahoo Finance may be rate-limiting — wait a few seconds and try again.")
        else:
            current   = float(hist["Close"].iloc[-1])
            prev_raw  = info.get("previousClose")
            prev      = float(prev_raw) if prev_raw else (float(hist["Close"].iloc[-2]) if len(hist) > 1 else current)
            change    = current - prev
            chg_pct   = (change / prev) * 100 if prev else 0
            chg_color = "#10b981" if change >= 0 else "#ef4444"
            chg_arrow = "▲" if change >= 0 else "▼"

            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:20px; flex-wrap:wrap; margin-bottom:24px;'>
                <div>
                    <div style='font-size:32px; font-weight:800; color:white;'>
                        {info.get("name", active_ticker)}
                    </div>
                    <div style='margin-top:4px;'>
                        <span class='badge'>{active_ticker}</span>
                        <span class='badge'>{info.get("sector","N/A")}</span>
                    </div>
                </div>
                <div style='margin-left:auto; text-align:right;'>
                    <div style='font-size:40px; font-weight:800; color:white;'>${current:,.2f}</div>
                    <div style='font-size:18px; font-weight:600; color:{chg_color};'>
                        {chg_arrow} {abs(change):,.2f} ({abs(chg_pct):.2f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            metrics = [
                (m1, "MARKET CAP",    format_large_num(info.get("marketCap"))),
                (m2, "52W HIGH",      f"${info.get('fiftyTwoWeekHigh', 'N/A')}" if info.get('fiftyTwoWeekHigh') else "N/A"),
                (m3, "52W LOW",       f"${info.get('fiftyTwoWeekLow', 'N/A')}" if info.get('fiftyTwoWeekLow') else "N/A"),
                (m4, "P/E RATIO",     f"{info.get('peRatio', 'N/A'):.1f}" if info.get('peRatio') else "N/A"),
                (m5, "BETA",          f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A"),
            ]
            for col, label, value in metrics:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value' style='font-size:20px;'>{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(make_candlestick(hist, active_ticker), use_container_width=True,
                            config={"displayModeBar": False})

            # Volume + Returns
            col_v, col_r = st.columns(2)
            with col_v:
                st.markdown('<div class="section-header">Recent OHLCV</div>', unsafe_allow_html=True)
                display = hist.tail(10)[["Open","High","Low","Close","Volume"]].copy()
                display = display.round(2)
                display.index = display.index.strftime("%b %d")
                st.dataframe(display, use_container_width=True)
            with col_r:
                st.markdown('<div class="section-header">Daily Returns Distribution</div>', unsafe_allow_html=True)
                returns = hist["Close"].pct_change().dropna() * 100
                fig_ret = px.histogram(
                    returns, nbins=30,
                    color_discrete_sequence=["#00d4aa"],
                    labels={"value": "Daily Return (%)"},
                )
                fig_ret.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                    height=260, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig_ret, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════
# PAGE: BATCH ANALYSIS
# ═══════════════════════════════════════════════
elif page == "📋 Batch Analysis":
    st.markdown("""
    <div class='main-title'>Batch Analysis</div>
    <div class='main-subtitle'>Analyze multiple headlines at once and visualize the sentiment distribution.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    default_batch = """Apple reports record quarterly profits driven by iPhone 15 sales.
Tesla stock plunges after missing delivery targets by 20%.
Federal Reserve holds interest rates steady amid mixed economic signals.
NVIDIA surges on AI chip demand, data center revenue triples.
Startup lays off 30% of workforce citing macroeconomic headwinds.
Warren Buffett increases stake in Japanese trading companies.
Oil prices fall sharply on weak China demand data.
Amazon Web Services growth accelerates, beating analyst expectations.
Regulators fine major bank $2 billion for compliance failures.
Inflation data comes in line with expectations, markets react mildly."""

    headlines_input = st.text_area(
        "Enter headlines (one per line):",
        value=default_batch,
        height=250,
        label_visibility="collapsed",
    )

    if st.button("🔮 Analyze All Headlines", key="batch_analyze"):
        if model is None:
            st.error("Model not loaded.")
        else:
            lines = [l.strip() for l in headlines_input.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one headline.")
            else:
                results = []
                progress = st.progress(0)
                status_text = st.empty()

                for i, line in enumerate(lines):
                    status_text.markdown(f'<div style="color:rgba(255,255,255,0.5);font-size:13px;">Analyzing {i+1}/{len(lines)}: {line[:60]}...</div>', unsafe_allow_html=True)
                    sentiment, probs = predict_sentiment(line, model, tokenizer, label_map)
                    results.append({
                        "Headline": line,
                        "Sentiment": sentiment,
                        "Positive %": f"{probs.get('Positive', 0)*100:.1f}",
                        "Negative %": f"{probs.get('Negative', 0)*100:.1f}",
                        "Neutral %":  f"{probs.get('Neutral', 0)*100:.1f}",
                        "Confidence %": f"{max(probs.values())*100:.1f}",
                    })
                    progress.progress((i + 1) / len(lines))

                status_text.empty()
                progress.empty()

                df = pd.DataFrame(results)

                # Summary metrics
                counts = df["Sentiment"].value_counts()
                total = len(df)
                c1, c2, c3, c4 = st.columns(4)
                for col, label, icon, color in [
                    (c1, "Total", "📰", "#ffffff"),
                    (c2, "Positive", "🟢", "#10b981"),
                    (c3, "Negative", "🔴", "#ef4444"),
                    (c4, "Neutral",  "🟡", "#f59e0b"),
                ]:
                    val = total if label == "Total" else counts.get(label, 0)
                    with col:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size:24px;'>{icon}</div>
                            <div class='metric-label'>{label}</div>
                            <div class='metric-value' style='color:{color};'>{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_pie, col_bar = st.columns(2)
                with col_pie:
                    st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
                    pie_counts = df["Sentiment"].value_counts()
                    fig_pie = go.Figure(go.Pie(
                        labels=pie_counts.index,
                        values=pie_counts.values,
                        marker=dict(colors=["#10b981","#ef4444","#f59e0b"]),
                        hole=0.5,
                        textfont=dict(size=13, family="Inter"),
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=280,
                        legend=dict(
                            bgcolor="rgba(255,255,255,0.03)",
                            bordercolor="rgba(255,255,255,0.1)",
                            borderwidth=1,
                        ),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                with col_bar:
                    st.markdown('<div class="section-header">Confidence per Headline</div>', unsafe_allow_html=True)
                    conf_vals = [float(r["Confidence %"]) for r in results]
                    color_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}
                    bar_colors = [color_map.get(r["Sentiment"], "#ffffff") for r in results]
                    fig_conf = go.Figure(go.Bar(
                        x=[f"#{i+1}" for i in range(len(results))],
                        y=conf_vals,
                        marker_color=bar_colors,
                        text=[f"{v:.0f}%" for v in conf_vals],
                        textposition="outside",
                        textfont=dict(color="white", size=11),
                    ))
                    fig_conf.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=280,
                        yaxis=dict(range=[0, 115], gridcolor="rgba(255,255,255,0.04)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

                # Results table — static HTML (no shaking)
                st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)

                SENT_COLORS = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}
                rows_html = ""
                for _, row in df.iterrows():
                    sc = SENT_COLORS.get(row["Sentiment"], "#ffffff")
                    rows_html += f"""
                    <tr>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:rgba(255,255,255,0.85); font-size:13px; max-width:400px;'>{row['Headline']}</td>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:{sc}; font-weight:700; white-space:nowrap;'>{row['Sentiment']}</td>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:rgba(255,255,255,0.6); text-align:center;'>{row['Positive %']}%</td>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:rgba(255,255,255,0.6); text-align:center;'>{row['Negative %']}%</td>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:rgba(255,255,255,0.6); text-align:center;'>{row['Neutral %']}%</td>
                        <td style='padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
                                   color:{sc}; font-weight:600; text-align:center;'>{row['Confidence %']}%</td>
                    </tr>"""

                st.markdown(f"""
                <div style='overflow-x:auto;'>
                <table style='width:100%; border-collapse:collapse;
                              background:rgba(255,255,255,0.02);
                              border:1px solid rgba(255,255,255,0.06);
                              border-radius:12px; overflow:hidden;'>
                    <thead>
                        <tr style='background:rgba(0,212,170,0.08);'>
                            <th style='padding:12px 14px; text-align:left; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Headline</th>
                            <th style='padding:12px 14px; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Sentiment</th>
                            <th style='padding:12px 14px; text-align:center; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Positive</th>
                            <th style='padding:12px 14px; text-align:center; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Negative</th>
                            <th style='padding:12px 14px; text-align:center; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Neutral</th>
                            <th style='padding:12px 14px; text-align:center; font-size:11px; font-weight:700;
                                       letter-spacing:1px; color:rgba(255,255,255,0.5); text-transform:uppercase;'>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)

                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results (CSV)",
                    csv,
                    file_name=f"finsight_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


# ═══════════════════════════════════════════════
# PAGE: GLOSSARY
# ═══════════════════════════════════════════════
elif page == "📚 Glossary":
    st.markdown("""
    <div class='main-title'>Financial Glossary</div>
    <div class='main-subtitle'>Key terms every investor should know.</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    terms = [
        ("Bull Market 🐂", "A market condition where prices are rising or expected to rise, typically by 20%+ from recent lows. Associated with investor confidence and economic growth."),
        ("Bear Market 🐻", "A market condition where prices fall 20%+ from recent highs over a sustained period. Often linked to economic downturns or recessions."),
        ("P/E Ratio 📊", "Price-to-Earnings ratio. Measures how much investors pay per dollar of earnings. High P/E may indicate overvaluation; low P/E may signal undervaluation."),
        ("Market Cap 💰", "Total market value of a company's outstanding shares = Stock Price × Shares Outstanding. Classifies companies as small-cap, mid-cap, or large-cap."),
        ("RSI 📈", "Relative Strength Index. A momentum oscillator (0-100). Above 70 = overbought (potential sell signal). Below 30 = oversold (potential buy signal)."),
        ("SMA 〰️", "Simple Moving Average. The average closing price over N periods. Used to smooth price action and identify trends."),
        ("Dividend Yield 💵", "Annual dividends per share ÷ stock price × 100%. Measures income generated per dollar invested."),
        ("Beta ⚡", "Measure of a stock's volatility relative to the market. Beta > 1 = more volatile than market; Beta < 1 = less volatile."),
        ("IPO 🚀", "Initial Public Offering. When a private company first sells shares to the public on a stock exchange."),
        ("ETF 🗂️", "Exchange-Traded Fund. A basket of securities that trades on an exchange like a single stock, offering instant diversification."),
        ("Short Selling 📉", "Borrowing and selling shares you don't own, hoping to buy them back cheaper later. Profit if price falls; loss if it rises."),
        ("EPS 🧮", "Earnings Per Share = Net Income ÷ Shares Outstanding. Key indicator of a company's profitability per share."),
        ("EBITDA 🏭", "Earnings Before Interest, Taxes, Depreciation & Amortization. Used to evaluate operational profitability independently of financing."),
        ("Stop Loss 🛑", "An order to automatically sell a security when it reaches a set price, limiting potential losses."),
        ("Dollar-Cost Averaging 🔄", "Investing a fixed amount at regular intervals regardless of price, reducing the impact of volatility over time."),
    ]

    cols = st.columns(2)
    for i, (term, definition) in enumerate(terms):
        with cols[i % 2]:
            with st.expander(term):
                st.markdown(f'<div style="color:rgba(255,255,255,0.75); font-size:14px; line-height:1.7;">{definition}</div>', unsafe_allow_html=True)
