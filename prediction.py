import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import os
import yfinance as yf
from yahooquery import Ticker
import finnhub
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="FINSIGHT AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #00a8e8 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 212, 170, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(0, 168, 232, 0.5)); }
    }
    
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 212, 170, 0.2);
        border-color: rgba(0, 212, 170, 0.3);
    }
    
    .positive { color: #00d4aa !important; }
    .negative { color: #ff6b6b !important; }
    .neutral { color: #ffd93d !important; }
    
    .recommendation {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: pulse 2s infinite;
    }
    
    .recommendation.buy {
        background: linear-gradient(135deg, #00d4aa 0%, #00a86b 100%);
        color: #0f0f23;
    }
    
    .recommendation.hold {
        background: linear-gradient(135deg, #ffd93d 0%, #ff9500 100%);
        color: #0f0f23;
    }
    
    .recommendation.sell {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 170, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00a8e8 100%);
        color: #0f0f23;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 212, 170, 0.4);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 170, 0.5), transparent);
        margin: 2rem 0;
    }
    
    .term-card {
        background: rgba(0, 212, 170, 0.08);
        border-left: 4px solid #00d4aa;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .term-title {
        color: #00d4aa;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    
    .term-desc {
        color: #a0aec0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .region-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    .region-us { background: rgba(0, 168, 232, 0.2); color: #00a8e8; }
    .region-india { background: rgba(255, 153, 0, 0.2); color: #ff9900; }
    .region-asia { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }
    .region-europe { background: rgba(124, 58, 237, 0.2); color: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# ==================== GLOBAL STOCK DATABASE ====================
GLOBAL_STOCKS = {
    # ===== US STOCKS (S&P 500 Major) =====
    'AAPL': ('Apple Inc.', 'US', 'Technology'),
    'MSFT': ('Microsoft Corporation', 'US', 'Technology'),
    'GOOGL': ('Alphabet Inc. (Google)', 'US', 'Technology'),
    'GOOG': ('Alphabet Inc. Class C', 'US', 'Technology'),
    'AMZN': ('Amazon.com Inc.', 'US', 'Consumer'),
    'NVDA': ('NVIDIA Corporation', 'US', 'Technology'),
    'META': ('Meta Platforms Inc.', 'US', 'Technology'),
    'TSLA': ('Tesla Inc.', 'US', 'Automotive'),
    'BRK-B': ('Berkshire Hathaway', 'US', 'Finance'),
    'UNH': ('UnitedHealth Group', 'US', 'Healthcare'),
    'JNJ': ('Johnson & Johnson', 'US', 'Healthcare'),
    'JPM': ('JPMorgan Chase & Co.', 'US', 'Finance'),
    'V': ('Visa Inc.', 'US', 'Finance'),
    'PG': ('Procter & Gamble Co.', 'US', 'Consumer'),
    'MA': ('Mastercard Inc.', 'US', 'Finance'),
    'HD': ('The Home Depot Inc.', 'US', 'Retail'),
    'CVX': ('Chevron Corporation', 'US', 'Energy'),
    'MRK': ('Merck & Co. Inc.', 'US', 'Healthcare'),
    'ABBV': ('AbbVie Inc.', 'US', 'Healthcare'),
    'LLY': ('Eli Lilly and Company', 'US', 'Healthcare'),
    'PEP': ('PepsiCo Inc.', 'US', 'Consumer'),
    'KO': ('The Coca-Cola Company', 'US', 'Consumer'),
    'COST': ('Costco Wholesale Corp.', 'US', 'Retail'),
    'WMT': ('Walmart Inc.', 'US', 'Retail'),
    'BAC': ('Bank of America Corp.', 'US', 'Finance'),
    'DIS': ('The Walt Disney Company', 'US', 'Entertainment'),
    'NFLX': ('Netflix Inc.', 'US', 'Entertainment'),
    'ADBE': ('Adobe Inc.', 'US', 'Technology'),
    'CRM': ('Salesforce Inc.', 'US', 'Technology'),
    'AMD': ('Advanced Micro Devices', 'US', 'Technology'),
    'INTC': ('Intel Corporation', 'US', 'Technology'),
    'CSCO': ('Cisco Systems Inc.', 'US', 'Technology'),
    'ORCL': ('Oracle Corporation', 'US', 'Technology'),
    'QCOM': ('Qualcomm Inc.', 'US', 'Technology'),
    'IBM': ('IBM Corporation', 'US', 'Technology'),
    'NKE': ('Nike Inc.', 'US', 'Consumer'),
    'MCD': ('McDonalds Corporation', 'US', 'Consumer'),
    'SBUX': ('Starbucks Corporation', 'US', 'Consumer'),
    'BA': ('Boeing Company', 'US', 'Aerospace'),
    'GE': ('General Electric Co.', 'US', 'Industrial'),
    'CAT': ('Caterpillar Inc.', 'US', 'Industrial'),
    'GS': ('Goldman Sachs Group', 'US', 'Finance'),
    'MS': ('Morgan Stanley', 'US', 'Finance'),
    'AXP': ('American Express Co.', 'US', 'Finance'),
    'PYPL': ('PayPal Holdings Inc.', 'US', 'Finance'),
    'UBER': ('Uber Technologies', 'US', 'Technology'),
    'ABNB': ('Airbnb Inc.', 'US', 'Travel'),
    'ZM': ('Zoom Video Communications', 'US', 'Technology'),
    'SQ': ('Block Inc. (Square)', 'US', 'Finance'),
    'SHOP': ('Shopify Inc.', 'US', 'Technology'),
    'SNAP': ('Snap Inc.', 'US', 'Technology'),
    'ROKU': ('Roku Inc.', 'US', 'Technology'),
    'SPOT': ('Spotify Technology', 'US', 'Entertainment'),
    'TWTR': ('Twitter/X Corp', 'US', 'Technology'),
    'COIN': ('Coinbase Global', 'US', 'Finance'),
    'PLTR': ('Palantir Technologies', 'US', 'Technology'),
    'RIVN': ('Rivian Automotive', 'US', 'Automotive'),
    'LCID': ('Lucid Group', 'US', 'Automotive'),
    'F': ('Ford Motor Company', 'US', 'Automotive'),
    'GM': ('General Motors', 'US', 'Automotive'),
    
    # ===== INDIA STOCKS (NIFTY 50 + Popular) =====
    'RELIANCE.NS': ('Reliance Industries Ltd', 'India', 'Energy'),
    'TCS.NS': ('Tata Consultancy Services', 'India', 'Technology'),
    'HDFCBANK.NS': ('HDFC Bank Limited', 'India', 'Finance'),
    'INFY.NS': ('Infosys Limited', 'India', 'Technology'),
    'ICICIBANK.NS': ('ICICI Bank Limited', 'India', 'Finance'),
    'HINDUNILVR.NS': ('Hindustan Unilever', 'India', 'Consumer'),
    'SBIN.NS': ('State Bank of India', 'India', 'Finance'),
    'BHARTIARTL.NS': ('Bharti Airtel Limited', 'India', 'Telecom'),
    'ITC.NS': ('ITC Limited', 'India', 'Consumer'),
    'KOTAKBANK.NS': ('Kotak Mahindra Bank', 'India', 'Finance'),
    'LT.NS': ('Larsen & Toubro', 'India', 'Industrial'),
    'AXISBANK.NS': ('Axis Bank Limited', 'India', 'Finance'),
    'ASIANPAINT.NS': ('Asian Paints Limited', 'India', 'Consumer'),
    'MARUTI.NS': ('Maruti Suzuki India', 'India', 'Automotive'),
    'SUNPHARMA.NS': ('Sun Pharmaceutical', 'India', 'Healthcare'),
    'TITAN.NS': ('Titan Company Limited', 'India', 'Consumer'),
    'BAJFINANCE.NS': ('Bajaj Finance Limited', 'India', 'Finance'),
    'WIPRO.NS': ('Wipro Limited', 'India', 'Technology'),
    'HCLTECH.NS': ('HCL Technologies', 'India', 'Technology'),
    'TATAMOTORS.NS': ('Tata Motors Limited', 'India', 'Automotive'),
    'TATASTEEL.NS': ('Tata Steel Limited', 'India', 'Industrial'),
    'POWERGRID.NS': ('Power Grid Corporation', 'India', 'Energy'),
    'NTPC.NS': ('NTPC Limited', 'India', 'Energy'),
    'ONGC.NS': ('Oil & Natural Gas Corp', 'India', 'Energy'),
    'COALINDIA.NS': ('Coal India Limited', 'India', 'Energy'),
    'JSWSTEEL.NS': ('JSW Steel Limited', 'India', 'Industrial'),
    'ADANIENT.NS': ('Adani Enterprises', 'India', 'Industrial'),
    'ADANIPORTS.NS': ('Adani Ports & SEZ', 'India', 'Industrial'),
    'TECHM.NS': ('Tech Mahindra Limited', 'India', 'Technology'),
    'ULTRACEMCO.NS': ('UltraTech Cement', 'India', 'Industrial'),
    'BAJAJFINSV.NS': ('Bajaj Finserv Limited', 'India', 'Finance'),
    'NESTLEIND.NS': ('Nestle India Limited', 'India', 'Consumer'),
    'DIVISLAB.NS': ('Divis Laboratories', 'India', 'Healthcare'),
    'DRREDDY.NS': ('Dr. Reddys Laboratories', 'India', 'Healthcare'),
    'CIPLA.NS': ('Cipla Limited', 'India', 'Healthcare'),
    'EICHERMOT.NS': ('Eicher Motors Limited', 'India', 'Automotive'),
    'HEROMOTOCO.NS': ('Hero MotoCorp Limited', 'India', 'Automotive'),
    'BAJAJ-AUTO.NS': ('Bajaj Auto Limited', 'India', 'Automotive'),
    'M&M.NS': ('Mahindra & Mahindra', 'India', 'Automotive'),
    'BRITANNIA.NS': ('Britannia Industries', 'India', 'Consumer'),
    'APOLLOHOSP.NS': ('Apollo Hospitals', 'India', 'Healthcare'),
    'GRASIM.NS': ('Grasim Industries', 'India', 'Industrial'),
    'INDUSINDBK.NS': ('IndusInd Bank Limited', 'India', 'Finance'),
    'SBILIFE.NS': ('SBI Life Insurance', 'India', 'Finance'),
    'HDFCLIFE.NS': ('HDFC Life Insurance', 'India', 'Finance'),
    
    # ===== CHINA STOCKS =====
    'BABA': ('Alibaba Group Holdings', 'China', 'Technology'),
    '9988.HK': ('Alibaba Group (HK)', 'China', 'Technology'),
    'JD': ('JD.com Inc.', 'China', 'Technology'),
    'PDD': ('PDD Holdings (Pinduoduo)', 'China', 'Technology'),
    'BIDU': ('Baidu Inc.', 'China', 'Technology'),
    'NIO': ('NIO Inc.', 'China', 'Automotive'),
    'XPEV': ('XPeng Inc.', 'China', 'Automotive'),
    'LI': ('Li Auto Inc.', 'China', 'Automotive'),
    'BILI': ('Bilibili Inc.', 'China', 'Entertainment'),
    '0700.HK': ('Tencent Holdings', 'China', 'Technology'),
    '9618.HK': ('JD.com (HK)', 'China', 'Technology'),
    '3690.HK': ('Meituan', 'China', 'Technology'),
    '1810.HK': ('Xiaomi Corporation', 'China', 'Technology'),
    '2318.HK': ('Ping An Insurance', 'China', 'Finance'),
    '0941.HK': ('China Mobile', 'China', 'Telecom'),
    '1398.HK': ('ICBC', 'China', 'Finance'),
    '3988.HK': ('Bank of China', 'China', 'Finance'),
    
    # ===== JAPAN STOCKS =====
    '7203.T': ('Toyota Motor Corp', 'Japan', 'Automotive'),
    '6758.T': ('Sony Group Corporation', 'Japan', 'Technology'),
    '9984.T': ('SoftBank Group', 'Japan', 'Technology'),
    '6861.T': ('Keyence Corporation', 'Japan', 'Technology'),
    '9432.T': ('NTT Corporation', 'Japan', 'Telecom'),
    '8306.T': ('MUFG Bank', 'Japan', 'Finance'),
    '7974.T': ('Nintendo Co Ltd', 'Japan', 'Entertainment'),
    '6501.T': ('Hitachi Ltd', 'Japan', 'Industrial'),
    '7267.T': ('Honda Motor Co', 'Japan', 'Automotive'),
    '4502.T': ('Takeda Pharmaceutical', 'Japan', 'Healthcare'),
    '6902.T': ('Denso Corporation', 'Japan', 'Automotive'),
    '8035.T': ('Tokyo Electron', 'Japan', 'Technology'),
    'TM': ('Toyota Motor (ADR)', 'Japan', 'Automotive'),
    'SONY': ('Sony Group (ADR)', 'Japan', 'Technology'),
    'HMC': ('Honda Motor (ADR)', 'Japan', 'Automotive'),
    
    # ===== SOUTH KOREA STOCKS =====
    '005930.KS': ('Samsung Electronics', 'Korea', 'Technology'),
    '000660.KS': ('SK Hynix', 'Korea', 'Technology'),
    '035420.KS': ('Naver Corporation', 'Korea', 'Technology'),
    '035720.KS': ('Kakao Corp', 'Korea', 'Technology'),
    '051910.KS': ('LG Chem', 'Korea', 'Industrial'),
    '006400.KS': ('Samsung SDI', 'Korea', 'Technology'),
    '003550.KS': ('LG Corp', 'Korea', 'Industrial'),
    '005380.KS': ('Hyundai Motor', 'Korea', 'Automotive'),
    
    # ===== TAIWAN STOCKS =====
    'TSM': ('Taiwan Semiconductor (ADR)', 'Taiwan', 'Technology'),
    '2330.TW': ('TSMC', 'Taiwan', 'Technology'),
    '2317.TW': ('Hon Hai Precision (Foxconn)', 'Taiwan', 'Technology'),
    '2454.TW': ('MediaTek Inc', 'Taiwan', 'Technology'),
    
    # ===== EUROPE STOCKS =====
    'ASML': ('ASML Holding NV', 'Europe', 'Technology'),
    'NVO': ('Novo Nordisk', 'Europe', 'Healthcare'),
    'SAP': ('SAP SE', 'Europe', 'Technology'),
    'TTE': ('TotalEnergies SE', 'Europe', 'Energy'),
    'SHEL': ('Shell PLC', 'Europe', 'Energy'),
    'AZN': ('AstraZeneca PLC', 'Europe', 'Healthcare'),
    'HSBC': ('HSBC Holdings', 'Europe', 'Finance'),
    'UL': ('Unilever PLC', 'Europe', 'Consumer'),
    'BP': ('BP PLC', 'Europe', 'Energy'),
    'GSK': ('GlaxoSmithKline', 'Europe', 'Healthcare'),
    'RIO': ('Rio Tinto Group', 'Europe', 'Industrial'),
    'BHP': ('BHP Group Limited', 'Europe', 'Industrial'),
    'DEO': ('Diageo PLC', 'Europe', 'Consumer'),
    'BTI': ('British American Tobacco', 'Europe', 'Consumer'),
    'VOD': ('Vodafone Group', 'Europe', 'Telecom'),
    'LVMUY': ('LVMH Moet Hennessy', 'Europe', 'Consumer'),
    'OR.PA': ('LOreal SA', 'Europe', 'Consumer'),
    'MC.PA': ('LVMH (Paris)', 'Europe', 'Consumer'),
    'SAN.PA': ('Sanofi SA', 'Europe', 'Healthcare'),
    'AIR.PA': ('Airbus SE', 'Europe', 'Aerospace'),
    'SIEGY': ('Siemens AG', 'Europe', 'Industrial'),
    'BAYN.DE': ('Bayer AG', 'Europe', 'Healthcare'),
    'BMW.DE': ('BMW AG', 'Europe', 'Automotive'),
    'VOW3.DE': ('Volkswagen AG', 'Europe', 'Automotive'),
    'MBG.DE': ('Mercedes-Benz Group', 'Europe', 'Automotive'),
    'SIE.DE': ('Siemens AG (DE)', 'Europe', 'Industrial'),
    
    # ===== OTHER ASIAN MARKETS =====
    'GRAB': ('Grab Holdings', 'Singapore', 'Technology'),
    'SE': ('Sea Limited', 'Singapore', 'Technology'),
    'CPALL.BK': ('CP All PCL', 'Thailand', 'Retail'),
    'PTT.BK': ('PTT PCL', 'Thailand', 'Energy'),
    'TLKM.JK': ('Telkom Indonesia', 'Indonesia', 'Telecom'),
    'BBCA.JK': ('Bank Central Asia', 'Indonesia', 'Finance'),
}

# ==================== TRADING GLOSSARY ====================
TRADING_GLOSSARY = {
    "Bull Market": "A market condition where prices are rising or expected to rise. Investors are optimistic.",
    "Bear Market": "A market condition where prices are falling or expected to fall. Investors are pessimistic.",
    "Stock": "A share of ownership in a company. When you buy a stock, you become a partial owner.",
    "Dividend": "A portion of a company's earnings paid to shareholders, usually quarterly.",
    "P/E Ratio": "Price-to-Earnings ratio. Stock price divided by earnings per share. Helps determine if a stock is overvalued or undervalued.",
    "Market Cap": "Market Capitalization. Total value of a company's shares = Stock Price × Number of Shares.",
    "Volume": "The number of shares traded during a specific time period.",
    "IPO": "Initial Public Offering. When a private company first sells shares to the public.",
    "Blue Chip": "Large, well-established companies with a history of reliable performance.",
    "RSI": "Relative Strength Index. A momentum indicator (0-100). Above 70 = overbought, below 30 = oversold.",
    "SMA": "Simple Moving Average. Average price over a specific period. Helps identify trends.",
    "EMA": "Exponential Moving Average. Like SMA but gives more weight to recent prices.",
    "Support Level": "Price level where a stock tends to stop falling due to buying pressure.",
    "Resistance Level": "Price level where a stock tends to stop rising due to selling pressure.",
    "Stop Loss": "An order to sell a stock when it reaches a certain price to limit losses.",
    "Take Profit": "An order to sell a stock when it reaches a certain price to lock in gains.",
    "Portfolio": "A collection of investments owned by an individual or institution.",
    "Diversification": "Spreading investments across different assets to reduce risk.",
    "Volatility": "How much a stock's price fluctuates. High volatility = higher risk but potentially higher returns.",
    "Liquidity": "How easily an asset can be bought or sold without affecting its price.",
    "Broker": "A person or firm that executes buy/sell orders on behalf of investors.",
    "Bid Price": "The highest price a buyer is willing to pay for a stock.",
    "Ask Price": "The lowest price a seller is willing to accept for a stock.",
    "Spread": "The difference between the bid and ask price.",
    "Going Long": "Buying a stock expecting its price to rise.",
    "Short Selling": "Borrowing and selling a stock expecting its price to fall, then buying it back cheaper.",
    "Candlestick": "A chart showing open, high, low, and close prices for a time period.",
    "Green Candle": "Price closed higher than it opened (bullish).",
    "Red Candle": "Price closed lower than it opened (bearish).",
    "52-Week High/Low": "The highest and lowest prices a stock has traded at in the past year.",
    "ETF": "Exchange-Traded Fund. A basket of stocks that trades like a single stock.",
    "Index": "A benchmark measuring market performance (e.g., S&P 500, NIFTY 50).",
    "NIFTY 50": "India's benchmark index of 50 large-cap companies on NSE.",
    "S&P 500": "US benchmark index of 500 large companies.",
    "Sensex": "India's benchmark index of 30 companies on BSE.",
    "NASDAQ": "US stock exchange focused on technology companies.",
    "NSE": "National Stock Exchange of India.",
    "BSE": "Bombay Stock Exchange, India's oldest stock exchange.",
}

# ==================== MODEL LOADING ====================
model_save_dir = "financial_sentiment_model"
weights_path = os.path.join(model_save_dir, "bert_weights.h5")

@st.cache_resource
def load_model():
    # Use pre-trained FinBERT directly - it's already trained on financial sentiment
    # DO NOT load custom weights as they are undertrained
    model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

model, tokenizer = load_model()

# FinBERT's label mapping: 0=positive, 1=negative, 2=neutral
# This is the correct mapping for ProsusAI/finbert
label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}

# ==================== HELPER FUNCTIONS ====================
import requests

# Set up a robust session for yfinance to bypass cloud blocking
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    # Using a common browser User-Agent that is less likely to be flagged
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    })
    return session

# Initialize Finnhub Client
@st.cache_resource
def get_finnhub_client():
    # Hardcoded key as per user request to "make it work"
    return finnhub.Client(api_key="d62s9vhr01qnpu82gau0d62s9vhr01qnpu82gaug")

finnhub_client = get_finnhub_client()

def predict_sentiment(text):
    encodings = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    logits = model(encodings.data)[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    # Get probabilities for each class
    positive_prob = probs[0]  # index 0 = positive
    negative_prob = probs[1]  # index 1 = negative
    neutral_prob = probs[2]   # index 2 = neutral
    
    # Smart neutral detection logic:
    # 1. If positive is clearly highest (>55% and significantly more than negative), return Positive
    # 2. If negative is clearly highest (>55% and significantly more than positive), return Negative
    # 3. If neutral has highest raw probability, return Neutral
    # 4. If no class is confident and probabilities are close together, return Neutral
    # 5. Otherwise return highest probability
    
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    
    # Check if neutral has highest probability
    if neutral_prob >= positive_prob and neutral_prob >= negative_prob:
        sentiment = "Neutral"
    # Clear positive signal
    elif positive_prob > 0.55 and positive_prob > negative_prob + 0.15:
        sentiment = "Positive"
    # Clear negative signal
    elif negative_prob > 0.55 and negative_prob > positive_prob + 0.15:
        sentiment = "Negative"
    # If all probabilities are close (no clear winner), it's neutral
    elif max_prob < 0.45 or (abs(positive_prob - negative_prob) < 0.1 and neutral_prob > 0.2):
        sentiment = "Neutral"
    # Default to highest probability
    else:
        sentiment = label_map[max_idx]
    
    return sentiment, {"Positive": round(float(positive_prob), 4), "Negative": round(float(negative_prob), 4), "Neutral": round(float(neutral_prob), 4)}

def search_companies(query):
    """Search for companies and return matching stocks"""
    if not query or len(query) < 2:
        return []
    
    query_lower = query.lower()
    matches = []
    
    for ticker, (name, region, sector) in GLOBAL_STOCKS.items():
        if query_lower in ticker.lower() or query_lower in name.lower():
            matches.append((ticker, name, region, sector))
    
    # Try yfinance if no matches found
    if not matches:
        try:
            stock = yf.Ticker(query.upper())
            info = stock.info
            if info.get('longName'):
                matches.append((query.upper(), info.get('longName', query.upper()), 'Other', 'Unknown'))
        except:
            pass
    
    return matches[:15]

def get_stock_data(ticker, period="1mo"):
    """
    ULTRA-ROBUST FETCHING (Professional Grade):
    Strategy 1: Finnhub API (Direct API connection, won't be blocked)
    Strategy 2: YahooQuery (Internal API)
    Strategy 3: yFinance History
    Strategy 4: yFinance Download
    """
    try:
        # --- STRATEGY 1: Finnhub API (Fast & Reliable) ---
        try:
            # Map period to days for Finnhub
            period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825}
            days = period_map.get(period, 30)
            end_ts = int(time.time())
            start_ts = end_ts - (days * 24 * 60 * 60)
            
            # Use 'D' for daily candles
            res = finnhub_client.stock_candles(ticker, 'D', start_ts, end_ts)
            
            if res.get('s') == 'ok':
                df = pd.DataFrame({
                    'Open': res['o'],
                    'High': res['h'],
                    'Low': res['l'],
                    'Close': res['c'],
                    'Volume': res['v'],
                }, index=pd.to_datetime(res['t'], unit='s'))
                
                # Get basic info from our DB or Finnhub profile
                info = {}
                try:
                    profile = finnhub_client.company_profile2(symbol=ticker)
                    if profile:
                        info = {
                            'longName': profile.get('name', ticker),
                            'sector': profile.get('finnhubIndustry', 'N/A'),
                            'currency': profile.get('currency', 'USD'),
                            'marketCap': profile.get('marketCapitalization'),
                        }
                except:
                    pass
                
                # Fill fallback if info fetch failed
                if not info or info.get('longName') == ticker:
                    if ticker in GLOBAL_STOCKS:
                        info['longName'] = GLOBAL_STOCKS[ticker][0]
                
                return df, info
        except Exception as e:
            print(f"DEBUG: Finnhub failed for {ticker}: {e}")

        # --- FALLBACK STRATEGIES (Yahoo based) ---
        session = get_yf_session()
        
        # --- STRATEGY 2: yahooquery ---
        try:
            yq_ticker = Ticker(ticker, session=session, retry=2)
            yq_hist = yq_ticker.history(period=period)
            if isinstance(yq_hist, pd.DataFrame) and not yq_hist.empty:
                if isinstance(yq_hist.index, pd.MultiIndex):
                    yq_hist = yq_hist.xs(ticker)
                
                info = {}
                try:
                    price = yq_ticker.price.get(ticker, {})
                    info = { 'longName': price.get('longName', ticker), 'currency': price.get('currency', 'USD') }
                except: pass
                return yq_hist, info
        except: pass

        # --- STRATEGY 3 & 4: yfinance ---
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period=period)
        if hist is not None and not hist.empty:
            return hist, {'longName': ticker}

        hist = yf.download(ticker, period=period, progress=False, session=session)
        if hist is not None and not hist.empty:
            return hist, {'longName': ticker}

        return None, None
    except Exception as e:
        print(f"DEBUG: Critical failure for {ticker}: {e}")
        return None, None

def calculate_technical_indicators(df):
    if df is None or len(df) < 5:
        return df
    df = df.copy()
    if len(df) >= 20:
        df['SMA20'] = df['Close'].rolling(window=20).mean()
    if len(df) >= 50:
        df['SMA50'] = df['Close'].rolling(window=min(50, len(df))).mean()
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

def get_recommendation(df, info):
    if df is None or len(df) < 5:
        return "HOLD", 50, "Insufficient data"
    score = 50
    reasons = []
    current_price = df['Close'].iloc[-1]
    if len(df) >= 5:
        momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
        if momentum > 5:
            score += 20
            reasons.append(f"Strong momentum (+{momentum:.1f}%)")
        elif momentum < -5:
            score -= 20
            reasons.append(f"Weak momentum ({momentum:.1f}%)")
    if 'SMA20' in df.columns and not pd.isna(df['SMA20'].iloc[-1]):
        if current_price > df['SMA20'].iloc[-1]:
            score += 15
            reasons.append("Above 20-day MA")
        else:
            score -= 15
            reasons.append("Below 20-day MA")
    if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            score += 20
            reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 20
            reasons.append("RSI overbought")
    if score >= 65:
        return "BUY", score, " | ".join(reasons[:3]) if reasons else "Positive signals"
    elif score <= 35:
        return "SELL", score, " | ".join(reasons[:3]) if reasons else "Negative signals"
    return "HOLD", score, " | ".join(reasons[:3]) if reasons else "Mixed signals"

def create_stock_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#00d4aa', decreasing_line_color='#ff6b6b'
    ))
    if 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='#ffd93d', width=1.5)))
    if 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA 50', line=dict(color='#00a8e8', width=1.5)))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'{ticker} Stock Price', font=dict(size=20, color='white')),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title='Price'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02), height=500, margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def get_region_badge(region):
    region_classes = {'US': 'region-us', 'India': 'region-india', 'China': 'region-asia', 'Japan': 'region-asia', 
                      'Korea': 'region-asia', 'Taiwan': 'region-asia', 'Singapore': 'region-asia', 
                      'Thailand': 'region-asia', 'Indonesia': 'region-asia', 'Europe': 'region-europe'}
    return region_classes.get(region, 'region-us')

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3rem;">🔮</span>
        <h2 style="color: #00d4aa; margin: 0.5rem 0;">FINSIGHT AI</h2>
        <p style="color: #a0aec0; font-size: 0.9rem;">AI-Powered Financial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["📊 Stock Analysis", "📰 Sentiment Analysis", "📚 Learn Trading", "ℹ️ About"],
        label_visibility="collapsed"
    )

# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-title">FINSIGHT AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Stock Analysis & Financial Sentiment Intelligence</p>', unsafe_allow_html=True)

if page == "📊 Stock Analysis":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Show coverage info
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color: white; margin-bottom: 1rem;">🔍 Search Company</h3>
        <p style="color: #a0aec0;">Search from <strong style="color: #00d4aa;">{len(GLOBAL_STOCKS)}+ stocks</strong> across global markets</p>
        <div style="margin-top: 0.8rem;">
            <span class="region-badge region-us">🇺🇸 US</span>
            <span class="region-badge region-india">🇮🇳 India</span>
            <span class="region-badge region-asia">🇨🇳 China</span>
            <span class="region-badge region-asia">🇯🇵 Japan</span>
            <span class="region-badge region-asia">🇰🇷 Korea</span>
            <span class="region-badge region-europe">🇪🇺 Europe</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input("Search", placeholder="Type company name (e.g., Apple, Reliance, Tata, Samsung...)", label_visibility="collapsed")
    
    if search_query:
        matches = search_companies(search_query)
        
        if matches:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #00d4aa;">Found {len(matches)} matching stocks:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            options = [f"{ticker} - {name} ({region})" for ticker, name, region, sector in matches]
            selected = st.selectbox("Select a stock:", options, label_visibility="collapsed")
            
            if selected:
                selected_ticker = selected.split(" - ")[0]
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    period_options = {"1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
                    period_label = st.selectbox("Period", list(period_options.keys()), index=2)
                    period = period_options[period_label]
                
                if st.button("📈 Analyze Stock", use_container_width=True):
                    with st.spinner(f"Fetching data for {selected_ticker}..."):
                        hist, info = get_stock_data(selected_ticker, period)
                    
                    if hist is not None and len(hist) > 0:
                        hist = calculate_technical_indicators(hist)
                        recommendation, score, reason = get_recommendation(hist, info)
                        
                        company_name = info.get('longName', selected_ticker) if info else selected_ticker
                        current_price = hist['Close'].iloc[-1]
                        prev_close = info.get('previousClose', hist['Close'].iloc[0]) if info else hist['Close'].iloc[0]
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close else 0
                        currency = info.get('currency', 'USD') if info else 'USD'
                        
                        change_class = 'positive' if change >= 0 else 'negative'
                        change_arrow = '▲' if change >= 0 else '▼'
                        
                        sector = info.get('sector', 'N/A') if info else 'N/A'
                        industry = info.get('industry', 'N/A') if info else 'N/A'
                        
                        st.markdown(f"""
                        <div class="glass-card">
                            <h2 style="color: white; margin-bottom: 0.5rem;">{company_name}</h2>
                            <p style="color: #a0aec0; margin-bottom: 1rem;">{sector} | {industry}</p>
                            <div style="display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;">
                                <span style="font-size: 2.5rem; font-weight: 700; color: white;">{currency} {current_price:.2f}</span>
                                <span class="{change_class}" style="font-size: 1.2rem; font-weight: 600;">
                                    {change_arrow} {abs(change):.2f} ({change_pct:+.2f}%)
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            mc = info.get('marketCap', 0) if info else 0
                            mc_str = f"${mc/1e9:.1f}B" if mc else "N/A"
                            st.metric("Market Cap", mc_str)
                        with col2:
                            vol = info.get('volume', 0) if info else 0
                            vol_str = f"{vol/1e6:.1f}M" if vol else "N/A"
                            st.metric("Volume", vol_str)
                        with col3:
                            h52 = info.get('fiftyTwoWeekHigh', 0) if info else 0
                            st.metric("52W High", f"${h52:.2f}" if h52 else "N/A")
                        with col4:
                            l52 = info.get('fiftyTwoWeekLow', 0) if info else 0
                            st.metric("52W Low", f"${l52:.2f}" if l52 else "N/A")
                        with col5:
                            pe = info.get('trailingPE', 0) if info else 0
                            st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        fig = create_stock_chart(hist, selected_ticker)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        rec_class = recommendation.lower()
                        st.markdown(f"""
                        <div class="glass-card" style="text-align: center;">
                            <h3 style="color: #a0aec0; margin-bottom: 1rem;">🤖 AI RECOMMENDATION</h3>
                            <div class="recommendation {rec_class}">{recommendation}</div>
                            <p style="color: #a0aec0; margin-top: 1rem;">Confidence: <strong style="color: #00d4aa;">{score}%</strong></p>
                            <p style="color: #6b7280; font-size: 0.85rem;">{reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Could not fetch data for {selected_ticker}. Please try another stock.")
        else:
            st.warning(f"No stocks found for '{search_query}'. Try a different term or enter the exact ticker symbol.")

elif page == "📰 Sentiment Analysis":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">📝 Financial News Sentiment Analyzer</h3>
        <p style="color: #a0aec0;">Analyze financial news headlines using our FinBERT AI model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area("Enter financial news:", placeholder="e.g., Apple reports record quarterly revenue...", height=150, label_visibility="collapsed")
    
    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                sentiment, probs = predict_sentiment(user_input)
            
            colors = {"Positive": "#00d4aa", "Neutral": "#ffd93d", "Negative": "#ff6b6b"}
            emoji = {"Positive": "📈", "Neutral": "➡️", "Negative": "📉"}
            
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <span style="font-size: 4rem;">{emoji[sentiment]}</span>
                <h2 style="color: {colors[sentiment]}; margin: 1rem 0;">{sentiment}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            for col, (label, color) in zip([col1, col2, col3], [("Positive", "#00d4aa"), ("Neutral", "#ffd93d"), ("Negative", "#ff6b6b")]):
                with col:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <p style="color: #a0aec0;">{label}</p>
                        <p style="font-size: 1.8rem; font-weight: 700; color: {color};">{probs[label]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")

elif page == "📚 Learn Trading":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white;">📚 Trading Glossary for Beginners</h3>
        <p style="color: #a0aec0;">Essential trading terms every investor should know. Perfect for newcomers!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search filter
    search_term = st.text_input("🔍 Search terms...", placeholder="Type to filter terms...", label_visibility="collapsed")
    
    filtered_terms = {k: v for k, v in TRADING_GLOSSARY.items() if not search_term or search_term.lower() in k.lower() or search_term.lower() in v.lower()}
    
    st.markdown(f"<p style='color: #a0aec0; margin: 1rem 0;'>Showing {len(filtered_terms)} of {len(TRADING_GLOSSARY)} terms</p>", unsafe_allow_html=True)
    
    # Display terms in columns
    terms_list = list(filtered_terms.items())
    col1, col2 = st.columns(2)
    
    for i, (term, definition) in enumerate(terms_list):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="term-card">
                <div class="term-title">{term}</div>
                <div class="term-desc">{definition}</div>
            </div>
            """, unsafe_allow_html=True)

else:  # About
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color: white;">About FINSIGHT AI</h3>
        <p style="color: #a0aec0; line-height: 1.8;">
            FINSIGHT AI is a cutting-edge financial intelligence platform combining real-time market data with AI-powered analysis.
        </p>
    </div>
    
    <div class="glass-card">
        <h4 style="color: #00d4aa;">Features</h4>
        <ul style="color: #a0aec0; line-height: 2;">
            <li><strong>{len(GLOBAL_STOCKS)}+ Global Stocks</strong> - US, India, China, Japan, Korea, Europe</li>
            <li><strong>Company Search</strong> - Find any stock by name or ticker</li>
            <li><strong>Interactive Charts</strong> - Candlestick charts with technical indicators</li>
            <li><strong>AI Recommendations</strong> - Buy/Hold/Sell based on technical analysis</li>
            <li><strong>Sentiment Analysis</strong> - FinBERT-powered news analysis</li>
            <li><strong>Trading Glossary</strong> - Learn essential trading terms</li>
        </ul>
    </div>
    
    <div class="glass-card">
        <h4 style="color: #ffd93d;">Disclaimer</h4>
        <p style="color: #a0aec0; font-size: 0.9rem;">
            This tool is for educational purposes only. Always consult a financial advisor. Past performance doesn't guarantee future results.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #6b7280; font-size: 0.85rem;">
    <p>Built with Streamlit & FinBERT | FINSIGHT AI</p>
</div>
""", unsafe_allow_html=True)
