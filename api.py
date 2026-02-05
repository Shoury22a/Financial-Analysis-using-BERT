"""
FINSIGHT AI - FastAPI Backend
Production API for sentiment analysis and stock data
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from stock_scraper import StockManager

# ==================== APP INITIALIZATION ====================
app = FastAPI(
    title="FINSIGHT AI API",
    description="AI-Powered Financial Sentiment Analysis & Stock Intelligence",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODEL LOADING ====================
print("Loading FinBERT model (PyTorch)...")
# PyTorch is much more memory efficient for inference on CPU
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model.eval() # Set to evaluation mode
label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
print("Model loaded successfully!")

# ==================== GLOBAL STOCK DATABASE ====================
print("Loading dynamic stock list...")
GLOBAL_STOCKS = StockManager.get_global_stocks()
print(f"Loaded {len(GLOBAL_STOCKS)} stocks dynamically.")


# ==================== PYDANTIC MODELS ====================
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: Dict[str, float]

class StockSearchRequest(BaseModel):
    query: str

class StockSearchResponse(BaseModel):
    stocks: List[Dict[str, str]]

class StockAnalysisRequest(BaseModel):
    ticker: str
    period: str = "1mo"

# ==================== HELPER FUNCTIONS ====================
def predict_sentiment(text: str) -> tuple:
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    
    positive_prob = probs[0]
    negative_prob = probs[1]
    neutral_prob = probs[2]
    
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    
    if neutral_prob >= positive_prob and neutral_prob >= negative_prob:
        sentiment = "Neutral"
    elif positive_prob > 0.55 and positive_prob > negative_prob + 0.15:
        sentiment = "Positive"
    elif negative_prob > 0.55 and negative_prob > positive_prob + 0.15:
        sentiment = "Negative"
    elif max_prob < 0.45 or (abs(positive_prob - negative_prob) < 0.1 and neutral_prob > 0.2):
        sentiment = "Neutral"
    else:
        sentiment = label_map[max_idx]
    
    return sentiment, {
        "Positive": round(float(positive_prob), 4),
        "Negative": round(float(negative_prob), 4),
        "Neutral": round(float(neutral_prob), 4)
    }

def get_stock_data(ticker: str, period: str = "1mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        return None, None

# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FINSIGHT AI API</title>
        <style>
            body { font-family: Arial, sans-serif; background: #0f0f23; color: white; padding: 50px; }
            h1 { color: #00d4aa; }
            a { color: #00a8e8; }
            .endpoint { background: rgba(255,255,255,0.05); padding: 15px; margin: 10px 0; border-radius: 10px; }
        </style>
    </head>
    <body>
        <h1>🔮 FINSIGHT AI API</h1>
        <p>AI-Powered Financial Sentiment Analysis & Stock Intelligence</p>
        <h3>Endpoints:</h3>
        <div class="endpoint"><strong>POST /api/sentiment</strong> - Analyze text sentiment</div>
        <div class="endpoint"><strong>GET /api/stocks/search?q=apple</strong> - Search stocks</div>
        <div class="endpoint"><strong>GET /api/stocks/{ticker}</strong> - Get stock data</div>
        <div class="endpoint"><strong>GET /api/health</strong> - Health check</div>
        <br>
        <a href="/docs">📚 Interactive API Documentation</a>
    </body>
    </html>
    """

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "FinBERT", "version": "1.0.0"}

@app.post("/api/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    sentiment, probs = predict_sentiment(request.text)
    return SentimentResponse(sentiment=sentiment, probabilities=probs)

@app.get("/api/stocks/search")
async def search_stocks(q: str):
    if not q or len(q) < 2:
        return {"stocks": []}
    
    query_lower = q.lower()
    matches = []
    
    for ticker, (name, region, sector) in GLOBAL_STOCKS.items():
        if query_lower in ticker.lower() or query_lower in name.lower():
            matches.append({
                "ticker": ticker,
                "name": name,
                "region": region,
                "sector": sector
            })
    
    return {"stocks": matches[:15]}

@app.get("/api/stocks/{ticker}")
async def get_stock(ticker: str, period: str = "1mo"):
    hist, info = get_stock_data(ticker, period)
    
    if hist is None or len(hist) == 0:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
    
    # Convert to JSON-serializable format
    price_data = []
    for idx, row in hist.iterrows():
        price_data.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"])
        })
    
    current_price = hist['Close'].iloc[-1]
    prev_close = info.get('previousClose', hist['Close'].iloc[0]) if info else hist['Close'].iloc[0]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0
    
    return {
        "ticker": ticker,
        "name": info.get('longName', ticker) if info else ticker,
        "sector": info.get('sector', 'N/A') if info else 'N/A',
        "currentPrice": round(current_price, 2),
        "change": round(change, 2),
        "changePercent": round(change_pct, 2),
        "marketCap": info.get('marketCap') if info else None,
        "volume": info.get('volume') if info else None,
        "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh') if info else None,
        "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow') if info else None,
        "peRatio": info.get('trailingPE') if info else None,
        "priceHistory": price_data
    }

@app.get("/api/glossary")
async def get_glossary():
    return {
        "terms": [
            {"term": "Bull Market", "definition": "A market condition where prices are rising or expected to rise."},
            {"term": "Bear Market", "definition": "A market condition where prices are falling or expected to fall."},
            {"term": "P/E Ratio", "definition": "Price-to-Earnings ratio. Helps determine if a stock is overvalued."},
            {"term": "Market Cap", "definition": "Total value of a company's shares = Stock Price × Number of Shares."},
            {"term": "RSI", "definition": "Relative Strength Index. Above 70 = overbought, below 30 = oversold."},
            {"term": "SMA", "definition": "Simple Moving Average. Average price over a specific period."},
            {"term": "Stop Loss", "definition": "An order to sell when a stock reaches a certain price to limit losses."},
            {"term": "Dividend", "definition": "A portion of a company's earnings paid to shareholders."},
            {"term": "IPO", "definition": "Initial Public Offering. When a company first sells shares to the public."},
            {"term": "ETF", "definition": "Exchange-Traded Fund. A basket of stocks that trades like a single stock."},
        ]
    }

# Run with: uvicorn api:app --host 0.0.0.0 --port 7860
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
