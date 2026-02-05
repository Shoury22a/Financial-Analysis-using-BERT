---
title: FINSIGHT AI
emoji: 🔮
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# FINSIGHT AI 🔮

AI-Powered Financial Sentiment Analysis & Stock Intelligence Platform

## Features

- **📊 Stock Analysis** - 200+ global stocks (US, India, China, Japan, Korea, Europe)
- **📰 Sentiment Analysis** - FinBERT-powered financial news analysis
- **📚 Trading Glossary** - Essential terms for beginners
- **🤖 AI Recommendations** - Buy/Hold/Sell based on technical analysis

## Tech Stack

- **Backend**: FastAPI + TensorFlow
- **NLP Model**: FinBERT (ProsusAI/finbert)
- **Data**: yfinance for real-time stock data
- **Deployment**: Docker + Render.com

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Swagger API docs |
| `/api/sentiment` | POST | Analyze text sentiment |
| `/api/stocks/search` | GET | Search stocks by name |
| `/api/stocks/{ticker}` | GET | Get stock data & charts |
| `/api/glossary` | GET | Trading terms |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api:app --host 0.0.0.0 --port 8000

# Run Streamlit UI
streamlit run prediction.py
```

## Docker

```bash
docker build -t finsight-ai .
docker run -p 8000:8000 finsight-ai
```

## Deploy to Render

1. Push to GitHub
2. Connect repo on render.com
3. Auto-deploys via `render.yaml`

## License

MIT
