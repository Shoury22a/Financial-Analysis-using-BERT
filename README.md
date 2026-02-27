# FINSIGHT AI 🔮

**AI-Powered Financial Sentiment Analysis & Stock Intelligence**

Built with fine-tuned **FinBERT** (BERT for Finance), Streamlit, yfinance, and Plotly.

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ✨ Features

| Page | Description |
|------|-------------|
| 🔮 Sentiment Analysis | Analyze any financial text — Positive / Negative / Neutral |
| 📈 Stock Explorer | Live candlestick charts, key metrics for global stocks |
| 📋 Batch Analysis | Classify multiple headlines at once, download CSV |
| 📚 Glossary | Key financial terms explained |

---

## 🤖 Model

- **Base**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **Fine-tuned** on a custom financial news dataset (3-class: Positive / Negative / Neutral)
- **Framework**: PyTorch + HuggingFace Transformers

> ⚠️ The trained model weights (`financial_sentiment_model/`) are excluded from this repo (too large). Run `python train_model.py` to retrain, or download the base FinBERT weights automatically on first run.

---

## 📦 Tech Stack

- `streamlit` · `transformers` · `torch`
- `yfinance` · `plotly` · `pandas`
- `finnhub-python` · `scikit-learn`

---

## 📄 License

MIT
