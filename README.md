---
title: FINSIGHT AI
emoji: 🔮
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.31.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# FINSIGHT AI 🔮

**AI-Powered Financial Sentiment Analysis & Stock Intelligence**

Built with fine-tuned **FinBERT** (BERT for Finance), Streamlit, yfinance, and Plotly.

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

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Tech Stack

`streamlit` · `transformers` · `torch` · `yfinance` · `plotly` · `pandas` · `scikit-learn`
