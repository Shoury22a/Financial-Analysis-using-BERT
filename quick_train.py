# -*- coding: utf-8 -*-
"""
QUICK TRAINING - Just 2 epochs to fix basic predictions
"""

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Check GPU
print("Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {len(gpus)} device(s)")
else:
    print("Using CPU")

print("\n[1/4] Creating diverse training dataset with SYNONYMS...")

# POSITIVE examples - diverse synonyms and sentence structures
positive = [
    # Growth & Increase (synonyms)
    "profit increased", "profit grew", "profit rose", "profit surged", "profit jumped",
    "revenue increased", "revenue grew", "revenue climbed", "revenue soared", "revenue expanded",
    "earnings increased", "earnings rose", "earnings improved", "earnings advanced", "earnings gained",
    "sales went up", "sales grew", "sales increased", "sales rose", "sales climbed",
    
    # Performance (variations)
    "beat expectations", "exceeded expectations", "surpassed estimates", "outperformed forecasts",
    "strong performance", "excellent results", "outstanding quarter", "impressive growth",
    "record profits", "record earnings", "all-time high", "best quarter ever",
    
    # Market Response
    "stock surged", "stock rallied", "stock jumped", "stock soared", "stock climbed",
    "share price increased", "shares rose", "stock gained ground", "price went up",
    
    # Positive Operations
    "expansion successful", "growth accelerated", "momentum building", "outlook positive",
    "dividend raised", "dividend increased", "payout grew", "shareholder returns up",
    "cash flow improved", "liquidity strong", "margins expanded", "margins improved",
    "demand strong", "demand robust", "demand growing", "orders increasing",
    
    # Sentiment & Outlook
    "optimistic outlook", "positive guidance", "bullish forecast", "confident projections",
    "rating upgraded", "analyst upgrade", "price target raised", "buy recommendation",
    "investor confidence", "market optimism", "strong fundamentals", "healthy balance sheet",
    
    # Competitive Advantage
    "market share gained", "competitive edge", "industry leader", "outperforming peers",
    "innovation success", "product launch successful", "new contract won", "strategic partnership"
] * 10  # 500+ examples

# NEGATIVE examples - diverse synonyms and sentence structures  
negative = [
    # Decline & Decrease (synonyms)
    "profit declined", "profit fell", "profit dropped", "profit plunged", "profit decreased",
    "revenue declined", "revenue fell", "revenue dropped", "revenue slumped", "revenue contracted",
    "earnings declined", "earnings fell", "earnings dropped", "earnings disappointed", "earnings missed",
    "sales went down", "sales fell", "sales dropped", "sales declined", "sales weakened",
    
    # Performance (variations)
    "missed expectations", "below estimates", "disappointed investors", "underperformed forecasts",
    "weak performance", "poor results", "disappointing quarter", "struggles continue",
    "losses reported", "operating loss", "net loss", "unprofitable quarter",
    
    # Market Response
    "stock plunged", "stock crashed", "stock dropped", "stock fell", "stock declined",
    "share price decreased", "shares fell", "stock lost ground", "price went down",
    
    # Negative Operations
    "growth slowed", "growth stalled", "momentum fading", "outlook negative",
    "dividend cut", "dividend reduced", "payout decreased", "dividend suspended",
    "cash flow weak", "liquidity concerns", "margins compressed", "margins shrinking",
    "demand falling", "demand weak", "demand declining", "orders decreasing",
    
    # Sentiment & Outlook
    "pessimistic outlook", "negative guidance", "bearish forecast", "concerning projections",
    "rating downgraded", "analyst downgrade", "price target lowered", "sell recommendation",
    "investor concern", "market pessimism", "weak fundamentals", "debt concerns",
    
    # Competitive Challenges
    "market share lost", "losing ground", "competition intensifying", "underperforming sector",
    "product recall", "legal troubles", "regulatory issues", "management shakeup",
    "layoffs announced", "restructuring needed", "cost cutting", "bankruptcy risk"
] * 10  # 500+ examples

# NEUTRAL examples - factual statements without sentiment
neutral = [
    # Announcements
    "results announced", "earnings released", "report published", "data disclosed",
    "meeting scheduled", "conference planned", "presentation set", "call scheduled",
    "announcement made", "statement released", "update provided", "filing submitted",
    
    # Changes & Transitions
    "CEO appointed", "board member joined", "executive hired", "leadership change",
    "policy updated", "strategy revised", "process modified", "system upgraded",
    
    # Status & Operations
    "operations continue", "business as usual", "steady performance", "stable operations",
    "unchanged revenue", "flat growth", "status quo maintained", "guidance maintained",
    "trading halted", "trading resumed", "stock split announced", "merger proposed",
    
    # Factual Information
    "company based in", "founded in", "operates in", "headquartered in",
    "market cap is", "employee count", "revenue reported", "quarter ended",
    "fiscal year", "financial statement", "balance sheet", "income statement",
    
    # Neutral Events
    "partnership formed", "acquisition completed", "deal finalized", "agreement signed",
    "investigation ongoing", "review in progress", "audit scheduled", "compliance check",
    "dividend date set", "earnings date scheduled", "AGM planned", "vote pending"
] * 10  # 400+ examples

# Create balanced dataset
df = pd.DataFrame({
    'text': positive + negative + neutral,
    'label': [2]*len(positive) + [0]*len(negative) + [1]*len(neutral)  # 0=Neg, 1=Neu, 2=Pos
})

print(f"Total: {len(df)} samples")
print(f"  - Positive: {len(positive)} (includes 'growth', 'increased', 'surged', etc.)")
print(f"  - Negative: {len(negative)} (includes 'decline', 'fell', 'dropped', etc.)")
print(f"  - Neutral: {len(neutral)} (factual statements)")
print(f"[OK] Model will learn SYNONYMS and variations!")

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(),
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\n[2/4] Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=64, return_tensors="tf")
val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=64, return_tensors="tf")

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), tf.constant(train_labels))).shuffle(500).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), tf.constant(val_labels))).batch(32)

print("\n[3/4] Training (4 epochs for better synonym learning)...")
tf.keras.backend.clear_session()
model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=4, verbose=1)

print("\n[4/4] Saving...")
os.makedirs("financial_sentiment_model", exist_ok=True)
model.save_pretrained("financial_sentiment_model")
tokenizer.save_pretrained("financial_sentiment_model")

import json
with open("financial_sentiment_model/label_map.json", "w") as f:
    json.dump({"0": "Negative", "1": "Neutral", "2": "Positive"}, f)

print("\n" + "="*60)
print("DONE! Model saved to financial_sentiment_model/")
print("="*60)
