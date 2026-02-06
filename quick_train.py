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

print("\n[1/4] Creating basic training dataset...")

# Simple, clear examples
positive = [
    "profit increased", "revenue grew", "earnings beat expectations", "stock surged",
    "sales up", "growth accelerated", "performance exceeded forecasts", "dividend raised",
    "strong quarter", "record profits", "market share gained", "expansion successful",
    "positive outlook", "exceeded targets", "investor optimism", "rating upgraded",
   "stock rallied", "cash flow improved", "margins expanded", "demand strong"
] * 15  # 300 examples

negative = [
    "profit declined",  "revenue fell", "earnings missed", "stock plunged",
    "sales down", "growth slowed", "performance disappointed", "dividend cut",
    "weak quarter", "losses reported", "market share lost", "layoffs announced",
    "negative outlook", "missed targets", "investor concern", "rating downgraded",
    "stock dropped", "cash flow weak", "margins compressed", "demand falling"
] * 15  # 300 examples

neutral = [
    "results announced", "meeting scheduled", "board appointed", "report released",
    "unchanged revenue", "flat growth", "guidance maintained", "status quo",
    "announcement made", "data published", "event planned", "review ongoing",
    "process continues", "activity normal", "operations stable", "position held"
] * 15  # 240 examples

# Create balanced dataset
df = pd.DataFrame({
    'text': positive + negative + neutral,
    'label': [2]*len(positive) + [0]*len(negative) + [1]*len(neutral)  # 0=Neg, 1=Neu, 2=Pos
})

print(f"Total: {len(df)} samples ({len(positive)} pos, {len(negative)} neg, {len(neutral)} neu)")

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

print("\n[3/4] Training (2 epochs - FAST!)...")
tf.keras.backend.clear_session()
model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=2, verbose=1)

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
