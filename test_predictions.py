"""
Quick test of the fixed API prediction logic with pre-trained FinBERT
"""
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

# Load pre-trained model (same as API)
print("Loading model...")
model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    encodings = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    logits = model(encodings.data)[0]
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    negative_prob = probs[0]
    neutral_prob = probs[1]
    positive_prob = probs[2]
    
    max_idx = np.argmax(probs)
    sentiment = label_map[max_idx]
    
    return sentiment, {
        "Positive": round(float(positive_prob), 4),
        "Negative": round(float(negative_prob), 4),
        "Neutral": round(float(neutral_prob), 4)
    }

# Test with your examples
print("\n" + "="*60)
print("POSITIVE TEST CASES")
print("="*60)
positive_tests = [
    "The company reported a strong quarterly profit exceeding analyst expectations",
    "Revenue grew by 25% year-over-year, driven by higher product demand",
    "The stock price surged after the successful product launch",
    "Dividend payments were increased for the third consecutive year",
    "Analysts upgraded the stock rating to buy due to strong fundamentals"
]

for text in positive_tests:
    sentiment, probs = predict_sentiment(text)
    status = "[OK]" if sentiment == "Positive" else "[FAIL]"
    print(f"\n{status} {text}")
    print(f"   -> {sentiment} | Pos: {probs['Positive']:.1%}, Neg: {probs['Negative']:.1%}, Neu: {probs['Neutral']:.1%}")

print("\n" + "="*60)
print("NEGATIVE TEST CASES")
print("="*60)
negative_tests = [
    "The company posted a significant quarterly loss due to declining sales",
    "Revenue fell short of market expectations, causing the stock to drop",
    "Rising debt levels are creating financial instability",
    "The stock price plunged after regulatory concerns emerged",
    "Credit rating agencies downgraded the firm's outlook to negative"
]

for text in negative_tests:
    sentiment, probs = predict_sentiment(text)
    status = "[OK]" if sentiment == "Negative" else "[FAIL]"
    print(f"\n{status} {text}")
    print(f"   -> {sentiment} | Pos: {probs['Positive']:.1%}, Neg: {probs['Negative']:.1%}, Neu: {probs['Neutral']:.1%}")

print("\n" + "="*60)
print("NEUTRAL TEST CASES")
print("="*60)
neutral_tests = [
    "The company released its quarterly financial results today",
    "Revenue remained unchanged compared to last year",
    "Management announced a new board member appointment",
    "Shares closed flat in today's trading session",
    "Analysts are monitoring the upcoming earnings call"
]

for text in neutral_tests:
    sentiment, probs = predict_sentiment(text)
    status = "[OK]" if sentiment == "Neutral" else "[FAIL]"
    print(f"\n{status} {text}")
    print(f"   -> {sentiment} | Pos: {probs['Positive']:.1%}, Neg: {probs['Negative']:.1%}, Neu: {probs['Neutral']:.1%}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
