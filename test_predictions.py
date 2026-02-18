"""
FINSIGHT AI - Comprehensive Sentiment Prediction Test
=====================================================
Tests the fine-tuned model on positive, negative, and neutral statements.
Loads model and label mapping dynamically from config (NO HARDCODING).
"""

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import os
import json


def load_model_and_labels():
    """Load the fine-tuned model and read label map from config.json."""
    model_path = "financial_sentiment_model"
    if not os.path.exists(model_path):
        model_path = "ProsusAI/finbert"
        print(f"  -> Fine-tuned model not found, using base: {model_path}")

    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Read label map from config.json (NO HARDCODING)
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        id2label = config.get("id2label", {})
        label_map = {int(k): v.capitalize() for k, v in id2label.items()}
    else:
        label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}

    print(f"  -> Model path: {model_path}")
    print(f"  -> Label mapping: {label_map}")
    return model, tokenizer, label_map


def predict_sentiment(text, model, tokenizer, label_map):
    """Predict sentiment using the model with pure argmax (no hardcoding)."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).numpy()[0]

    predicted_idx = int(np.argmax(probs))
    sentiment = label_map[predicted_idx]

    # Build probability dict using label names from config
    prob_dict = {}
    for idx, label_name in label_map.items():
        prob_dict[label_name] = round(float(probs[idx]), 4)

    return sentiment, prob_dict


def run_tests():
    """Run comprehensive sentiment tests."""
    print("=" * 70)
    print("FINSIGHT AI - SENTIMENT PREDICTION TEST")
    print("=" * 70)

    print("\nLoading model...")
    model, tokenizer, label_map = load_model_and_labels()
    model.eval()

    # Test cases organized by expected sentiment
    test_suites = {
        "Positive": [
            "The company reported a strong quarterly profit exceeding analyst expectations",
            "Revenue grew by 25% year-over-year, driven by higher product demand",
            "The stock price surged after the successful product launch",
            "Dividend payments were increased for the third consecutive year",
            "Analysts upgraded the stock rating to buy due to strong fundamentals",
            "Apple has a high growth potential",
            "Record-breaking earnings drove the stock to an all-time high",
            "The firm secured a major contract worth billions of dollars",
            "Strong demand for new products boosted quarterly revenue significantly",
            "Investor confidence soared following the impressive earnings report",
        ],
        "Negative": [
            "The company posted a significant quarterly loss due to declining sales",
            "Revenue fell short of market expectations, causing the stock to drop",
            "Rising debt levels are creating financial instability",
            "The stock price plunged after regulatory concerns emerged",
            "Credit rating agencies downgraded the firm's outlook to negative",
            "Massive layoffs were announced as the company struggles financially",
            "Profit margins collapsed amid rising costs and supply chain issues",
            "The CEO resigned amid allegations of financial misconduct",
            "Bankruptcy fears sent shares tumbling to record lows",
            "The company faces a major lawsuit that could cost billions",
        ],
        "Neutral": [
            "The company released its quarterly financial results today",
            "Revenue remained unchanged compared to last year",
            "Management announced a new board member appointment",
            "Shares closed flat in today's trading session",
            "Analysts are monitoring the upcoming earnings call",
            "The company will hold its annual general meeting next week",
            "Trading volume remained average during the session",
            "The firm published its annual sustainability report",
            "Board members discussed strategic options at the meeting",
            "The stock exchange updated its listing requirements",
        ],
    }

    total_correct = 0
    total_tests = 0
    results_by_class = {}

    for expected_sentiment, test_cases in test_suites.items():
        print(f"\n{'=' * 60}")
        print(f"  {expected_sentiment.upper()} TEST CASES")
        print(f"{'=' * 60}")

        class_correct = 0
        class_total = len(test_cases)

        for text in test_cases:
            sentiment, probs = predict_sentiment(text, model, tokenizer, label_map)
            is_correct = sentiment == expected_sentiment
            status = "[OK]  " if is_correct else "[FAIL]"

            if is_correct:
                class_correct += 1
                total_correct += 1
            total_tests += 1

            # Format probability string
            prob_parts = []
            for label_name in ["Positive", "Negative", "Neutral"]:
                if label_name in probs:
                    prob_parts.append(f"{label_name[:3]}: {probs[label_name]:.1%}")
            prob_str = ", ".join(prob_parts)

            print(f"\n{status} {text}")
            print(f"       -> {sentiment} | {prob_str}")

        accuracy = class_correct / class_total * 100
        results_by_class[expected_sentiment] = {
            "correct": class_correct,
            "total": class_total,
            "accuracy": accuracy,
        }
        print(f"\n  >> {expected_sentiment} Accuracy: {class_correct}/{class_total} ({accuracy:.0f}%)")

    # Final Summary
    overall_accuracy = total_correct / total_tests * 100
    print(f"\n\n{'=' * 70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for sentiment, result in results_by_class.items():
        bar_filled = int(result["accuracy"] / 5)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        print(f"  {sentiment:>8s}: {result['correct']:2d}/{result['total']:2d} ({result['accuracy']:5.1f}%) [{bar}]")

    print(f"\n  {'-' * 50}")
    print(f"  Overall:  {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
    print(f"{'=' * 70}")

    # Pass/Fail verdict
    if overall_accuracy >= 80:
        print("\n  [PASS] Model predictions are correct!")
    else:
        print("\n  [FAIL] Model needs more training or debugging.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_tests()
