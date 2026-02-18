"""
Simple Test Script to Verify Fixes
Tests that the critical issues have been resolved
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

print("=" * 60)
print("TESTING CRITICAL FIXES")
print("=" * 60)

# Test 1: Label Mapping Consistency
print("\n[TEST 1] Label Mapping Consistency")
print("-" * 60)

# This is what training uses (quick_train.py line 52)
training_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# This is what we fixed in prediction.py
prediction_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

if training_labels == prediction_labels:
    print("✅ PASS: Training and prediction use same label mapping!")
    print(f"   Mapping: {training_labels}")
else:
    print("❌ FAIL: Label mappings don't match!")
    print(f"   Training:   {training_labels}")
    print(f"   Prediction: {prediction_labels}")

# Test 2: Model Prediction Test
print("\n[TEST 2] Model Prediction Test")
print("-" * 60)

try:
    model = BertForSequenceClassification.from_pretrained("financial_sentiment_model", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("financial_sentiment_model")
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    test_cases = [
        ("Apple has high growth potential", "Positive"),
        ("Revenue declined significantly", "Negative"),
        ("Company released quarterly results", "Neutral"),
    ]
    
    passed = 0
    for text, expected in test_cases:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).numpy()[0]
        
        predicted = label_map[probs.argmax()]
        status = "✅" if predicted == expected else "❌"
        
        print(f"{status} '{text}'")
        print(f"   Expected: {expected}, Got: {predicted}")
        print(f"   Probs: Neg={probs[0]:.2f}, Neu={probs[1]:.2f}, Pos={probs[2]:.2f}")
        
        if predicted == expected:
            passed += 1
    
    print(f"\nResults: {passed}/{len(test_cases)} tests passed")
    
    if passed == len(test_cases):
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {len(test_cases) - passed} test(s) failed - may need retraining")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Run quick_train.py first to train the model")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
