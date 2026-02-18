from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F

print("Loading model from financial_sentiment_model...")
model = BertForSequenceClassification.from_pretrained("financial_sentiment_model")
tokenizer = BertTokenizer.from_pretrained("financial_sentiment_model")

print(f"Model Config ID2LABEL: {model.config.id2label}")

test_sentences = [
    # --- USER REPORTED FAILURE ---
    ("Apple has skyrocketed growth", "Positive"),

    # --- POSITIVE ---
    ("The company reported a strong quarterly profit exceeding analyst expectations.", "Positive"),
    ("Revenue grew by 25% year-over-year, driven by higher product demand.", "Positive"),
    ("The stock price surged after the successful product launch.", "Positive"),
    
    # --- NEGATIVE ---
    ("The company posted a significant quarterly loss due to declining sales.", "Negative"),
    ("Rising debt levels are creating financial instability.", "Negative"),
    ("Profit margins shrunk because of increasing raw material prices.", "Negative"),

    # --- NEUTRAL (Previously Problematic) ---
    ("Revenue remained unchanged compared to last year.", "Neutral"),
    ("Management announced a new board member appointment.", "Neutral"),
    ("The firm plans to expand operations into Asia next year.", "Neutral"),

    # --- IMPLICIT POSITIVE (User's New List) ---
    ("The company exceeded analyst revenue estimates for the third consecutive quarter.", "Positive"),
    ("Cash reserves increased while operating expenses remained stable.", "Positive"),
    
    # --- IMPLICIT NEGATIVE ---
    ("Operating expenses rose faster than total revenue.", "Negative"),
    ("Inventory levels continued to accumulate in warehouses.", "Negative"),

    # --- TRUE NEUTRAL ---
    ("The board meeting is scheduled for next Monday.", "Neutral"),
    ("The company operates in the renewable energy sector.", "Neutral"),

    # --- MIXED/COMPLEX ---
    ("Revenue increased, while capital expenditure also rose significantly.", "Neutral"), # Offsetting
    ("Sales volumes improved even as pricing pressure persisted.", "Positive"), # Sales up usually wins
]

# Fixed label mapping to match training
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("\n--- Testing Current Mapping ---")
for text, expected in test_sentences:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
        pred_idx = probs.argmax()
        pred_label = label_map.get(pred_idx, "Unknown")
        
        print(f"Text: {text[:30]}...")
        print(f"Expected: {expected}")
        print(f"Predicted Index: {pred_idx}")
        print(f"Predicted Label (Current Map): {pred_label}")
        print(f"Probabilities: {probs}")
        print("-" * 30)
