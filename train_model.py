# -*- coding: utf-8 -*-
"""
FINSIGHT AI - Financial Sentiment Model Training
=================================================
Strategy:
  - Uses LOCAL data (financial_news_augmented.csv) — NO network downloads
  - Programmatic augmentation to reach balanced dataset
  - Reads all hyperparameters from config/config.yaml
  - PyTorch + HuggingFace Trainer API
  - Label mapping: 0=Positive, 1=Negative, 2=Neutral (finbert native)
  - Saves model + correct config.json for prediction.py / api.py
"""

import os
import re
import json
import random
import time
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ==================== LOAD CONFIG ====================
CONFIG_PATH = os.path.join("config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

MODEL_NAME      = CFG["model"]["name"]
NUM_LABELS      = CFG["model"]["num_labels"]
MAX_LENGTH      = CFG["model"]["max_length"]
OUTPUT_DIR      = "financial_sentiment_model"
RESULTS_DIR     = CFG["training"]["output_dir"]
EPOCHS          = CFG["training"]["num_epochs"]
BATCH_SIZE      = CFG["training"]["batch_size"]
LR              = CFG["training"]["learning_rate"]
WEIGHT_DECAY    = CFG["training"]["weight_decay"]
WARMUP_RATIO    = CFG["training"]["warmup_ratio"]
TEST_SIZE       = CFG["training"]["test_size"]
RANDOM_SEED     = CFG["training"]["random_seed"]
PATIENCE        = CFG["training"]["early_stopping_patience"]

# ProsusAI/finbert native label mapping (must match model's config.json)
ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 70)
print("FINSIGHT AI - MODEL TRAINING")
print(f"Base model:   {MODEL_NAME}")
print(f"Label map:    {ID2LABEL}")
print(f"Config:       {CONFIG_PATH}")
print("=" * 70)


# ==================== DATA CLEANING ====================
def is_clean_financial_sentence(text: str) -> bool:
    """
    Filter out noisy rows: news article blurbs, ads, non-financial text.
    Returns True if the sentence is a clean, short financial statement.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    # Too short or too long
    if len(text) < 15 or len(text) > 300:
        return False
    # Contains URLs
    if "http" in text or "www." in text:
        return False
    # Looks like a news article snippet (has multiple sentences and is very long)
    if text.count(".") > 4 and len(text) > 200:
        return False
    # Non-English characters (German, etc.)
    if re.search(r"[äöüÄÖÜß]", text):
        return False
    # Promotional / boilerplate text
    noise_phrases = [
        "click here", "subscribe", "sign up", "download", "qr code",
        "reprints", "permissions", "licensing", "business insider",
        "kostenfrei", "werbung", "jetzt", "scan the qr",
    ]
    text_lower = text.lower()
    if any(p in text_lower for p in noise_phrases):
        return False
    return True


# ==================== AUGMENTATION ====================
# Financial synonym dictionary for augmentation (no network needed)
FINANCIAL_SYNONYMS = {
    "revenue": ["sales", "income", "turnover", "earnings"],
    "profit": ["earnings", "net income", "surplus", "gain"],
    "loss": ["deficit", "shortfall", "decline", "drop"],
    "growth": ["expansion", "increase", "rise", "improvement"],
    "decline": ["decrease", "fall", "drop", "reduction"],
    "strong": ["robust", "solid", "impressive", "excellent"],
    "weak": ["poor", "disappointing", "sluggish", "subdued"],
    "increased": ["grew", "rose", "expanded", "surged"],
    "decreased": ["fell", "dropped", "declined", "contracted"],
    "exceeded": ["surpassed", "beat", "outperformed", "topped"],
    "missed": ["fell short of", "underperformed", "failed to meet"],
    "announced": ["reported", "disclosed", "revealed", "stated"],
    "company": ["firm", "organization", "corporation", "business"],
    "quarterly": ["three-month", "Q3", "Q4", "fiscal quarter"],
    "analyst": ["expert", "researcher", "market observer"],
    "significant": ["substantial", "considerable", "notable", "major"],
    "improved": ["enhanced", "strengthened", "advanced", "progressed"],
    "concerns": ["worries", "issues", "challenges", "risks"],
}

NUMBER_VARIANTS = {
    "25%": ["20%", "30%", "28%", "22%"],
    "10%": ["12%", "8%", "15%", "9%"],
    "third": ["fourth", "second", "fifth"],
    "three": ["four", "five", "two"],
    "record": ["historic", "all-time", "unprecedented"],
}


def augment_sentence(text: str, n: int = 3) -> list:
    """
    Generate n augmented variants of a sentence using synonym substitution
    and number variation. No external libraries or network needed.
    """
    variants = []
    words = text.split()

    for _ in range(n * 3):  # Try more times than needed, keep best n
        new_words = words.copy()
        changed = False

        # Apply synonym substitution (1-2 words per sentence)
        for i, word in enumerate(new_words):
            clean_word = word.lower().rstrip(".,;:")
            if clean_word in FINANCIAL_SYNONYMS and random.random() < 0.4:
                replacement = random.choice(FINANCIAL_SYNONYMS[clean_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words[i] = word.replace(clean_word, replacement).replace(
                    clean_word.capitalize(), replacement.capitalize()
                )
                changed = True
                break  # One substitution per variant

        # Apply number variation
        new_text = " ".join(new_words)
        for num, variants_list in NUMBER_VARIANTS.items():
            if num in new_text and random.random() < 0.3:
                new_text = new_text.replace(num, random.choice(variants_list), 1)
                changed = True

        if changed and new_text != text:
            variants.append(new_text)

        if len(variants) >= n:
            break

    return variants[:n]


def build_augmented_dataset(df: pd.DataFrame, target_per_class: int = 300) -> pd.DataFrame:
    """
    Augment the dataset to reach target_per_class samples per label.
    Uses programmatic augmentation — no network required.
    """
    augmented_rows = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label].copy()
        current_count = len(class_df)
        needed = max(0, target_per_class - current_count)

        label_name = ID2LABEL.get(label, str(label))
        print(f"  Class {label} ({label_name:>8s}): {current_count} original -> augmenting to {current_count + needed}")

        augmented_rows.append(class_df)

        if needed > 0:
            # Cycle through existing samples and augment
            source_texts = class_df["sentence"].tolist()
            generated = []
            idx = 0
            while len(generated) < needed:
                text = source_texts[idx % len(source_texts)]
                new_variants = augment_sentence(text, n=3)
                for v in new_variants:
                    if len(generated) < needed:
                        generated.append({"sentence": v, "label": label})
                idx += 1

            augmented_rows.append(pd.DataFrame(generated))

    result = pd.concat(augmented_rows, ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return result


# ==================== DATASET CLASS ====================
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ==================== METRICS ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ==================== MAIN TRAINING ====================
def train():
    start_time = time.time()

    # --- Step 1: Load & Clean Data ---
    print("\n[STEP 1/6] Loading and cleaning local data...")
    local_csv = "financial_news_augmented.csv"
    df = pd.read_csv(local_csv)

    # Clean noisy rows
    before = len(df)
    df = df[df["sentence"].apply(is_clean_financial_sentence)].copy()
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    after = len(df)
    print(f"  -> Loaded {before} rows, kept {after} clean rows")
    print(f"  -> Distribution: {df['label'].value_counts().to_dict()}")

    # --- Step 2: Augment ---
    print("\n[STEP 2/6] Augmenting dataset to 300 samples per class...")
    df_aug = build_augmented_dataset(df, target_per_class=300)
    print(f"  -> Total after augmentation: {len(df_aug)} samples")
    print(f"  -> Distribution: {df_aug['label'].value_counts().to_dict()}")

    # --- Step 3: Split ---
    print("\n[STEP 3/6] Splitting into train/validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_aug["sentence"].tolist(),
        df_aug["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_aug["label"],
    )
    print(f"  -> Train: {len(train_texts)}, Validation: {len(val_texts)}")

    # --- Step 4: Tokenize ---
    print("\n[STEP 4/6] Tokenizing...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc   = tokenizer(val_texts,   truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SentimentDataset(train_enc, train_labels)
    val_dataset   = SentimentDataset(val_enc,   val_labels)
    print("  -> Tokenization complete")

    # --- Step 5: Train ---
    print(f"\n[STEP 5/6] Training (up to {EPOCHS} epochs, early stopping patience={PATIENCE})...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={str(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=20,
        seed=RANDOM_SEED,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    # --- Step 6: Save & Evaluate ---
    print("\n[STEP 6/6] Saving model and evaluating...")

    # Save best model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Ensure config.json has correct id2label / label2id
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["id2label"]  = {str(k): v for k, v in ID2LABEL.items()}
    cfg["label2id"]  = LABEL2ID
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Save label_map.json
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({str(k): v for k, v in ID2LABEL.items()}, f, indent=2)

    # Final evaluation
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", 0)
    f1       = eval_results.get("eval_f1", 0)

    # Detailed report
    predictions_output = trainer.predict(val_dataset)
    preds  = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    print("\n  DETAILED CLASSIFICATION REPORT:")
    print(classification_report(labels, preds, target_names=label_names, zero_division=0))

    cm = confusion_matrix(labels, preds)
    print("  CONFUSION MATRIX (rows=Actual, cols=Predicted):")
    print(f"  {'':>10s}  " + "  ".join(f"{n:>8s}" for n in label_names))
    for i, row_label in enumerate(label_names):
        row = "  ".join(f"{cm[i][j]:>8d}" for j in range(NUM_LABELS))
        print(f"  {row_label:>10s}  {row}")

    # Save metrics
    elapsed = time.time() - start_time
    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "training_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "label_mapping": ID2LABEL,
        "base_model": MODEL_NAME,
        "training_time_seconds": round(elapsed, 1),
        "framework": "PyTorch + HuggingFace Trainer",
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    quality = "EXCELLENT" if accuracy > 0.9 else "GOOD" if accuracy > 0.8 else "FAIR"
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Accuracy:      {accuracy*100:.2f}%")
    print(f"  F1-Score:      {f1:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Model quality: {quality}")
    print(f"  Saved to:      {OUTPUT_DIR}/")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. python test_predictions.py")
    print("  2. streamlit run prediction.py")
    print("=" * 70)


if __name__ == "__main__":
    train()
