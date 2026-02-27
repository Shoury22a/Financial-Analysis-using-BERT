# -*- coding: utf-8 -*-
"""
FINSIGHT AI - Financial Sentiment Model Training
=================================================
Strategy:
  - Uses Financial PhraseBank (sentences_allagree) from HuggingFace Hub
    → 2,264 expert-labeled sentences, unanimously agreed upon by annotators
  - Balances classes by downsampling neutral to match minority class count
  - Reads all hyperparameters from config/config.yaml
  - PyTorch + HuggingFace Trainer API
  - Label mapping: Financial PhraseBank native → 0=negative, 1=neutral, 2=positive
  - Saves model + correct config.json for prediction.py / api.py
"""

import os
import json
import time
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import zipfile
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

MODEL_NAME   = CFG["model"]["name"]
NUM_LABELS   = CFG["model"]["num_labels"]
MAX_LENGTH   = CFG["model"]["max_length"]
OUTPUT_DIR   = "financial_sentiment_model"
RESULTS_DIR  = CFG["training"]["output_dir"]
EPOCHS       = CFG["training"]["num_epochs"]
BATCH_SIZE   = CFG["training"]["batch_size"]
LR           = CFG["training"]["learning_rate"]
WEIGHT_DECAY = CFG["training"]["weight_decay"]
WARMUP_RATIO = CFG["training"]["warmup_ratio"]
TEST_SIZE    = CFG["training"]["test_size"]
RANDOM_SEED  = CFG["training"]["random_seed"]
PATIENCE     = CFG["training"]["early_stopping_patience"]

# Financial PhraseBank native label mapping
# 0 = negative, 1 = neutral, 2 = positive
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 70)
print("FINSIGHT AI - MODEL TRAINING (Financial PhraseBank)")
print(f"Base model:   {MODEL_NAME}")
print(f"Label map:    {ID2LABEL}")
print(f"Config:       {CONFIG_PATH}")
print("=" * 70)


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

    # --- Step 1: Load Financial PhraseBank ---
    print("\n[STEP 1/6] Loading Financial PhraseBank (all agreement levels)...")
    # Download the source ZIP directly from the HuggingFace dataset repo
    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    # Combine all four agreement subsets for maximum coverage
    # More agreement subsets = more nuanced examples (e.g. layoffs, lawsuits as negative)
    # File format: each line is  "sentence@label"  (label = positive/negative/neutral)
    TEXT_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
    TARGET_FILES = [
        "Sentences_AllAgree.txt",
        "Sentences_75Agree.txt",
        "Sentences_66Agree.txt",
        "Sentences_50Agree.txt",
    ]
    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        for target_key in TARGET_FILES:
            match = next((n for n in all_names if target_key in n), None)
            if not match:
                print(f"  -> Warning: {target_key} not found in ZIP, skipping")
                continue
            with zf.open(match) as f:
                for raw_line in f:
                    line = raw_line.decode("latin-1").strip()
                    if "@" not in line:
                        continue
                    sentence, label_str = line.rsplit("@", 1)
                    label_str = label_str.strip().lower()
                    if label_str in TEXT_LABEL_MAP:
                        rows.append({"sentence": sentence.strip(), "label": TEXT_LABEL_MAP[label_str]})

    df = pd.DataFrame(rows)
    # Deduplicate — keep first occurrence (highest agreement)
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    print(f"  -> Loaded {len(df)} unique samples across all agreement levels")
    print(f"  -> Raw distribution: {df['label'].value_counts().to_dict()}")
    print(f"  -> Label mapping: 0=negative, 1=neutral, 2=positive")

    # --- Step 2: Balance Classes ---
    print("\n[STEP 2/6] Balancing classes...")
    # Find the minority class count
    label_counts = df["label"].value_counts()
    min_count = int(label_counts.min())
    print(f"  -> Minority class size: {min_count}")

    balanced_dfs = []
    for label_id in sorted(df["label"].unique()):
        class_df = df[df["label"] == label_id]
        sampled = class_df.sample(n=min_count, random_state=RANDOM_SEED)
        balanced_dfs.append(sampled)
        print(f"  -> Class {label_id} ({ID2LABEL[label_id]:>8s}): {len(class_df)} -> {min_count} samples")

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  -> Total balanced samples: {len(df_balanced)}")
    print(f"  -> Distribution: {df_balanced['label'].value_counts().to_dict()}")

    # --- Step 3: Split ---
    print("\n[STEP 3/6] Splitting into train/validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_balanced["sentence"].tolist(),
        df_balanced["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_balanced["label"],
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
        ignore_mismatched_sizes=True,
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
        report_to="none",
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

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Ensure config.json has correct id2label / label2id
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["id2label"] = {str(k): v for k, v in ID2LABEL.items()}
    cfg["label2id"] = LABEL2ID
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
    labels_arr = predictions_output.label_ids

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    print("\n  DETAILED CLASSIFICATION REPORT:")
    print(classification_report(labels_arr, preds, target_names=label_names, zero_division=0))

    cm = confusion_matrix(labels_arr, preds)
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
        "dataset": "financial_phrasebank/sentences_allagree (balanced)",
        "samples_per_class": min_count,
        "training_time_seconds": round(elapsed, 1),
        "framework": "PyTorch + HuggingFace Trainer",
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    quality = "EXCELLENT" if accuracy > 0.9 else "GOOD" if accuracy > 0.8 else "FAIR"
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Dataset:       Financial PhraseBank (sentences_allagree, balanced)")
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
