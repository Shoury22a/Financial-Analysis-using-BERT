import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import os
import shutil

# --- CONFIGURATION ---
MODEL_NAME = "ProsusAI/finbert"
DATA_FILE = "financial_news_augmented.csv"
OUTPUT_DIR = "financial_sentiment_model"

# Clean up previous model dir if exists to ensure clean save
if os.path.exists(OUTPUT_DIR):
    try:
        shutil.rmtree(OUTPUT_DIR)
    except:
        pass

# --- DATA PREPARATION ---
class FinancialDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Split (small dataset, so we use most for training to overfit to user request)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

print(f"Training on {len(train_texts)} samples, Validating on {len(val_texts)} samples.")

# --- TOKENIZATION ---
print("Tokenizing...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = FinancialDataset(train_encodings, train_labels)
val_dataset = FinancialDataset(val_encodings, val_labels)

# --- MODEL SETUP ---
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Training Arguments (Optimized for small dataset fine-tuning)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,              # Enough to learn new patterns without destroying old knowledge
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,                 # Short warmup
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="no",              # We save manually at end
    learning_rate=2e-5,              # Low LR for fine-tuning
    use_cpu=True                     # Force CPU if no GPU
)

# --- TRAINING ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")
trainer.train()

# --- SAVING ---
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model retraining complete!")
