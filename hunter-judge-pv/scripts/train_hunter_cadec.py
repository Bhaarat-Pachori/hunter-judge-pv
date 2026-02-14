import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# --- CONFIGURATION ---
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TRAIN_DATA_PATH = "data/cadec_v2/data_splits/train.csv" 
VAL_DATA_PATH = "data/cadec_v2/data_splits/val.csv" 
OUTPUT_DIR = "models/hunter_v1" 
MAX_LEN = 128
BATCH_SIZE = 16 
EPOCHS = 4
LEARNING_RATE = 2e-5

# Detect Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CADECV2Dataset(torch.utils.data.Dataset):
    """
    Standard PyTorch Dataset.
    Expects 'encodings' to be a Dictionary of Lists (not Tensors).
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert List -> Tensor on the fly
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that accepts calculated class weights.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Verify weights are on the correct device
        self.class_weights = class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Use the calculated weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions)
    }

def main():
    print(f"🚀 Initializing Hunter Training on {device}...")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    
    # Extract Columns for Sequence Pairs
    # Context (Sequence A) = Drug List
    # Target (Sequence B) = Sentence Text
    train_contexts = train_df['meta_doc_drugs'].astype(str).tolist()
    train_texts = train_df['text'].astype(str).tolist()
    train_labels = train_df['label'].tolist()

    val_contexts = val_df['meta_doc_drugs'].astype(str).tolist()
    val_texts = val_df['text'].astype(str).tolist()
    val_labels = val_df['label'].tolist()
    
    # 2. Tokenize (Sequence Pairs)
    print("🧠 Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Note: Removed return_tensors="pt". We let the Dataset handle tensor conversion.
    train_encodings = tokenizer(
        train_contexts,      # Seq A
        train_texts,         # Seq B
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    
    val_encodings = tokenizer(
        val_contexts, 
        val_texts, 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )

    train_ds = CADECV2Dataset(train_encodings, train_labels)
    val_ds = CADECV2Dataset(val_encodings, val_labels)

    # 3. Calculate Class Weights (Dynamic)
    # This automatically finds the ratio (e.g., 65% neg / 35% pos) and creates the inverse weight
    print("⚖️ Calculating Class Weights...")
    weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"   Weights: {class_weights} (Neg, Pos)")

    # 4. Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # 5. Train Config
    args = TrainingArguments(
        output_dir='./results_hunter',
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall", # Optimize for RECALL
        greater_is_better=True,
        logging_steps=50,
        use_mps_device=True 
    )

    # Pass weights to Custom Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights 
    )

    # 6. Execute
    print("🏋️ Starting Training...")
    trainer.train()

    # 7. Save
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Hunter Training Complete.")

if __name__ == "__main__":
    main()