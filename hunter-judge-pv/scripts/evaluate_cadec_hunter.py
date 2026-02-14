import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# --- CONFIGURATION ---
# POINT THIS TO OUR BEST CHECKPOINT (checkpoint-1000 from Epoch 2)
MODEL_CHECKPOINT_PATH = "./results_hunter/checkpoint-680" 
TEST_DATA_PATH = "data/cadec_v2/test_split/test.csv"
MAX_LEN = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- REUSE DATASET CLASS ---
class CADECDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_tokenize(data_path, tokenizer):
    """
    Loads CSV and tokenizes as Sequence Pairs (Context + Text).
    Returns dataset, labels, and the raw text list.
    """
    df = pd.read_csv(data_path)
    
    # Ensure strings
    contexts = df['meta_doc_drugs'].astype(str).tolist()
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    print(f"📊 Loading {len(df)} examples from {data_path}...")
    
    # Sequence Pair Tokenization
    encodings = tokenizer(
        contexts, 
        texts, 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    
    dataset = CADECDataset(encodings, labels)
    
    # Return texts as well so we can save them later
    return dataset, labels, texts

def print_metrics(y_true, y_pred, threshold_name):
    """
    Pretty prints the classification report and confusion matrix.
    """
    print(f"\n{'='*20} RESULTS (Threshold: {threshold_name}) {'='*20}")
    
    # 1. Standard Metrics
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"🎯 Recall:    {rec:.4f} (Priority)")
    print(f"⚖️ Precision: {prec:.4f}")
    print(f"⭐ F1 Score:  {f1:.4f}")
    
    # 2. Detailed Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Negative (0)", "Positive (1)"]))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("--- Confusion Matrix ---")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print(f"⚠️ MISSED ADEs (False Negatives): {fn} (Lower is better)")
    
    return cm

def main():
    print(f"🚀 Loading Hunter Model from: {MODEL_CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    model.to(DEVICE)
    model.eval()

    # 1. Prepare Data
    # Modified to unpack 'texts' as well
    test_ds, y_true, texts = load_and_tokenize(TEST_DATA_PATH, tokenizer)
    
    # 2. Run Inference (Using Trainer is easiest way to handle batches)
    print("🧠 Running Inference...")
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_ds)
    
    # Logits are raw scores (before Softmax)
    logits = predictions.predictions
    
    # Convert Logits to Probabilities
    # We need Softmax to get a 0.0 - 1.0 score
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # --- STRATEGY: High Recall Threshold (0.20) ---
    hunter_threshold = 0.20
    y_pred_hunter = (probs[:, 1] > hunter_threshold).astype(int)
    
    # Print Metrics
    cm = print_metrics(y_true, y_pred_hunter, f"Hunter {hunter_threshold}")

    # Optional: Save Confusion Matrix Image
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"Hunter Confusion Matrix (Threshold {hunter_threshold})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("hunter_confusion_matrix.png")
    print("\n💾 Saved confusion matrix plot to 'hunter_confusion_matrix.png'")

    # ---SAVE RESULTS TO CSV FOR MCNEMAR'S TEST ---
    results_df = pd.DataFrame({
        'text': texts,
        'ground_truth': y_true,
        'hunter_pred': y_pred_hunter
    })
    
    output_filename = "hunter_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"💾 Saved inference results to '{output_filename}' (Rows: {len(results_df)})")

if __name__ == "__main__":
    main()
