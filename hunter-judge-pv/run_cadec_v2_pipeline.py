import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
from src.agents.cadec_v2.agent import CADECAgent


# --- CONFIGURATION ---
# CRITICAL: Point to the specific checkpoint (Epoch 2), NOT the final model folder
HUNTER_CHECKPOINT = "results_hunter/checkpoint-680" 
DATA_PATH = "data/cadec_v2/data_splits/test.csv" # Change to val.csv for tuning
HUNTER_THRESHOLD = 0.20 # The "Recall First" setting
MAX_SAMPLES = None # Set to 50 for quick debugging

# Device Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_hunter():
    print(f"🔫 Loading Hunter from {HUNTER_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(HUNTER_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(HUNTER_CHECKPOINT)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_hunter_predictions(contexts, texts, tokenizer, model):
    """
    Runs BERT inference using Sequence Pairs (Context + Text).
    """
    probs_list = []
    batch_size = 32
    print("🕵️  Hunter is scanning sentences...")
    
    # Iterate in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_contexts = contexts[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize as Pairs: [CLS] Context [SEP] Text [SEP]
        inputs = tokenizer(
            batch_contexts,
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply Softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Store probability of Class 1 (ADE)
            probs_list.extend(probs[:, 1].cpu().numpy())
            
    return np.array(probs_list)

def calculate_document_metrics(df, pred_col, label_col):
    """
    Performs the 'Roll-Up' Logic:
    If ANY sentence in a document is Positive -> Document is Positive.
    """
    print("\n📜 Calculating Document-Level Metrics (The Roll-Up)...")
    
    # Group by Source File
    doc_groups = df.groupby('source_file')
    
    doc_true = []
    doc_pred = []
    
    for filename, group in doc_groups:
        # Ground Truth: Is there at least one true ADE in this file?
        is_pos_doc = (group[label_col] == 1).any()
        doc_true.append(1 if is_pos_doc else 0)
        
        # Prediction: Did we flag at least one sentence in this file?
        is_pred_doc = (group[pred_col] == 1).any()
        doc_pred.append(1 if is_pred_doc else 0)
        
    print(classification_report(doc_true, doc_pred, target_names=["Neg Doc", "Pos Doc"]))
    return doc_true, doc_pred

def main():
    # 1. Load Data
    print(f"📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"   ⚠️ Debug Mode: Using only {MAX_SAMPLES} samples.")

    # Ensure strings
    texts = df['text'].astype(str).tolist()
    drugs = df['meta_doc_drugs'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # 2. Stage 1: Hunter (BERT)
    tokenizer, hunter_model = load_hunter()
    
    # Run Inference
    hunter_probs = get_hunter_predictions(drugs, texts, tokenizer, hunter_model)
    
    # Identify Suspects (Indices where Prob > Threshold)
    suspect_indices = [i for i, p in enumerate(hunter_probs) if p >= HUNTER_THRESHOLD]
    
    print(f"\n📊 Hunter Results (Threshold {HUNTER_THRESHOLD}):")
    print(f"   Total Sentences: {len(df)}")
    print(f"   Suspects Sent to Agent: {len(suspect_indices)} ({len(suspect_indices)/len(df):.1%})")
    print(f"   Auto-Rejected (Safe): {len(df) - len(suspect_indices)}")

    # 3. Stage 2: Agent (The Judge)
    print("\n⚖️  Initializing CADECAgent (The Context Judge)...")
    agent = CADECAgent() 

    final_predictions = []
    agent_reasoning_list = []  # <--- NEW LIST to store CoT
    agent_rejections = 0
    
    print(f"🚀 Running Hybrid Pipeline...")
    
    for i in tqdm(range(len(df))):
        # Path A: Auto-Rejection (Hunter said No)
        if i not in suspect_indices:
            final_predictions.append(0)
            agent_reasoning_list.append("Auto-Rejected by Hunter (Low Probability)") # Log this too
            continue

        # Path B: Agent Verification
        text = texts[i]
        drug_context = drugs[i]
        
        # Call the Agent
        result = agent.run(text=text, drug_context=drug_context)
        decision = 1 if result["is_ade"] else 0
        reason = result["reasoning"]  # <--- CAPTURE REASONING
        
        if decision == 0:
            agent_rejections += 1
            
        final_predictions.append(decision)
        agent_reasoning_list.append(reason) # Store it

    # 4. Final Evaluation
    df['final_pred'] = final_predictions
    df['agent_reasoning'] = agent_reasoning_list
    
    print("\n" + "="*40)
    print("🚀 SENTENCE-LEVEL PERFORMANCE")
    print("="*40)
    print(classification_report(labels, final_predictions, target_names=["Negative", "ADE"]))
    
    print(f"ℹ️  The Agent filtered out {agent_rejections} False Positives from the Hunter.")

    # 5. Document-Level Evaluation (Crucial for Paper)
    print("\n" + "="*40)
    print("📄 DOCUMENT-LEVEL PERFORMANCE (SOTA Comparison)")
    print("="*40)
    calculate_document_metrics(df, 'final_pred', 'label')
    
    # Save Results
    output_file = "final_cadec_results_with_cot.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results (with CoT) saved to {output_file}")

if __name__ == "__main__":
    main()
