import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from src.agents.cadec_v2.agent import CADECAgent

# --- CONFIGURATION ---
HUNTER_CHECKPOINT = "results_hunter/checkpoint-680" 
DATA_PATH = "data/ade/test_dataset/test copy.csv" # Ensure this path is correct
HUNTER_THRESHOLD = 0.15 # Lower threshold slightly for Cross-Domain to boost Recall
MAX_SAMPLES = None 

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_hunter():
    print(f"🔫 Loading Hunter from {HUNTER_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(HUNTER_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(HUNTER_CHECKPOINT)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_hunter_predictions(contexts, texts, tokenizer, model):
    probs_list = []
    batch_size = 32
    print("🕵️  Hunter is scanning sentences (Cross-Domain)...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_contexts = contexts[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
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
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_list.extend(probs[:, 1].cpu().numpy())
            
    return np.array(probs_list)

def main():
    print(f"📂 Loading ADE Corpus from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    # --- CRITICAL FIX FOR ADE CORPUS ---
    # ADE Corpus likely lacks 'meta_doc_drugs'. We simulate it.
    # If the text mentions a drug, great. If not, we use a generic placeholder.
    # Ideally, we would extract entities, but for Zero-Shot, we use "the medication".
    # This matches the clinical tone of ADE Corpus better than "this drug".
    if 'meta_doc_drugs' not in df.columns:
        print("⚠️ 'meta_doc_drugs' column missing. Injecting generic context 'the medication'.")
        df['meta_doc_drugs'] = "the medication"

    texts = df['text'].astype(str).tolist()
    drugs = df['meta_doc_drugs'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Stage 1: Hunter
    tokenizer, hunter_model = load_hunter()
    hunter_probs = get_hunter_predictions(drugs, texts, tokenizer, hunter_model)
    
    suspect_indices = [i for i, p in enumerate(hunter_probs) if p >= HUNTER_THRESHOLD]
    
    print(f"\n📊 Hunter Results (Threshold {HUNTER_THRESHOLD}):")
    print(f"   Total Sentences: {len(df)}")
    print(f"   Suspects Sent to Agent: {len(suspect_indices)} ({len(suspect_indices)/len(df):.1%})")

    # Stage 2: Agent
    print("\n⚖️  Initializing Agent...")
    agent = CADECAgent() 

    final_predictions = []
    agent_rejections = 0
    
    print(f"🚀 Running Hybrid Pipeline on ADE Corpus...")
    for i in tqdm(range(len(df))):
        if i not in suspect_indices:
            final_predictions.append(0)
            continue

        text = texts[i]
        drug_context = drugs[i]
        
        result = agent.run(text=text, drug_context=drug_context)
        decision = 1 if result["is_ade"] else 0
        
        if decision == 0:
            agent_rejections += 1
            
        final_predictions.append(decision)

    # Final Evaluation (Sentence-Level ONLY)
    print("\n" + "="*40)
    print("🚀 ADE CORPUS (ZERO-SHOT) PERFORMANCE")
    print("="*40)
    print(classification_report(labels, final_predictions, target_names=["Negative", "ADE"]))
    
    print(f"ℹ️  The Agent filtered out {agent_rejections} False Positives from the Hunter.")
    print("⚠️  Note: Document-Level metrics skipped (No 'source_file' ID in ADE Corpus).")

    # Save
    df['final_pred'] = final_predictions
    df.to_csv("final_ade_results.csv", index=False)

if __name__ == "__main__":
    main()