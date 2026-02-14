import pandas as pd

# Load the results (Make sure this matches our output file)
FILE_PATH = "final_cadec_results_with_cot.csv" # or "final_ade_results.csv"

def analyze_performance(file_path):
    print(f"📊 Analyzing {file_path}...\n")
    df = pd.read_csv(file_path)
    
    # Ensure reasoning column exists; fill with N/A if missing (e.g. for Hunter auto-rejects)
    if 'agent_reasoning' not in df.columns:
        df['agent_reasoning'] = "N/A (Hunter Auto-Reject or Old Run)"

    # Define the buckets
    # 1. True Positives (Agent said YES, Truth is YES)
    tp = df[(df['final_pred'] == 1) & (df['label'] == 1)]
    
    # 2. False Positives (Agent said YES, Truth is NO) -> "Hallucinations"
    fp = df[(df['final_pred'] == 1) & (df['label'] == 0)]
    
    # 3. False Negatives (Agent said NO, Truth is YES) -> "Misses"
    fn = df[(df['final_pred'] == 0) & (df['label'] == 1)]
    
    # 4. Agent Saves (Hunter said YES, Agent said NO, Truth is NO)
    saves = df[
        (df['final_pred'] == 0) & 
        (df['label'] == 0) & 
        (df['agent_reasoning'].str.len() > 20) # Filter out "N/A" or short reject logs
    ]

    # --- HELPER PRINT FUNCTION ---
    def print_examples(subset, title, limit=5):
        print(f"--- {title} [{len(subset)} examples] ---")
        for i, row in subset.head(limit).iterrows():
            print(f"[{i}] Text: {row['text'][:100]}...")
            print(f"    Drug: {row.get('meta_doc_drugs', 'N/A')}")
            print(f"    Reasoning: {row['agent_reasoning']}")
            print("-" * 60)
        print("\n")

    # --- EXECUTE ANALYSIS ---
    print_examples(tp, "🟢 AGENT WINS (True Positives) - Explainable AI")
    print_examples(saves, "🛡️ AGENT SAVES (Correct Rejections) - Context Filtering")
    print_examples(fp, "⚠️ AGENT HALLUCINATIONS (False Positives) - Analysis Needed")
    print_examples(fn, "🔴 MISSES (False Negatives) - Sensitivity Issues")

if __name__ == "__main__":
    analyze_performance(FILE_PATH)