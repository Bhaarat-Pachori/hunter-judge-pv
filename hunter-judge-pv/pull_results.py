import os
import json
import glob

results_dir = "experiment_results"
json_files = glob.glob(os.path.join(results_dir, "metrics_summary_*.json"))

print(f"| {'Ablation Mode':<15} | {'Sent F1':<8} | {'Sent Prec':<9} | {'Sent Rec':<8} | {'Doc F1':<8} | {'Doc Prec':<8} | {'Doc Rec':<8} |")
print(f"|{'-'*17}|{'-'*10}|{'-'*11}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")

for fpath in json_files:
    with open(fpath, "r") as f:
        data = json.load(f)
    meta = data["experiment_metadata"]
    mode = meta["ablation_mode"]
    
    # Extract sentence metrics (safely handling string key variations)
    sent_rep = data["sentence_level_matrix"]["classification_report"]
    sent_ade = sent_rep.get("ADE", sent_rep.get("1", {}))
    s_f1 = sent_ade.get("f1-score", 0.0)
    s_pr = sent_ade.get("precision", 0.0)
    s_rc = sent_ade.get("recall", 0.0)
    
    # Extract document metrics
    doc_rep = data["document_level_matrix"]["classification_report"]
    doc_pos = doc_rep.get("Pos Doc", doc_rep.get("1", {}))
    d_f1 = doc_pos.get("f1-score", 0.0)
    d_pr = doc_pos.get("precision", 0.0)
    d_rc = doc_pos.get("recall", 0.0)
    
    print(f"| {mode:<15} | {s_f1:.4f}   | {s_pr:.4f}    | {s_rc:.4f}   | {d_f1:.4f}   | {d_pr:.4f}   | {d_rc:.4f}   |")