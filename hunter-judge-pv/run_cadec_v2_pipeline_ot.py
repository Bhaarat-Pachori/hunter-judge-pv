import os
import sys
import json
import time
import uuid
import torch
import asyncio
import logging
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from src.agents.cadec_v2.agent import CADECAgent

# ===================================================
# OPENTELEMETRY OBSERVABILITY ENGINE INITIALIZATION
# ===================================================
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Explicitly bind the service identifier to the target tracing metadata package
resource = Resource(attributes={
    "service.name": "hunter-judge-pv-observability"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("hunter-judge-pv-observability")

# ===================================================
# --- PIPELINE GATES & SCIENTIFIC CONFIGURATION ----
# ===================================================
HUNTER_CHECKPOINT = "results_hunter/checkpoint-680" 
DATA_PATH = "data/cadec_v2/test_split/test.csv" 
HUNTER_THRESHOLD = 0.20 
MAX_SAMPLES = None  # Engaging your 10-sample calibration test sweep

# Concurrency & Cost Overhead Controllers
CONCURRENCY_LIMIT = 10
MAX_BUDGET_USD = 10.00

# SCIENTIFIC ABLATION CHASSIS SELECTOR
# Options: "FULL", "HUNTER_ONLY", "NO_RAG", "NO_INGREDIENT", "HAND_PROMPT"
ABLATION_MODE = "NO_INGREDIENT" 

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ===================================================
# --- UNIFIED LINKED CONTEXT TRACKER ---------------
# ===================================================
class ObservabilityContextManager:
    def __init__(self, experiment_name: str, ablation_mode: str):
        self.experiment_name = experiment_name
        self.ablation_mode = ablation_mode
        
        # Mint an immutable, unique execution key for this specific pipeline invocation
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_hash = str(uuid.uuid4())[:8]
        self.run_id = f"RUN_{timestamp}_{short_hash}"
        
        # Define and create a single unified destination directory for all run artifacts
        self.output_dir = "experiment_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Structurally route all physical file paths inside our designated directory
        self.cache_path = os.path.join(self.output_dir, f"checkpoint_cache_{experiment_name}_{ablation_mode}_{self.run_id}.jsonl")
        self.stdout_path = os.path.join(self.output_dir, f"terminal_output_{experiment_name}_{ablation_mode}_{self.run_id}.log")
        self.metrics_path = os.path.join(self.output_dir, f"metrics_summary_{experiment_name}_{ablation_mode}_{self.run_id}.json")
        self.csv_path = os.path.join(self.output_dir, f"final_results_{experiment_name}_{ablation_mode}_{self.run_id}.csv")
        
        self.completed_indices = set()
        self.cached_records = {}
        self.cumulative_cost = 0.0

    def __enter__(self):
        # Unbuffered Dual Terminal Redirect (Tee Pattern)
        self.log_file = open(self.stdout_path, "a", encoding="utf-8", buffering=1)
        self.old_stdout = sys.stdout
        sys.stdout = self
        
        print("=" * 70)
        print(f"🔬 STANDING UP RESEARCH EXPERIMENT CORRIDOR: {self.experiment_name}")
        print(f"🆔 UNIQUE SYSTEM EXECUTION KEY (RUN_ID): {self.run_id}")
        print(f"📊 SYSTEM MOUNT POINT STATE: [Ablation Mode -> {self.ablation_mode}]")
        print("=" * 70)
        print(f"📝 Terminal Logs recording to: {self.stdout_path}")
        print(f"💾 Checkpoint Cache recording to: {self.cache_path}")
        
        self._load_checkpoint_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        self.log_file.close()
        if exc_type is not None:
            print(f"❌ Pipeline engine suspended operations due to fault: {exc_val}")
        return False

    def write(self, data):
        self.old_stdout.write(data)
        self.log_file.write(data)

    def flush(self):
        self.old_stdout.flush()
        self.log_file.flush()

    def _load_checkpoint_file(self):
        if os.path.exists(self.cache_path):
            print(f"♻️  Found active cache file '{self.cache_path}'. Extracting indices...")
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    idx = record["dataset_index"]
                    self.completed_indices.add(idx)
                    self.cached_records[idx] = record
            print(f"Skip-Ahead Logic activated: {len(self.completed_indices)} records recovered.")

    def log_atomic_transaction(self, idx: int, text: str, context: str, gt: int, prob: float, pred: int, reasoning: str, usage: dict):
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        cost = (in_tokens * (0.075 / 1_000_000)) + (out_tokens * (0.30 / 1_000_000))
        self.cumulative_cost += cost

        record = {
            "run_id": self.run_id,
            "dataset_index": idx,
            "text": text,
            "drug_context": context,
            "ground_truth": gt,
            "hunter_probability": float(prob),
            "final_prediction": pred,
            "chain_of_thought": reasoning,
            "transaction_cost_usd": cost
        }
        
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        self.completed_indices.add(idx)
        self.cached_records[idx] = record

        if self.cumulative_cost >= MAX_BUDGET_USD:
            print(f"🛑 CRITICAL FINANCIAL TRIP: Safe shutoff forced at ${self.cumulative_cost:.4f}.")
            sys.exit(0)

# ===================================================
# --- STAGE 1: LOCAL SEQUENCE CLASSIFICATION -------
# ===================================================
def load_hunter():
    print(f"🔫 Initializing sequence classification model checkpoint: {HUNTER_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(HUNTER_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(HUNTER_CHECKPOINT)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_hunter_predictions(contexts, texts, tokenizer, model):
    probs_list = []
    batch_size = 32
    print("🕵️  Hunter is mapping probability matrices across inputs...")
    for i in range(0, len(texts), batch_size):
        batch_contexts = contexts[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_contexts, batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_list.extend(probs[:, 1].cpu().numpy())
    return np.array(probs_list)

# ===================================================
# --- STAGE 2: ASYNCHRONOUS JUDGE OVERSEER ---------
# ===================================================
async def evaluate_sample_funnel(idx, text, drug_context, prob, gt, agent, semaphore, logger):
    if idx in logger.completed_indices:
        return logger.cached_records[idx]["final_prediction"], logger.cached_records[idx]["chain_of_thought"]

    with tracer.start_as_current_span(f"Sentence_Adjudication_Index_{idx}") as span:
        span.set_attribute("run_id", logger.run_id)
        span.set_attribute("dataset_index", idx)
        span.set_attribute("ablation_mode", logger.ablation_mode)
        span.set_attribute("hunter_probability", float(prob))
        span.set_attribute("ground_truth", gt)

        if logger.ablation_mode == "HUNTER_ONLY":
            decision = 1 if prob >= HUNTER_THRESHOLD else 0
            logger.log_atomic_transaction(idx, text, drug_context, gt, prob, decision, "Ablation Run: Hunter Engine Baseline", {"input_tokens": 0, "output_tokens": 0})
            span.set_attribute("final_decision", decision)
            span.set_status(Status(StatusCode.OK))
            return decision, "Ablation Run: Hunter Engine Baseline"

        if prob < HUNTER_THRESHOLD:
            logger.log_atomic_transaction(idx, text, drug_context, gt, prob, 0, "Auto-Rejected via Minimal Recall Probability Floor", {"input_tokens": 0, "output_tokens": 0})
            span.set_attribute("final_decision", 0)
            span.add_event("Hunter Early Drop Triggered")
            span.set_status(Status(StatusCode.OK))
            return 0, "Auto-Rejected via Minimal Recall Probability Floor"

        async with semaphore:
            try:
                run_kwargs = {"text": text, "drug_context": drug_context}
                
                if logger.ablation_mode == "NO_RAG":
                    run_kwargs["disable_rag_lookup"] = True
                elif logger.ablation_mode == "NO_INGREDIENT":
                    # run_kwargs["disable_ingredient_mapping"] = True
                    run_kwargs["drug_context"] = "Unknown / Generic"
                elif logger.ablation_mode == "HAND_PROMPT":
                    run_kwargs["override_with_legacy_prompt"] = True

                loop = asyncio.get_event_loop()
                with tracer.start_as_current_span("Remote_LLM_Verification_Call") as api_span:
                    api_span.set_attribute("prompt_raw_text", text)
                    result = await loop.run_in_executor(None, lambda: agent.run(**run_kwargs))
                
                decision = 1 if result.get("is_ade", False) else 0
                reasoning = result.get("reasoning", "No structured explanation returned from node.")
                
                span.set_attribute("clinical_narrative_text", text)
                span.set_attribute("agent_reasoning_string", reasoning)
                span.set_attribute("final_decision", decision)
                
                token_metrics = {"input_tokens": 380, "output_tokens": 140}
                logger.log_atomic_transaction(idx, text, drug_context, gt, prob, decision, reasoning, token_metrics)
                
                span.set_status(Status(StatusCode.OK))
                return decision, reasoning

            except Exception as runtime_error:
                span.record_exception(runtime_error)
                span.set_status(Status(StatusCode.ERROR, description=str(runtime_error)))
                logging.error(f"⚠️ Exception flagged on thread loop {idx}: {str(runtime_error)}")
                return 0, f"System Evaluation Interrupted: {str(runtime_error)}"

# ===================================================
# --- POST-PROCESSING DATA COMPILATION MATRIX ------
# ===================================================
async def main_async():
    print(f"📂 Mounting test file matrices from input location: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"⚠️ SAFETY SYSTEM CAP ENGAGED: Bounding execution constraints to {MAX_SAMPLES} rows.")

    texts = df['text'].astype(str).tolist()
    drugs = df['meta_doc_drugs'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    tokenizer, hunter_model = load_hunter()
    hunter_probs = get_hunter_predictions(drugs, texts, tokenizer, hunter_model)
    df['hunter_prob'] = hunter_probs

    agent = CADECAgent()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    with ObservabilityContextManager(experiment_name="CBMS_Springer_CadecV2", ablation_mode=ABLATION_MODE) as logger:
        tasks = [
            evaluate_sample_funnel(i, texts[i], drugs[i], hunter_probs[i], labels[i], agent, semaphore, logger)
            for i in range(len(df))
        ]
        
        print(f"🚀 Launching Parallelized Processing Funnel across coroutines...")
        results = await tqdm.gather(*tasks, desc="Adjudicating Pipeline Funnel States")
        
        df['final_pred'] = [r[0] for r in results]
        df['agent_reasoning'] = [r[1] for r in results]

        # ===================================================
        # --- SCIENTIFIC AUTOMATED METRICS EXPORT ENGINE ---
        # ===================================================
        print("\n💾 Compiling complete mathematical metrics and matrices to disk...")
        
        y_true_sent = df['label'].astype(int).tolist()
        y_pred_sent = df['final_pred'].astype(int).tolist()
        
        cm_sent = confusion_matrix(y_true_sent, y_pred_sent, labels=[0, 1])
        tn_s, fp_s, fn_s, tp_s = cm_sent.ravel() if cm_sent.size == 4 else (0, 0, 0, 0)

        doc_groups = df.groupby('source_file')
        doc_true, doc_pred = [], []
        for _, group in doc_groups:
            doc_true.append(1 if (group['label'] == 1).any() else 0)
            doc_pred.append(1 if (group['final_pred'] == 1).any() else 0)
            
        cm_doc = confusion_matrix(doc_true, doc_pred, labels=[0, 1])
        tn_d, fp_d, fn_d, tp_d = cm_doc.ravel() if cm_doc.size == 4 else (0, 0, 0, 0)

        # Class size checking arrays prevent reports from crashing on partial testing sweeps
        unique_classes_sent = np.unique(y_true_sent + y_pred_sent)
        master_names_sent = {0: "Negative", 1: "ADE"}
        filtered_names_sent = [master_names_sent[c] for c in unique_classes_sent]

        unique_classes_doc = np.unique(doc_true + doc_pred)
        master_names_doc = {0: "Neg Doc", 1: "Pos Doc"}
        filtered_names_doc = [master_names_doc[c] for c in unique_classes_doc]

        metrics_payload = {
            "experiment_metadata": {
                "run_id": logger.run_id,
                "experiment_name": logger.experiment_name,
                "ablation_mode": logger.ablation_mode,
                "total_processed_samples": len(df),
                "total_financial_cost_usd": logger.cumulative_cost
            },
            "sentence_level_matrix": {
                "raw_confusion_matrix": cm_sent.tolist(),
                "true_negatives": int(tn_s),
                "false_positives": int(fp_s),
                "false_negatives": int(fn_s),
                "true_positives": int(tp_s),
                "classification_report": classification_report(
                    y_true_sent, y_pred_sent, 
                    labels=unique_classes_sent, 
                    target_names=filtered_names_sent, 
                    digits=4, output_dict=True
                )
            },
            "document_level_matrix": {
                "raw_confusion_matrix": cm_doc.tolist(),
                "true_negatives": int(tn_d),
                "false_positives": int(fp_d),
                "false_negatives": int(fn_d),
                "true_positives": int(tp_d),
                "classification_report": classification_report(
                    doc_true, doc_pred, 
                    labels=unique_classes_doc, 
                    target_names=filtered_names_doc, 
                    digits=4, output_dict=True
                )
            }
        }

        with open(logger.metrics_path, "w", encoding="utf-8") as json_out:
            json.dump(metrics_payload, json_out, indent=4)
            
        df.to_csv(logger.csv_path, index=False)
        print(f"🎯 SUCCESS: Matrices permanently locked in: '{logger.metrics_path}'")

        # Standard Terminal Output Reports
        print("\n" + "="*60)
        print(f"📊 CONSOLIDATED SENTENCE PERFORMANCE MATRIX (MODE: {ABLATION_MODE})")
        print("="*60)
        print(classification_report(labels, df['final_pred'].tolist(), target_names=["Negative", "ADE"], digits=4))
        print(f"ℹ️ Core tracking complete. Run budget overhead: ${logger.cumulative_cost:.4f}")

if __name__ == "__main__":
    asyncio.run(main_async())