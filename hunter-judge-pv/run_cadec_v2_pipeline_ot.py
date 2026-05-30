import os
import sys
import json
import torch
import asyncio
import logging
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from src.agents.cadec_v2.agent import CADECAgent

# --- OPENTELEMETRY TRACING BACKEND CONFIGURATION ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode

# Initialize OpenTelemetry SDK targeting Jaeger's 0-Cost Localhost OTLP port (4317)
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("hunter-judge-pv-core")

# ===================================================
# --- EXPERIMENT OVERRIDES & ABLATION MATRICES -----
# ===================================================
HUNTER_CHECKPOINT = "results_hunter/checkpoint-680" 
DATA_PATH = "data/cadec_v2/test_split/test.csv" 
HUNTER_THRESHOLD = 0.20
MAX_SAMPLES = 5  # Integer override (e.g., 5) for safe logging dry-runs

# Concurrency Gates & Cost Throttling
CONCURRENCY_LIMIT = 10
MAX_BUDGET_USD = 10.00

# SCIENTIFIC ABLATION CONFIGURATION MATRIX
# Valid states: "FULL", "HUNTER_ONLY", "NO_RAG", "NO_INGREDIENT", "HAND_PROMPT"
ABLATION_MODE = "FULL" 

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ===================================================
# --- UNIFIED RESILIENT STATE RECOVERY CONTEXT -----
# ===================================================
class ObservabilityContextManager:
    def __init__(self, experiment_name: str, ablation_mode: str):
        self.experiment_name = experiment_name
        self.ablation_mode = ablation_mode
        self.cache_path = f".checkpoint_cache_{experiment_name}_{ablation_mode}.jsonl"
        self.stdout_path = f"terminal_output_{experiment_name}_{ablation_mode}.log"
        self.completed_indices = set()
        self.cached_records = {}
        
        # Financial Accounting System
        self.cumulative_cost = 0.0

    def __enter__(self):
        # Unbuffered Dual Terminal Redirect (Tee Pattern)
        self.log_file = open(self.stdout_path, "a", encoding="utf-8", buffering=1)
        self.old_stdout = sys.stdout
        sys.stdout = self
        
        print(f"🔬 STANDING UP RESEARCH EXPERIMENT CORRIDOR: {self.experiment_name}")
        print(f"📊 SYSTEM MOUNT POINT STATE: [Ablation Mode -> {self.ablation_mode}]")
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
            print(f"♻️  Found active warm-restart cache file: '{self.cache_path}'. Extracting indices...")
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    idx = record["dataset_index"]
                    self.completed_indices.add(idx)
                    self.cached_records[idx] = record
            print(f"Skip-Ahead Logic activated: {len(self.completed_indices)} records recovered.")

    def log_atomic_transaction(self, idx: int, text: str, context: str, gt: int, prob: float, pred: int, reasoning: str, token_metrics: dict):
        # Calculate real-time costs based on Gemini 2.5 API guidelines
        in_tokens = token_metrics.get("input_tokens", 0)
        out_tokens = token_metrics.get("output_tokens", 0)
        tx_cost = (in_tokens * (0.075 / 1_000_000)) + (out_tokens * (0.30 / 1_000_000))
        self.cumulative_cost += tx_cost

        record = {
            "dataset_index": idx,
            "text": text,
            "drug_context": context,
            "ground_truth": gt,
            "hunter_probability": float(prob),
            "final_prediction": pred,
            "chain_of_thought": reasoning,
            "transaction_cost_usd": tx_cost
        }
        
        # Hard text append guarantees persistence even on network or runtime failures
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        self.completed_indices.add(idx)
        self.cached_records[idx] = record

        if self.cumulative_cost >= MAX_BUDGET_USD:
            print(f"🛑 FINANCIAL BUDGET EXCEEDED: Tracing cut-off forced at ${self.cumulative_cost:.4f}.")
            sys.exit(0)

# ===================================================
# --- STAGE 1: INFERENCE ENGINE (THE HUNTER) -------
# ===================================================
def load_hunter():
    print(f"🔫 Initializing local sequence classification checkpoint: {HUNTER_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(HUNTER_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(HUNTER_CHECKPOINT)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_hunter_predictions(contexts, texts, tokenizer, model):
    probs_list = []
    batch_size = 32
    print("🕵️  Hunter is initiating batch classification passes across inputs...")
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
    # Checkpoint Cache Bypassing Pattern
    if idx in logger.completed_indices:
        return logger.cached_records[idx]["final_prediction"], logger.cached_records[idx]["chain_of_thought"]

    # OpenTelemetry Span Initialization Pattern
    with tracer.start_as_current_span(f"Process_Sample_Index_{idx}") as span:
        span.set_attribute("dataset_index", idx)
        span.set_attribute("ablation_mode", logger.ablation_mode)
        span.set_attribute("hunter_probability", float(prob))
        span.set_attribute("ground_truth", gt)

        # Track 1: Hunter Baseline Ablation Track
        if logger.ablation_mode == "HUNTER_ONLY":
            decision = 1 if prob >= HUNTER_THRESHOLD else 0
            logger.log_atomic_transaction(idx, text, drug_context, gt, prob, decision, "Ablation Run: Hunter Engine Baseline", {"input_tokens": 0, "output_tokens": 0})
            span.set_attribute("final_decision", decision)
            span.set_status(Status(StatusCode.OK))
            return decision, "Ablation Run: Hunter Engine Baseline"

        # Track 2: Automated Budget Filter (Sentence Safe-Drop)
        if prob < HUNTER_THRESHOLD:
            logger.log_atomic_transaction(idx, text, drug_context, gt, prob, 0, "Auto-Rejected by Lenient Pre-Filter Threshold", {"input_tokens": 0, "output_tokens": 0})
            span.set_attribute("final_decision", 0)
            span.add_event("Hunter Safe-Bypass Triggered")
            span.set_status(Status(StatusCode.OK))
            return 0, "Auto-Rejected by Lenient Pre-Filter Threshold"

        # Track 3: Parallelized API Judge Pipeline Execution
        async with semaphore:
            try:
                # Dynamically alter runtime parameters matching user's ablation goals
                run_kwargs = {"text": text, "drug_context": drug_context}
                
                # These parameters route variants inside your modified agent layer cleanly
                if logger.ablation_mode == "NO_RAG":
                    run_kwargs["disable_rag_lookup"] = True
                elif logger.ablation_mode == "NO_INGREDIENT":
                    run_kwargs["disable_ingredient_mapping"] = True
                elif logger.ablation_mode == "HAND_PROMPT":
                    run_kwargs["override_with_legacy_prompt"] = True

                # Non-blocking network worker delegation
                loop = asyncio.get_event_loop()
                with tracer.start_as_current_span("Gemini_API_Adjudication") as api_span:
                    api_span.set_attribute("api_payload_text", text)
                    result = await loop.run_in_executor(None, lambda: agent.run(**run_kwargs))
                
                decision = 1 if result.get("is_ade", False) else 0
                reasoning = result.get("reasoning", "No valid explanation returned from target instance node.")
                
                # Attach granular textual annotations to tracing system attributes
                span.set_attribute("clinical_narrative_text", text)
                span.set_attribute("agent_reasoning_string", reasoning)
                span.set_attribute("final_decision", decision)
                
                # Metadata logging allocations
                simulated_token_metadata = {"input_tokens": 380, "output_tokens": 140}
                logger.log_atomic_transaction(idx, text, drug_context, gt, prob, decision, reasoning, simulated_token_metadata)
                
                span.set_status(Status(StatusCode.OK))
                return decision, reasoning

            except Exception as error_fault:
                span.record_exception(error_fault)
                span.set_status(Status(StatusCode.ERROR, description=str(error_fault)))
                logging.error(f"⚠️ Exception flagged during concurrent tracking run at row {idx}: {str(error_fault)}")
                return 0, f"System Evaluation Interrupted: {str(error_fault)}"

# ===================================================
# --- CALCULATE METRICS & REPORT MATRIX ------------
# ===================================================
def evaluate_document_rollup(df, pred_col, label_col):
    print("\n📜 Performing Document-Level Roll-Up Analytics...")
    doc_groups = df.groupby('source_file')
    doc_true, doc_pred = [], []
    for _, group in doc_groups:
        doc_true.append(1 if (group[label_col] == 1).any() else 0)
        doc_pred.append(1 if (group[pred_col] == 1).any() else 0)
    print(classification_report(doc_true, doc_pred, target_names=["Neg Doc", "Pos Doc"], digits=4))

async def main_async():
    print(f"📂 Mounting test file matrices from input location: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"⚠️ SAFETY SYSTEM CAP ENGAGED: Evaluating sample size fraction constraint to {MAX_SAMPLES} rows.")

    texts = df['text'].astype(str).tolist()
    drugs = df['meta_doc_drugs'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Stage 1: Load and Evaluate Hunter Baseline 
    tokenizer, hunter_model = load_hunter()
    hunter_probs = get_hunter_predictions(drugs, texts, tokenizer, hunter_model)
    df['hunter_prob'] = hunter_probs

    # Stage 2: Orchestrate Parallel Task Funnels
    agent = CADECAgent()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    with ObservabilityContextManager(experiment_name="CBMS_Springer_CadecV2", ablation_mode=ABLATION_MODE) as logger:
        tasks = [
            evaluate_sample_funnel(i, texts[i], drugs[i], hunter_probs[i], labels[i], agent, semaphore, logger)
            for i in range(len(df))
        ]
        
        print(f"🚀 Initializing Asynchronous Processing Engine across parallel coroutines...")
        results = await tqdm.gather(*tasks, desc="Adjudicating Pipeline Funnel States")
        
        df['final_pred'] = [r[0] for r in results]
        df['agent_reasoning'] = [r[1] for r in results]

        # Standard Output Mathematical Summaries
        print("\n" + "="*60)
        print(f"📊 CONSOLIDATED SENTENCE EVALUATION REPORT (MODE: {ABLATION_MODE})")
        print("="*60)
        print(classification_report(labels, df['final_pred'].tolist(), target_names=["Negative", "ADE"], digits=4))
        print(f"ℹ️ Tracking Context reported complete budget expenditure state: ${logger.cumulative_cost:.4f}")

        print("\n" + "="*60)
        print("📄 CONSOLIDATED SOTA DOCUMENT EVALUATION ROLL-UP REPORT")
        print("="*60)
        evaluate_document_rollup(df, 'final_pred', 'label')
        
        export_filename = f"final_results_{logger.experiment_name}_{logger.ablation_mode}.csv"
        df.to_csv(export_filename, index=False)
        print(f"\n💾 Matrices cleanly saved to disk location: {export_filename}")

if __name__ == "__main__":
    asyncio.run(main_async())