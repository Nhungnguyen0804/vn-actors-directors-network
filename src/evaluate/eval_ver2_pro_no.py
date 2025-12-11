import google.generativeai as genai
import time
import re
import json
import csv
import math
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.log"),
        logging.StreamHandler()
    ]
)

# ================================
# 1) MODEL WRAPPERS
# ================================
def call_graphrag(prompt: str) -> str:
    """
    TODO: Implement actual GraphRAG call
    Hiện tại đang mock, cần thay thế bằng API thực
    """
    # Example implementation:
    # response = requests.post("http://your-graphrag-api/query", json={"prompt": prompt})
    # return response.json()["answer"]
    raise NotImplementedError("GraphRAG implementation needed!")

def call_gemini(prompt: str, max_retries: int = 5) -> str:
    with open("key_pro.txt", "r") as f:
        api_key = f.read().strip()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            time.sleep(1)  # Pro tier có thể giảm xuống 1s
            return response.text
        except Exception as e:
            if '429' in str(e) or 'quota' in str(e).lower():
                wait_time = 2 ** retry_count
                logging.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
            else:
                logging.error(f"Unexpected error: {e}")
                raise
    raise ValueError("Max retries exceeded for Gemini API call.")

# ================================
# 2) ANSWER NORMALIZATION
# ================================
def normalize_answer(output: str, is_mcq: bool) -> Tuple[str, bool]:
    """
    Returns: (normalized_answer, parse_success)
    """
    text = output.strip().upper()
    
    if is_mcq:
        # Tìm tất cả A/B/C/D, lấy cái cuối cùng
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1], True
        return "", False
    else:
        # Mở rộng cho True/False, thêm synonym
        match = re.search(r'\b(TRUE|FALSE|YES|NO|T|F)\b', text)
        if match:
            ans = match.group(1)
            mapping = {"YES": "True", "NO": "False", "T": "True", "F": "False"}
            normalized = mapping.get(ans, ans).capitalize()
            return normalized, True
        return "", False
    
# def normalize_answer(output: str, is_mcq: bool) -> Tuple[str, bool]:
#     """
#     Returns: (normalized_answer, parse_success)
#     """
#     text = output.strip().upper()
    
#     if is_mcq:
#         # Tìm tất cả A/B/C/D, lấy cái cuối cùng
#         matches = re.findall(r'\b([A-D])\b', text)
#         if matches:
#             return matches[-1], True
#         return "", False
#     else:
#         match = re.search(r'\b(TRUE|FALSE)\b', text)
#         if match:
#             return match.group(1).capitalize(), True
#         return "", False

# ================================
# 3) BATCH PROMPTING
# ================================
def format_batch_prompt(batch: List[Dict]) -> str:
    prompt_parts = []
    for idx, q in enumerate(batch, 1):
        if "options" in q:
            opts = "\n".join(q["options"])
            prompt_parts.append(
                f"Question {idx}: {q['question']}\n"
                f"Options:\n{opts}\n"
                f"Answer: (A/B/C/D only)"
            )
        else:
            prompt_parts.append(
                f"Question {idx}: {q['question']}\n"
                f"Answer: (True/False only)"
            )
    
    prompt = f"""Answer ALL {len(batch)} questions below in EXACT format:
1. <A/B/C/D or True/False>
2. <A/B/C/D or True/False>
...
{len(batch)}. <A/B/C/D or True/False>

NO explanations. NO extra text. ONLY numbered answers.

{'='*50}
""" + "\n\n".join(prompt_parts)
    return prompt


def parse_batch_response(response: str, batch_size: int) -> List[str]:
    """
    Robust parser cho batch responses
    """
    answers = [""] * batch_size
    lines = response.strip().split('\n')
    
    for line in lines:
        # Pattern 1: "1. A" hoặc "1. True"
        match = re.match(r'^\s*(\d+)\.\s*(.+)', line.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < batch_size:
                answers[idx] = match.group(2).strip()
                continue
        
        # Pattern 2: "Answer 1: A"
        match = re.match(r'(?:Answer\s+)?(\d+)[:\.]\s*(.+)', line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < batch_size:
                answers[idx] = match.group(2).strip()
    
    # Fallback nếu parse fail: Extract từ toàn bộ text
    if not any(answers):  # Nếu tất cả rỗng
        all_matches = re.findall(r'(\d+)\.\s*([A-D]|TRUE|FALSE|YES|NO|T|F)', response.upper(), re.IGNORECASE)
        for idx_str, ans in all_matches:
            idx = int(idx_str) - 1
            if 0 <= idx < batch_size:
                answers[idx] = ans
    
    return answers


# def parse_batch_response(response: str, batch_size: int) -> List[str]:
#     """
#     Robust parser cho batch responses
#     """
#     answers = [""] * batch_size
#     lines = response.strip().split('\n')
    
#     for line in lines:
#         # Pattern 1: "1. A" hoặc "1. True"
#         match = re.match(r'^\s*(\d+)\.\s*(.+)', line.strip())
#         if match:
#             idx = int(match.group(1)) - 1
#             if 0 <= idx < batch_size:
#                 answers[idx] = match.group(2).strip()
#                 continue
        
#         # Pattern 2: "Answer 1: A"
#         match = re.match(r'(?:Answer\s+)?(\d+)[:\.]\s*(.+)', line.strip(), re.IGNORECASE)
#         if match:
#             idx = int(match.group(1)) - 1
#             if 0 <= idx < batch_size:
#                 answers[idx] = match.group(2).strip()
    
#     return answers

# ================================
# 4) METRICS CALCULATION
# ================================

from collections import Counter, defaultdict
import math
import random

# 1) Precision / Recall / F1 (binary for TF, per-class for MCQ)
def compute_prf(results: List[Dict]):
    # results: mỗi item có gold_answer (str), model_normalized_answer (str), parse_success (bool)
    labels = set()
    for r in results:
        labels.add(r["gold_answer"])
    labels = sorted(labels)

    per_label = {lab: {"tp":0,"fp":0,"fn":0} for lab in labels}
    for r in results:
        if not r["parse_success"]:
            continue
        gold = r["gold_answer"]
        pred = r["model_normalized_answer"]
        if pred == gold:
            per_label[gold]["tp"] += 1
        else:
            if pred in per_label:
                per_label[pred]["fp"] += 1
            per_label[gold]["fn"] += 1

    metrics = {}
    sum_tp = sum_fp = sum_fn = 0
    for lab, c in per_label.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        metrics[lab] = {"precision":prec,"recall":rec,"f1":f1,"tp":tp,"fp":fp,"fn":fn}
        sum_tp += tp; sum_fp += fp; sum_fn += fn

    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp+sum_fp)>0 else 0.0
    micro_rec  = sum_tp / (sum_tp + sum_fn) if (sum_tp+sum_fn)>0 else 0.0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec)>0 else 0.0

    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics) if metrics else 0.0

    return {"per_label":metrics, "micro": {"precision":micro_prec,"recall":micro_rec,"f1":micro_f1}, "macro_f1":macro_f1}

# 2) Latency stats (pass a list of latencies)
def latency_stats(latencies: List[float]):
    if not latencies:
        return {}
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    def pct(p): return lat_sorted[max(0, min(n-1, int(math.floor(p*n/100))))] 
    return {
        "count": n,
        "mean": sum(lat_sorted)/n,
        "min": lat_sorted[0],
        "max": lat_sorted[-1],
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p95": pct(95)
    }

# 3) MCQ confusion matrix (A-D -> A-D)
def mcq_confusion_matrix(results: List[Dict], options=("A","B","C","D")):
    opts = list(options)
    idx = {o:i for i,o in enumerate(opts)}
    n = len(opts)
    mat = [[0]*n for _ in range(n)]
    for r in results:
        if not r["is_mcq"] or not r["parse_success"]:
            continue
        g = r["gold_answer"]
        p = r["model_normalized_answer"]
        if g in idx and p in idx:
            mat[idx[g]][idx[p]] += 1
    # return as dict for readability
    return {"labels":opts, "matrix":mat}

# 4) Bootstrap significance test (difference in accuracy between two result lists)
def bootstrap_accuracy_diff(a_results: List[Dict], b_results: List[Dict], n_bootstrap: int = 1000, seed: int = 42):
    # require same ordering and same length
    assert len(a_results) == len(b_results)
    random.seed(seed)
    n = len(a_results)
    diffs = []
    for _ in range(n_bootstrap):
        idxs = [random.randrange(n) for _ in range(n)]
        a_acc = sum(1 for i in idxs if a_results[i]["correct"]) / n
        b_acc = sum(1 for i in idxs if b_results[i]["correct"]) / n
        diffs.append(a_acc - b_acc)
    diffs.sort()
    lower = diffs[int(0.025 * n_bootstrap)]
    upper = diffs[int(0.975 * n_bootstrap)]
    mean_diff = sum(diffs)/len(diffs)
    return {"mean_diff": mean_diff, "95ci": (lower, upper)}



def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Tính toán các metrics chi tiết
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    parse_success = sum(1 for r in results if r["parse_success"])
    
    # Error breakdown by question type
    mcq_correct = sum(1 for r in results if r["is_mcq"] and r["correct"])
    mcq_total = sum(1 for r in results if r["is_mcq"])
    tf_correct = sum(1 for r in results if not r["is_mcq"] and r["correct"])
    tf_total = sum(1 for r in results if not r["is_mcq"])
    
    # Confusion matrix (for True/False)
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in results:
        if not r["is_mcq"] and r["parse_success"]:
            gold = r["gold_answer"]
            pred = r["model_normalized_answer"]
            if gold == "True" and pred == "True":
                confusion["TP"] += 1
            elif gold == "False" and pred == "False":
                confusion["TN"] += 1
            elif gold == "False" and pred == "True":
                confusion["FP"] += 1
            elif gold == "True" and pred == "False":
                confusion["FN"] += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0,
        "parse_rate": parse_success / total if total > 0 else 0,
        "mcq_accuracy": mcq_correct / mcq_total if mcq_total > 0 else 0,
        "tf_accuracy": tf_correct / tf_total if tf_total > 0 else 0,
        "confusion_matrix": confusion,
        "total_questions": total,
        "correct_answers": correct
    }

# ================================
# 5) EVALUATE MODEL
# ================================
import math
import time
import json
import csv
import logging

# Assuming other required functions are defined: format_batch_prompt, parse_batch_response, normalize_answer, call_gemini, call_graphrag, calculate_metrics, compute_prf, latency_stats, mcq_confusion_matrix

def evaluate_model(model_name: str, dataset: List[Dict], batch_size: int = 1):
    """
    Đánh giá model với batch processing
    """
    if model_name == "gemini":
        call_fn = call_gemini
    elif model_name == "graphrag":
        call_fn = call_graphrag
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    results = []
    all_latencies = []
    total_latency = 0
    num_batches = math.ceil(len(dataset) / batch_size)
    
    logging.info(f"Evaluating {model_name}: {len(dataset)} questions in {num_batches} batches")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        
        # Format and call with retry
        prompt = format_batch_prompt(batch)
        retry_count = 0
        max_retries = 2  # Thêm retry cho batch
        while retry_count < max_retries:
            start_time = time.time()
            try:
                raw_output = call_fn(prompt)
                latency = time.time() - start_time
                total_latency += latency
                break  # Success
            except Exception as e:
                logging.exception(f"Batch {batch_idx + 1} failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    raw_output = ""
                    latency = 0
        all_latencies.append(latency)
        logging.info(f"Batch {batch_idx + 1}/{num_batches} - Latency: {latency:.2f}s")
        
        # Parse answers
        parsed_answers = parse_batch_response(raw_output, len(batch))
        
        # Process each question
        for q_idx, q in enumerate(batch):
            is_mcq = "options" in q
            raw_ans = parsed_answers[q_idx]
            model_ans, parse_ok = normalize_answer(raw_ans, is_mcq)
            gold = q["answer"]
            
            is_correct = (model_ans == gold) and parse_ok
            
            results.append({
                "question_id": start_idx + q_idx + 1,
                "question": q["question"],
                "is_mcq": is_mcq,
                "gold_answer": gold,
                "model_raw_output": raw_ans,
                "model_normalized_answer": model_ans,
                "parse_success": parse_ok,
                "correct": is_correct
            })
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # After building 'results':
    prf_metrics = compute_prf(results)
    lat_stats = latency_stats(all_latencies)
    mcq_conf = mcq_confusion_matrix(results)
    metrics.update({
        "prf": prf_metrics,
        "latency_stats": lat_stats,
        "mcq_confusion": mcq_conf,
        "model": model_name,
        "avg_latency_per_batch": total_latency / num_batches if num_batches > 0 else 0,
        "total_latency": total_latency,
        "total_batches": num_batches,
        "batch_size": batch_size
    })
    
    # Save full metrics to JSON
    with open(f"data/eval/{model_name}_detailed_metrics.json", "w", encoding="utf-8") as outf:
        json.dump(metrics, outf, ensure_ascii=False, indent=2)
    
    # Save detailed results
    csv_filename = f"{model_name}_evaluation_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question_id", "question", "is_mcq", "gold_answer",
                      "model_raw_output", "model_normalized_answer",
                      "parse_success", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logging.info(f"Results saved to {csv_filename}")
    return metrics, results

# def evaluate_model(model_name: str, dataset: List[Dict], batch_size: int = 1):
#     """
#     Đánh giá model với batch processing
#     """
#     metrics = {}
#     all_latencies = []
    
#     # after building 'results':
#     prf_metrics = compute_prf(results)
#     lat_stats = latency_stats(all_latencies)
#     mcq_conf = mcq_confusion_matrix(results)
#     metrics.update({
#         "prf": prf_metrics,
#         "latency_stats": lat_stats,
#         "mcq_confusion": mcq_conf
#     })
#     # Save full metrics to JSON (already saving comparison later)
#     with open(f"data/eval/{model_name}_detailed_metrics.json", "w", encoding="utf-8") as outf:
#         json.dump(metrics, outf, ensure_ascii=False, indent=2)

#     if model_name == "gemini":
#         call_fn = call_gemini
#     elif model_name == "graphrag":
#         call_fn = call_graphrag
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
    
#     results = []
#     total_latency = 0
#     num_batches = math.ceil(len(dataset) / batch_size)
    
#     logging.info(f"Evaluating {model_name}: {len(dataset)} questions in {num_batches} batches")
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, len(dataset))
#         batch = dataset[start_idx:end_idx]
        
#         # Format and call with retry
#         prompt = format_batch_prompt(batch)
#         retry_count = 0
#         max_retries = 2  # Thêm retry cho batch
#         while retry_count < max_retries:
#             start_time = time.time()
#             try:
#                 raw_output = call_fn(prompt)
#                 latency = time.time() - start_time
#                 total_latency += latency
#                 break  # Success
#             except Exception as e:
#                 logging.exception(f"Batch {batch_idx + 1} failed (attempt {retry_count + 1}): {e}")
#                 retry_count += 1
#                 if retry_count == max_retries:
#                     raw_output = ""
#                     latency = 0
#         all_latencies.append(latency)
#         logging.info(f"Batch {batch_idx + 1}/{num_batches} - Latency: {latency:.2f}s")
        
#         # Parse answers
#         parsed_answers = parse_batch_response(raw_output, len(batch))
        
#         # Process each question
#         for q_idx, q in enumerate(batch):
#             is_mcq = "options" in q
#             raw_ans = parsed_answers[q_idx]
#             model_ans, parse_ok = normalize_answer(raw_ans, is_mcq)
#             gold = q["answer"]
            
#             is_correct = (model_ans == gold) and parse_ok
            
#             results.append({
#                 "question_id": start_idx + q_idx + 1,
#                 "question": q["question"],
#                 "is_mcq": is_mcq,
#                 "gold_answer": gold,
#                 "model_raw_output": raw_ans,
#                 "model_normalized_answer": model_ans,
#                 "parse_success": parse_ok,
#                 "correct": is_correct
#             })
    
#     # Calculate metrics
#     metrics = calculate_metrics(results)
#     metrics.update({
#         "model": model_name,
#         "avg_latency_per_batch": total_latency / num_batches if num_batches > 0 else 0,
#         "total_latency": total_latency,
#         "total_batches": num_batches,
#         "batch_size": batch_size
#     })
    
#     # Save detailed results
#     csv_filename = f"{model_name}_evaluation_results.csv"
#     with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
#         fieldnames = ["question_id", "question", "is_mcq", "gold_answer",
#                      "model_raw_output", "model_normalized_answer",
#                      "parse_success", "correct"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)
    
#     logging.info(f"Results saved to {csv_filename}")
#     return metrics, results

# def evaluate_model(model_name: str, dataset: List[Dict], batch_size: int = 1):
#     """
#     Đánh giá model với batch processing
    
#     Args:
#         model_name: "gemini" hoặc "graphrag"
#         dataset: List các câu hỏi
#         batch_size: Số câu hỏi mỗi batch
#     """
#     if model_name == "gemini":
#         call_fn = call_gemini
#     elif model_name == "graphrag":
#         call_fn = call_graphrag
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
    
#     results = []
#     total_latency = 0
#     num_batches = math.ceil(len(dataset) / batch_size)
    
#     logging.info(f"Evaluating {model_name}: {len(dataset)} questions in {num_batches} batches")
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, len(dataset))
#         batch = dataset[start_idx:end_idx]
        
#         # Format and call
#         prompt = format_batch_prompt(batch)
#         start_time = time.time()
        
#         try:
#             raw_output = call_fn(prompt)
#             latency = time.time() - start_time
#             total_latency += latency
#         except Exception as e:
#             logging.error(f"Batch {batch_idx + 1} failed: {e}")
#             # Fill with empty answers
#             raw_output = ""
#             latency = 0
        
#         logging.info(f"Batch {batch_idx + 1}/{num_batches} - Latency: {latency:.2f}s")
        
#         # Parse answers
#         parsed_answers = parse_batch_response(raw_output, len(batch))
        
#         # Process each question
#         for q_idx, q in enumerate(batch):
#             is_mcq = "options" in q
#             raw_ans = parsed_answers[q_idx]
#             model_ans, parse_ok = normalize_answer(raw_ans, is_mcq)
#             gold = q["answer"]
            
#             is_correct = (model_ans == gold) and parse_ok
            
#             results.append({
#                 "question_id": start_idx + q_idx + 1,
#                 "question": q["question"],
#                 "is_mcq": is_mcq,
#                 "gold_answer": gold,
#                 "model_raw_output": raw_ans,
#                 "model_normalized_answer": model_ans,
#                 "parse_success": parse_ok,
#                 "correct": is_correct
#             })
    
#     # Calculate metrics
#     metrics = calculate_metrics(results)
#     metrics.update({
#         "model": model_name,
#         "avg_latency_per_batch": total_latency / num_batches if num_batches > 0 else 0,
#         "total_latency": total_latency,
#         "total_batches": num_batches,
#         "batch_size": batch_size
#     })
    
#     # Save detailed results
#     csv_filename = f"{model_name}_evaluation_results.csv"
#     with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
#         fieldnames = ["question_id", "question", "is_mcq", "gold_answer", 
#                      "model_raw_output", "model_normalized_answer", 
#                      "parse_success", "correct"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)
    
#     logging.info(f"Results saved to {csv_filename}")
#     return metrics, results

# ================================
# 6) COMPARE MODELS
# ================================
# def compare_models(dataset: List[Dict], batch_size_gemini: int = 50, batch_size_graphrag: int = 1):
#     """
#     So sánh 2 models
#     """
#     print("\n" + "="*60)
#     print("EVALUATING GEMINI")
#     print("="*60)
#     gemini_metrics, gemini_results = evaluate_model("gemini", dataset, batch_size_gemini)
    
#     print("\n" + "="*60)
#     print("EVALUATING GRAPHRAG")
#     print("="*60)
#     try:
#         graphrag_metrics, graphrag_results = evaluate_model("graphrag", dataset, batch_size_graphrag)
#     except NotImplementedError:
#         print("GraphRAG chưa được implement! Bỏ qua...")
#         graphrag_metrics = None
#         graphrag_results = []
    
#     # Comparison report
#     comparison = {
#         "gemini": gemini_metrics,
#         "graphrag": graphrag_metrics,
#         "comparison": {
#             "accuracy_diff": gemini_metrics["accuracy"] - (graphrag_metrics["accuracy"] if graphrag_metrics else 0),
#             "speed_ratio": gemini_metrics["avg_latency_per_batch"] / (graphrag_metrics["avg_latency_per_batch"] if graphrag_metrics else 1)
#         }
#     }
    
#     # Save comparison
#     with open("data/eval/model_comparison.json", "w", encoding="utf-8") as f:
#         json.dump(comparison, f, ensure_ascii=False, indent=2)
    
#     print("\n" + "="*60)
#     print("COMPARISON SUMMARY")
#     print("="*60)
#     print(f"Gemini Accuracy: {gemini_metrics['accuracy']:.2%}")
#     if graphrag_metrics:
#         print(f"GraphRAG Accuracy: {graphrag_metrics['accuracy']:.2%}")
#         print(f"Winner: {'Gemini' if comparison['comparison']['accuracy_diff'] > 0 else 'GraphRAG'}")
    
#     return comparison
# ver1 

# ver 2: 
def compare_models(dataset: List[Dict], batch_size_gemini: int = 50, batch_size_graphrag: int = 1):
    print("\n" + "="*60)
    print("EVALUATING GEMINI")
    print("="*60)
    gemini_metrics, gemini_results = evaluate_model("gemini", dataset, batch_size_gemini)

    print("\n" + "="*60)
    print("EVALUATING GRAPHRAG")
    print("="*60)
    try:
        graphrag_metrics, graphrag_results = evaluate_model("graphrag", dataset, batch_size_graphrag)
    except NotImplementedError:
        print("GraphRAG chưa được implement! Bỏ qua...")
        graphrag_metrics = None
        graphrag_results = []

    comparison = {
        "gemini": {
            "metrics": gemini_metrics,
            "results": gemini_results
        },
        "graphrag": {
            "metrics": graphrag_metrics,
            "results": graphrag_results
        }
    }

    # Only compute diff if graphrag exists
    if graphrag_metrics is not None:
        comparison["comparison"] = {
            "accuracy_diff": gemini_metrics["accuracy"] - graphrag_metrics["accuracy"],
            "speed_ratio": gemini_metrics["avg_latency_per_batch"] / graphrag_metrics["avg_latency_per_batch"]
        }
    else:
        comparison["comparison"] = {
            "accuracy_diff": None,
            "speed_ratio": None
        }

    with open("data/eval/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Gemini Accuracy: {gemini_metrics['accuracy']:.2%}")
    if graphrag_metrics is not None:
        print(f"GraphRAG Accuracy: {graphrag_metrics['accuracy']:.2%}")
        print(f"Winner: {'Gemini' if gemini_metrics['accuracy'] >= graphrag_metrics['accuracy'] else 'GraphRAG'}")
    else:
        print("GraphRAG: Không có dữ liệu (Not implemented)")

    return comparison


# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    DATASET = "data/evaluation_dataset.json"
    
    with open(DATASET, "r", encoding="utf-8") as f:
        QA_dict = json.load(f)
    
    # Với Pro key và 2000 QA:
    # - Batch size 50: 40 batches, ~40-80 giây tổng
    # - Batch size 100: 20 batches, ~20-40 giây tổng
    
    comparison = compare_models(
        dataset=QA_dict,
        batch_size_gemini=100,  # Recommended cho Pro key
        batch_size_graphrag=1   # GraphRAG thường không hỗ trợ batch
    )
    
    print("\nEvaluation complete! Check data/eval/model_comparison.json")