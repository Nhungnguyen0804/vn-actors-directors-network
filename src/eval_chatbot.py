import time
import re
import json
import csv
import math
import logging

from networkx.readwrite import json_graph
import networkx as nx
from pathlib import Path
import sys
from typing import List, Dict, Tuple
from collections import defaultdict


from model_v2 import get_answer, load_llm_model 


try:
    from load_graph import load_graphs
except ImportError:
    try:
        from .load_graph import load_graphs
    except ImportError:
        def load_graphs():
            return nx.Graph(), nx.Graph()
# Load graphs
G_actor_collab, G_bipartite = load_graphs()
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.log"),
        logging.StreamHandler()
    ]
)

# Helper: Determine intent from question (rule-based; expand as needed)
def determine_intent(question: str) -> str:
    question_lower = question.lower()
    if 'other actors in' in question_lower or 'diễn viên khác trong' in question_lower:
        return 'actor_via_movie'
    elif 'other movies of' in question_lower or 'phim khác của' in question_lower:
        return 'movie_via_actor'
    elif 'collaborators of' in question_lower or 'hợp tác với' in question_lower:
        return 'actor_via_collaboration'
    elif 'bridge actors between' in question_lower or 'cầu nối giữa' in question_lower:
        return 'indirect_collaboration'
    elif 'movies with actors from' in question_lower or 'phim chung với' in question_lower:
        return 'movie_chain'
    elif 'spouse' in question_lower or 'vợ/chồng' in question_lower:
        return 'spouse'
    elif 'common movies' in question_lower or 'phim chung' in question_lower:
        return 'common_movies'
    # Add more for basic queries (e.g., 'movies by actor' -> 'movies_by_actor')
    return 'unknown'

print(">>> Loading Model for Evaluation...")
LLM_PACK = load_llm_model(use_finetuned=False)  # Hoặc False tùy nhu cầu
print(">>> Model Loaded!")

def chatbot(question):
    """
    Hàm wrapper gọi pipeline của model_v1
    """
    try:
        # Gọi pipeline get_answer. debug=False để log sạch hơn
        response = get_answer(question, LLM_PACK, use_finetuned=False, debug=False)
        return response
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
        return "Error"
    
    
def call_graphrag(batch) -> str:
    answers = []
    for item in batch:
        question = item["question"]
        options = item["options"]
        answer = chatbot(question)
        answers.append(answer)
    formatted = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(answers))
    return formatted


def normalize_answer(output: str, is_mcq: bool) -> Tuple[str, bool]:
    text = output.strip().upper()
   
    if is_mcq:
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1], True
        return "", False
    else:
        match = re.search(r'\b(TRUE|FALSE|YES|NO|T|F)\b', text)
        if match:
            ans = match.group(1)
            mapping = {"YES": "True", "NO": "False", "T": "True", "F": "False"}
            normalized = mapping.get(ans, ans).capitalize()
            return normalized, True
        return "", False
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
    answers = [""] * batch_size
    lines = response.strip().split('\n')
   
    for line in lines:
        match = re.match(r'^\s*(\d+)\.\s*(.+)', line.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < batch_size:
                answers[idx] = match.group(2).strip()
                continue
       
        match = re.match(r'(?:Answer\s+)?(\d+)[:\.]\s*(.+)', line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < batch_size:
                answers[idx] = match.group(2).strip()
   
    if not any(answers): 
        all_matches = re.findall(r'(\d+)\.\s*([A-D]|TRUE|FALSE|YES|NO|T|F)', response.upper(), re.IGNORECASE)
        for idx_str, ans in all_matches:
            idx = int(idx_str) - 1
            if 0 <= idx < batch_size:
                answers[idx] = ans
   
    return answers
def compute_prf(results: List[Dict]):
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
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        metrics[lab] = {"precision":prec,"recall":rec,"f1":f1,"tp":tp,"fp":fp,"fn":fn}
        sum_tp += tp; sum_fp += fp; sum_fn += fn
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp+sum_fp)>0 else 0.0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp+sum_fn)>0 else 0.0
    micro_f1 = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec)>0 else 0.0
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics) if metrics else 0.0
    return {"per_label":metrics, "micro": {"precision":micro_prec,"recall":micro_rec,"f1":micro_f1}, "macro_f1":macro_f1}
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
    return {"labels":opts, "matrix":mat}
def calculate_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    parse_success = sum(1 for r in results if r["parse_success"])
   
    mcq_correct = sum(1 for r in results if r["is_mcq"] and r["correct"])
    mcq_total = sum(1 for r in results if r["is_mcq"])
    tf_correct = sum(1 for r in results if not r["is_mcq"] and r["correct"])
    tf_total = sum(1 for r in results if not r["is_mcq"])
   
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

def eval_graphrag(dataset: List[Dict], batch_size: int = 1) -> Dict:
    model_name = "graphrag"
    call_fn = call_graphrag  # GẮN HÀM 
   
    results = []
    all_latencies = []
    total_latency = 0
    num_batches = math.ceil(len(dataset) / batch_size)
   
    logging.info(f"Evaluating {model_name}: {len(dataset)} questions in {num_batches} batches")
   
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
       
        # # prompt = format_batch_prompt(batch)
        # prompt = batch       # truyền list dict
        

        retry_count = 0
        max_retries = 2 
        while retry_count < max_retries:
            start_time = time.time()
            try:
                raw_output = call_fn(batch)
                # print(raw_output)
                latency = time.time() - start_time
                total_latency += latency
                break 
            except Exception as e:
                logging.exception(f"Batch {batch_idx + 1} failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    raw_output = ""
                    latency = 0
        all_latencies.append(latency)
        logging.info(f"Batch {batch_idx + 1}/{num_batches} - Latency: {latency:.2f}s")
       
        parsed_answers = parse_batch_response(raw_output, len(batch))
       
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
   
    metrics = calculate_metrics(results)
   
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
   
    with open(f"data/eval/{model_name}_detailed_metrics.json", "w", encoding="utf-8") as outf:
        json.dump(metrics, outf, ensure_ascii=False, indent=2)
   
    csv_filename = f"{model_name}_evaluation_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question_id", "question", "is_mcq", "gold_answer",
                      "model_raw_output", "model_normalized_answer",
                      "parse_success", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
   
    logging.info(f"Results saved to {csv_filename}")
    return metrics

# TEST 1 CÁI 
# sample_prompt = [
#     {
#         "question": "Đạo diễn ...?",
#         "options": ["A...", "B...", "C...", "D..."]
#     }
# ]

# print(call_graphrag(sample_prompt))





if __name__ == "__main__":
    DATASET = f"data/eval/evaluation_dataset.json"
   
    with open(DATASET, "r", encoding="utf-8") as f:
        QA_dict = json.load(f)
   
    metrics = eval_graphrag(
        dataset=QA_dict,
        batch_size=1  
    )
   
    print("\nEvaluation complete! Check data/eval/graphrag_detailed_metrics.json")