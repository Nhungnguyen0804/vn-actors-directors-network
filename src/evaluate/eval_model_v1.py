
import google.generativeai as genai

with open("key.txt", "r") as f:
    api_key = f.read().strip()

import time
import re
import json
import csv
import math
import logging
from typing import List, Dict
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
# 1) WRAPPER FOR MODELS
# ================================
# A. GRAPH RAG CALL FUNCTION
def call_graphrag(prompt: str) -> str:

    return "demo"  # Replace with actual implementation

# B. GEMINI FLASH CALL FUNCTION
def call_gemini(prompt: str, max_retries: int = 5) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            time.sleep(12)  # Free tier: 5 requests/minute => 12s/request
            return response.text
        except Exception as e:
            if '429' in str(e) or 'quota' in str(e).lower():
                logging.warning(f"Rate limit or quota error: {e}. Retrying in {2 ** retry_count} seconds...")
                time.sleep(2 ** retry_count)
                retry_count += 1
            else:
                logging.error(f"Unexpected error: {e}")
                raise
    raise ValueError("Max retries exceeded for Gemini API call.")

# ================================
# 2) NORMALIZE MODEL ANSWER
# ================================
def normalize_answer(output: str, is_mcq: bool):
    text = output.strip().upper()
    if is_mcq:
        # Extract last standalone A/B/C/D using regex for robustness
        # match = re.search(r'\b([A-D])\b', text[::-1])  # Search from end
        # if match:
        #     return match.group(1)[::-1]  # Reverse back
        # return ""  # If no match
        match = re.findall(r"\b([A-D])\b", text)
        if match:
            return match[-1]
        return ""

    else:  # True/False
        match = re.search(r'\b(TRUE|FALSE)\b', text)
        if match:
            return match.group(1).capitalize()
        return ""



# ================================
# 3) BATCH PROMPT FORMATTING
# ================================
def format_batch_prompt(batch: List[Dict]) -> str:
    prompt_parts = []
    for idx, q in enumerate(batch, 1):
        if "options" in q:
            opts = "\n".join(q["options"])
            prompt_parts.append(f"Question {idx}: {q['question']}\nOptions:\n{opts}\nAnswer with a single character (A/B/C/D).")
        else:
            prompt_parts.append(f"Question {idx}: {q['question']}\nAnswer with True or False only.")
    prompt = f"""
Answer all questions below. Format your response exactly as:
1. <answer>
2. <answer>
...
{len(batch)}. <answer>

Do not add extra text or explanations.

{'-' * 40}\n""" + "\n\n".join(prompt_parts)
    return prompt

# ================================
# 4) PARSE BATCH RESPONSE
# ================================
def parse_batch_response(response: str, batch_size: int) -> List[str]:
    answers = [""] * batch_size
    lines = response.strip().split('\n')
    for line in lines:
        match = re.match(r'(\d+)\.\s*(.+)', line.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < batch_size:
                answers[idx] = match.group(2).strip()
    return answers


# ================================
# 3) EVALUATE SINGLE MODEL
# ================================

def evaluate_model(model_name: str, dataset: List[Dict], batch_size: int = 100): # batch size 10 => 200 batch, 100 => 20 batch
    if model_name != "gemini":
        raise ValueError("This batched evaluation is focused on Gemini only.")
    
    correct = 0
    total_latency = 0
    results = []  # For CSV logging
    
    # Split dataset into batches
    num_batches = math.ceil(len(dataset) / batch_size)
    logging.info(f"Evaluating {len(dataset)} questions in {num_batches} batches of size {batch_size}.")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        
        # Format batch prompt
        prompt = format_batch_prompt(batch)
        
        # Call Gemini with retry
        start = time.time()
        raw_output = call_gemini(prompt)
        latency = time.time() - start
        total_latency += latency
        
        logging.info(f"Batch {batch_idx + 1}/{num_batches} processed. Latency: {latency:.2f}s")
        logging.debug(f"Raw output for batch {batch_idx + 1}:\n{raw_output}")
        
        # Parse answers
        parsed_answers = parse_batch_response(raw_output, len(batch))
        
        # Process each question in batch
        for q_idx, q in enumerate(batch):
            is_mcq = "options" in q
            raw_ans = parsed_answers[q_idx]
            model_ans = normalize_answer(raw_ans, is_mcq)
            gold = q["answer"]
            
            is_correct = model_ans == gold
            if is_correct:
                correct += 1
            
            # Log to results
            results.append({
                "question_id": start_idx + q_idx + 1,
                "question": q["question"],
                "gold_answer": gold,
                "model_raw_output": raw_ans,
                "model_normalized_answer": model_ans,
                "correct": is_correct
            })
    
    accuracy = correct / len(dataset) if len(dataset) > 0 else 0
    avg_latency = total_latency / num_batches if num_batches > 0 else 0  # Per batch, since requests are batched

    # Save to CSV
    csv_filename = "data/eval/gemini_evaluation_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question_id", "question", "gold_answer", "model_raw_output", "model_normalized_answer", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"Results saved to {csv_filename}")
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_latency_per_batch": avg_latency,
        "total_questions": len(dataset),
        "total_batches": num_batches
    }

# ================================
# MAIN EXECUTION
# ================================
DATASET = "data/evaluation_dataset.json"

with open(DATASET, "r", encoding="utf-8") as f:
    QA_dict = json.load(f)




def eval_gemini():
    result = evaluate_model("gemini", QA_dict)
    print(result)

    OUTPUT_FILE = "data/eval/eval_gem.json"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Đã xuất res ra", OUTPUT_FILE)


# eval_gemini()