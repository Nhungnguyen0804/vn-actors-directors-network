import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== IMPORT MODULES ====================
try:
    from src.chatbot.entity_linking_node import normalize_text, entity_linking_graph
    from src.nlp.ner import extract_entity_from_sentences
    from src.chatbot.graph_query import (
        build_query_from_relationships, query_flexible, 
        close_driver
    )
    from src.intent_pattern import INTENT_PATTERNS, FUNC_MAP
except ImportError as e:
    print(f"Warning: Could not import modules - {e}")


# ==================== 1. LOAD LLM ====================

def load_llm_model(lora_path="./outputs_graphrag_lora", fine_tune=False):
    if not os.path.exists("offload_weights"):
        os.makedirs("offload_weights")

    print("Dang khoi tao model...")
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", offload_folder="offload_weights",
            trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )

        if not fine_tune:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            print("Tai thanh cong Base Model!")
            return model, tokenizer

        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, lora_path, offload_folder="offload_weights")
        print("Tai thanh cong! Model dang chay voi LoRA adapter.")
        return model, tokenizer

    except Exception as e:
        print(f"\nError: Loi tai model: {e}")
        sys.exit(1)



def classify_intent(question, entity_type="PERSON", num_entities=1, debug=False):
    """Classify intent with proper priority handling"""
    q_lower = normalize_text(question)
    
    
    for intent_name, config in INTENT_PATTERNS.items():
        # Skip if intent requires 2 entities but we only have 1
        if config.get("needs_2_entities") and num_entities < 2:
            continue
        
       
        if config.get("max_entities") and num_entities > config["max_entities"]:
            if debug:
                print(f"[SKIP] {intent_name} - requires max {config['max_entities']} entities, got {num_entities}")
            continue
        
        # Skip 1-entity intents if we have 2+ entities (unless it's a spouse query with max_entities=1)
        if num_entities >= 2 and not config.get("needs_2_entities") and not config.get("max_entities"):
            if len(config.get("rels", [])) <= 1:
                continue
        
        for pattern in config["patterns"]:
            if re.search(pattern, q_lower):
                # Adjust for entity type
                adjusted_config = config.copy()
                if entity_type == "FILM" and "rels" in config:
                    rels = config["rels"]
                    adjusted_rels = []
                    for r in rels:
                        if r == 'PERSON_ACTED_IN_FILM':
                            adjusted_rels.append('FILM_HAS_ACTOR')
                        elif r == 'PERSON_DIRECTED_FILM':
                            adjusted_rels.append('FILM_HAS_DIRECTOR')
                        else:
                            adjusted_rels.append(r)
                    adjusted_config["rels"] = adjusted_rels
                
                if debug:
                    print(f"[INTENT] Matched: {intent_name}")
                    if "rels" in adjusted_config:
                        print(f"[INTENT] Relationships: {adjusted_config['rels']}")
                    elif "func" in adjusted_config:
                        print(f"[INTENT] Function: {adjusted_config['func']}")
                
                return intent_name, adjusted_config
    
    # Fallback
    if num_entities >= 2:
        return "get_common_movies", INTENT_PATTERNS["get_common_movies"]
    return "get_general_info", INTENT_PATTERNS["get_general_info"]


# ==================== 4. QUERY EXECUTOR ====================

def execute_query(entity_name, entity_type, intent_name, config, entity_name_2=None, debug=False):
    """Execute graph query based on config"""
    
    # Custom function
    if "func" in config:
        func = FUNC_MAP[config["func"]]
        if config.get("needs_2_entities"):
            if not entity_name_2:
                return None
            return func(entity_name, entity_name_2, debug=debug)
        return func(entity_name, debug=debug)
    
    # Relationship-based query
    if "rels" in config:
        rels = config["rels"]
        if len(rels) == 1:
            result = build_query_from_relationships(
                entity_name, entity_type, rels, limit=20, debug=debug
            )
            return result
        else:
            return query_flexible(entity_name, entity_type, rels, debug=debug)
    
    return None


# ==================== 5. FORMATTER ====================

def format_result(data, intent_name, entity_name=None, entity_name_2=None):
    """Format graph data for display"""
    if not data:
        return "KHONG TIM THAY thong tin."
    
    
    if intent_name == "get_spouse_info":
        if isinstance(data, list) and data:
            spouse_name = data[0] if isinstance(data[0], str) else data[0].get('name', data[0])
            return f"VO/CHONG: {spouse_name}."
        elif isinstance(data, str):
            return f"VO/CHONG: {data}."
    
    #
    if intent_name == "get_common_movies" and entity_name_2:
        if isinstance(data, list) and data:
            items_str = ", ".join(str(x) for x in data)
            return f"PHIM CHUNG cua {entity_name} va {entity_name_2}: {items_str}."
    
    # Handle dict (node properties)
    if isinstance(data, dict):
        if 'properties' in data or 'info_name' in data:
            props = data.get('properties', data)
            parts = [f"Ten: {props.get('info_name') or entity_name}"]
            key_map = {
                'info_birth_date': 'Nam sinh', 'info_birth_place': 'Noi sinh',
                'info_occupation': 'Nghe nghiep', 'info_spouse': 'Vo/Chong'
            }
            for k, v in props.items():
                if k in key_map and v:
                    parts.append(f"{key_map[k]}: {v}")
            return "THONG TIN: " + ". ".join(parts) + "."
    
    # Handle list
    if isinstance(data, list):
        if not data:
            return "KHONG TIM THAY ket qua."
        
        # Complex structures
        if isinstance(data[0], dict):
            first = data[0]
            if 'director' in first and 'film' in first:
                items = [f"{d['film']} (dao dien: {d['director']})" for d in data]
                return f"DANH SACH: {', '.join(items)}."
            if 'actor' in first and 'film' in first:
                items = [f"{d['actor']} (phim: {d['film']})" for d in data]
                return f"DANH SACH: {', '.join(items)}."
        
        # Simple list
        items_str = ", ".join(str(x) for x in data)
        
        templates = {
            "get_movies_by_actor": f"PHIM: {items_str}.",
            "get_actors_of_movie": f"DIEN VIEN: {items_str}.",
            "get_director_of_movie": f"DAO DIEN: {items_str}.",
            "get_common_movies": f"PHIM: {items_str}.",
            "get_film_genre": f"THE LOAI: {items_str}.",
            "get_birthdate": f"NAM SINH: {items_str}.",
            "get_birthplace": f"QUE QUAN: {items_str}.",
            "get_occupation": f"NGHE NGHIEP: {items_str}.",
            "get_spouse_info": f"VO/CHONG: {items_str}.",
            "get_spouse_movies": f"PHIM cua VO/CHONG: {items_str}.",
        }
        
        return templates.get(intent_name, f"KET QUA: {items_str}.")
    
    return f"KET QUA: {str(data)}."

# ==================== 6. LLM PARAPHRASER ====================

def llm_paraphrase(model_pack, formatted, question, debug=False):
    model, tokenizer = model_pack
    if "KHONG TIM THAY" in formatted:
        return "Xin lỗi, tôi chưa có thông tin về câu hỏi này."
    
    # ✅ Prompt chặt chẽ hơn
    prompt = f"""Câu hỏi: {question}
Dữ liệu: {formatted}

Yêu cầu:
1. Chỉ viết lại dữ liệu trên thành câu văn TỰ NHIÊN
2. KHÔNG ĐƯỢC thêm bất kỳ thông tin nào không có trong "Dữ liệu"
3. KHÔNG ĐƯỢC bịa đặt tên người, năm, địa điểm

Câu trả lời:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    ids = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.0,  # Greedy 100%
        do_sample=False,
        repetition_penalty=1.3,  # Tăng lên
        no_repeat_ngram_size=3   # Thêm constraint
    )
    
    response = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    
    
    if any(banned in response.lower() for banned in ['năm', 'sinh năm', 'đạo diễn', 'tại']):
        if debug:
            print(f"[BLOCKED] Detected hallucination: {response}")
        return formatted
    
    return response
# ==================== 7. MAIN PIPELINE ====================

def get_answer(question, model_pack, use_finetuned=False, debug=False):
    """Main pipeline"""
    print(f"\n{'='*60}\nQUESTION: {question}")
    
    # 1. Extract entities
    entities = extract_entity_from_sentences(question)
    if not entities:
        return "Khong tim thay ten rieng."
    
    if debug:
        print(f"[1] Entities: {entities}")
    
    # 2. Link to graph
    linked = entity_linking_graph(question)
    if not linked:
        return "Khong tim thay thuc the trong Graph."
    
    entity_name = linked[0]['node_name']
    entity_type = linked[0]['type'].upper()
    entity_name_2 = linked[1]['node_name'] if len(linked) > 1 else None
    
    if debug:
        print(f"[2] Linked: {entity_name} ({entity_type})")
        if entity_name_2:
            print(f"    Entity 2: {entity_name_2}")
    
    # 3. Classify intent
    num_entities = len(linked)
    intent_name, config = classify_intent(
        question, entity_type, num_entities, debug
    )
    
    if debug:
        print(f"[3] Intent: {intent_name} (num_entities={num_entities})")
    
    # 4. Execute query
    data = execute_query(
        entity_name, entity_type, intent_name, config, entity_name_2, debug
    )
    
    if not data:
        return "Khong tim thay thong tin."
    
    if debug:
        print(f"[4] Data: {data}")
    
    # 5. Format
    formatted = format_result(data, intent_name, entity_name)
    
    if debug:
        print(f"[5] Formatted: {formatted}")
    
    # 6. Generate natural response
    final = llm_paraphrase(model_pack, formatted, question, debug)
    
    print(f"ANSWER: {final}\n{'='*60}\n")
    return final


# ==================== 8. MAIN ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GRAPHRAG CHATBOT - FIXED SPOUSE QUERY")
    print("="*60)

    try:
        llm_pack = load_llm_model(fine_tune=args.use_finetuned)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    test_questions = [
       
       "người hợp tác với người hợp tác với Trấn Thành là ai?",
        "Trấn Thành đóng phim gì?",
        "Ai đạo diễn phim Bố già?",
        "Phim Lật Mặt thuộc thể loại gì?",
        "Vợ Trấn Thành đóng phim gì?",
        "Đạo diễn phim Trấn Thành đóng phim gì",
        "Thể loại mà phim Trấn Thành đã đóng là gì",
        "Trấn Thành và Ninh Dương Lan Ngọc đóng chung phim nào?",
        "Bạn học của Trấn Thành đóng phim gì"
    
    ]
    
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    for i, q in enumerate(test_questions, 1):
        print(f"\n[TEST {i}/{len(test_questions)}]")
        try:
            get_answer(q, llm_pack, use_finetuned=args.use_finetuned, debug=True)
        except Exception as e:
            print(f"ERROR: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    close_driver()
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)