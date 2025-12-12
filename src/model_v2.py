import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import re
from peft import PeftModel

# Thêm đường dẫn cha để import module
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.nlp.ner import extract_entity_from_sentences
# ==================== IMPORT MODULES ====================
try:
    from src.chatbot.entity_linking_node import (
        extract_entities,
        normalize_text,
        
    )
    
    from src.chatbot.extract_entities_from_question import VIETNAMESE_STOPWORDS
    
    # IMPORT CÁC HÀM QUERY TỪ FILE NEO4J CYPHER
    from src.chatbot.graph_query import (
        graph_query_movies_by_actor,
        graph_query_actors_of_movie,
        graph_query_movies_by_director,
        graph_query_director_of_movie,
        graph_query_same_schoolmates,
        graph_query_same_location,
        graph_query_common_movies,
        graph_query_shortest_path,
        graph_query_node_info,
        close_driver
    )
    
except ImportError as e:
    print(f"⚠ Warning: Could not import from chatbot modules - {e}")


# ==================== 1. LOAD LLM ====================

def load_llm_model(model_path="./outputs_graphrag_lora", use_finetuned=False):
    """
    Tải model. 
    - Nếu use_finetuned=True: Tải Base Model + LoRA Adapter
    - Nếu use_finetuned=False: Tải Base Model gốc
    """
    
    # Xác định thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    if use_finetuned:
        print(f"[INFO] Đang tải model FINE-TUNED từ {model_path}...")
        
        # 1. Load Base Model (Model nền dùng lúc train)
        # Lưu ý: Lúc train dùng unsloth, lúc chạy inference dùng transformers thường
        base_model_name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit" 
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # 2. Load Adapter (LoRA) và gộp vào Base Model
            print("   -> Đang gộp Adapter LoRA...")
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload() # Gộp để chạy nhanh hơn
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải Fine-tuned model: {e}")
            print("Hãy chắc chắn thư mục 'outputs_graphrag_lora' tồn tại.")
            sys.exit(1)

    else:
        print("[INFO] Đang tải BASE model Qwen/Qwen2.5-0.5B-Instruct...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=None # Tránh lỗi disk offload với model nhỏ
            )
            model.to(device)
        except Exception as e:
            print(f"[ERROR] Không thể tải model gốc: {e}")
            sys.exit(1)
            
    print("[INFO] Tải model thành công!")
    return model, tokenizer

# ==================== 2. INTENT DETECTION ====================

def detect_intent(question):
    q_lower = normalize_text(question)
    
    if any(k in q_lower for k in ["dong phim", "phim cua", "tham gia"]):
        return {"intent": "get_movies_by_actor", "confidence": 0.9}
    elif any(k in q_lower for k in ["la ai", "sinh nam nao", "la gi", "thong tin ve"]):
        return {"intent": "get_general_info", "confidence": 0.7}
    elif any(k in q_lower for k in ["ai dong", "dien vien", "cast", "ai vai"]):
        return {"intent": "get_actors_of_movie", "confidence": 0.9}
    
    elif any(k in q_lower for k in ["dao dien", "ai chi dao", "lam phim"]):
        if "phim nao" in q_lower or "danh sach" in q_lower:
            return {"intent": "get_movies_by_director", "confidence": 0.9}
        else:
            return {"intent": "get_director_of_movie", "confidence": 0.9}
            
    elif any(k in q_lower for k in ["hoc cung", "ban hoc", "truong nao", "hoc truong","cung truong"]):
        return {"intent": "get_same_school", "confidence": 0.95}

    elif any(k in q_lower for k in ["cung que", "dong huong", "o dau", "song o","cung noi"]):
        return {"intent": "get_same_location", "confidence": 0.95}

    elif any(k in q_lower for k in ["dong chung", "hop tac", "cung dong"]):
        return {"intent": "get_common_movies", "confidence": 0.95}

    elif any(k in q_lower for k in ["quan he", "lien quan", "ket noi", "nhu the nao voi","co quan he"]):
        return {"intent": "get_relationship_path", "confidence": 0.85}

    else:
        return {"intent": "unknown", "confidence": 0.5}


# ==================== 3. ENTITY LINKING FOR CYPHER ====================

def detect_entity_type_from_context(entity, question):
    """Nhận diện loại entity từ ngữ cảnh trong câu hỏi"""
    
    context_window = question.lower()
    
    type_scores = {
        'film': 0,
        'person': 0,
    }
    
    # Tìm vị trí entity trong câu (dùng find vì nó trả -1 nếu không tìm thấy)
    entity_idx = context_window.find(entity)  # -1 nếu không tìm thấy
    if entity_idx != -1:
        before = context_window[:entity_idx].strip()
        after = context_window[entity_idx + len(entity):].strip()
        # vùng ngữ cảnh hẹp xung quanh entity để xét keyword gần entity
        local_window = context_window[max(0, entity_idx-30): entity_idx + len(entity) + 30]
    else:
        before = context_window
        after = ""
        local_window = context_window  # fallback: xét cả câu

    # === PATTERN MATCHING ===
    
    # Pattern 1: "ai đóng trong phim X" → X là FILM
    if ('ai dong' in before or 'dien vien' in before) and ('phim' in before or 'trong phim' in before):
        type_scores['film'] += 20
    
    # Pattern 2: "X đóng phim gì" → X là PERSON
    if ('dong phim' in after or 'tham gia' in after or 'dong' in after):
        type_scores['person'] += 20
    
    # # Pattern 3: "phim X" hoặc "bo phim X" → X là FILM
    # if 'phim' in before and entity_idx - before.rfind('phim') < 10:
    #     type_scores['film'] += 15
    # Pattern 3: "phim X" hoặc "bo phim X" → X là FILM
    # chỉ xét nếu entity thực sự có index
    if entity_idx != -1:
        last_phim_idx = before.rfind('phim')
        if last_phim_idx != -1 and (entity_idx - last_phim_idx) < 12:
            type_scores['film'] += 15
    
    # Pattern 4: "dao dien X" hoặc "dien vien X" → X là PERSON
    if 'dao dien' in before or 'dien vien' in before:
        type_scores['person'] += 15
    
    # === KEYWORD MATCHING ===
    
    # FILM INDICATORS
    film_keywords = ['phim', 'bo', 'tap', 'chieu', 'xem', 'flim']
    for kw in film_keywords:
        if kw in context_window:
            type_scores['film'] += 2
    
    # PERSON INDICATORS  
    person_keywords = ['dong', 'dien vien', 'dao dien', 'san xuat', 'ai', 'cung', 'va', 'hop tac', 'la ai']
    for kw in person_keywords:
        if kw in context_window:
            type_scores['person'] += 2
    
    # === SORT BY SCORE ===
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    result = [t[0] for t in sorted_types if t[1] > 0]
    
    if not result:
        result = ['person', 'film']
    
    return result


def fuzzy_match_node_cypher(entity, node_name, node_type=None, expected_types=None, threshold=80):
    """
    So khớp mờ giữa entity và tên node
    
    Ưu tiên:
    1. Exact match (không dấu)
    2. Substring match
    3. Fuzzy ratio
    """
    entity_norm = normalize_text(entity)
    node_norm = normalize_text(node_name)
    
    entity_tokens = entity_norm.split()
    node_tokens = node_norm.split()
    
    base_score = 0
    match_type = None
    
    # LEVEL 1: EXACT MATCH
    if entity_norm == node_norm:
        base_score = 100
        match_type = "exact"
    
    # LEVEL 2: TOKEN MATCH
    elif set(entity_tokens) == set(node_tokens):
        # Check order
        if entity_tokens == node_tokens:
            base_score = 95
            match_type = "tokens_ordered"
        else:
            base_score = 70
            match_type = "tokens_unordered"
    
    # LEVEL 3: SUBSTRING MATCH
    elif entity_norm in node_norm:
        base_score = 85
        match_type = "substring_entity_in_node"
    
    elif node_norm in entity_norm:
        base_score = 80
        match_type = "substring_node_in_entity"
    
    # LEVEL 4: FUZZY RATIO
    else:
        try:
            from fuzzywuzzy import fuzz
            score_ratio = fuzz.ratio(entity_norm, node_norm)
            score_token = fuzz.token_sort_ratio(entity_norm, node_norm)
            base_score = min(score_ratio, score_token)
            match_type = "fuzzy"
        except:
            base_score = 0
            match_type = "no_match"
    
    # TYPE BOOSTING
    score = base_score
    
    if expected_types and node_type:
        if node_type in expected_types:
            type_rank = expected_types.index(node_type)
            if type_rank == 0:
                score += 15
            else:
                score += 8
        else:
            score -= 10
    
    matched = score >= threshold
    
    return score, matched, match_type


def entity_linking_cypher(question, threshold=70, debug=False):
    """
    Entity linking sử dụng Cypher query + fuzzy matching logic
    
    Returns:
        list: [{'name': str, 'type': str, 'score': float, 'match_type': str}, ...]
    """
    from neo4j import GraphDatabase
    
    # Connection
    URI = "neo4j+s://0538a688.databases.neo4j.io"
    AUTH = ("neo4j", "askC5IvfBm2QXlzpKKn6gb9CEGxdouOCdTTKMhI6Si4")
    
    # Extract entities từ câu hỏi
    entities = extract_entity_from_sentences(question)
    
    # Làm sạch entities (bỏ dấu chấm hỏi, dấu phẩy...)
    entities = [e.rstrip('?.,!;:') for e in entities]
    
    if not entities:
        return []
    
    if debug:
        print(f"[ENTITY LINKING] Extracted entities: {entities}")
    
    linked_entities = []
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            for entity in entities:
                if debug:
                    print(f"\n[LINKING] Entity: '{entity}'")
                
                # Bước 1: Detect expected types từ context
                expected_types = detect_entity_type_from_context(entity, question)
                
                if debug:
                    print(f"  [TYPE] Expected types: {expected_types}")
                
                # Bước 2: Xác định threshold theo type
                if 'FILM' in expected_types and len(expected_types) == 1:
                    type_threshold = 85  # FILM: threshold cao hơn
                else:
                    type_threshold = threshold  # Person: 70
                
                if debug:
                    print(f"  [THRESHOLD] Using threshold: {type_threshold}")
                
                # Bước 3: Query nodes từ Neo4j theo expected types
                # Build WHERE clause cho labels
                label_conditions = []
                for exp_type in expected_types:
                    if exp_type == 'person':
                        label_conditions.append("'PERSON' IN labels(n)")
                    elif exp_type == 'film':
                        label_conditions.append("'FILM' IN labels(n)")
                
                if not label_conditions:
                    label_conditions = ["'PERSON' IN labels(n) OR 'FILM' IN labels(n)"]
                
                where_clause = " OR ".join(label_conditions)
                
                query = f"""
                MATCH (n)
                WHERE n.info_name IS NOT NULL
                AND ({where_clause})
                RETURN 
                    n.info_name AS node_name,
                    labels(n) AS labels
                LIMIT 200
                """
                
                result = session.run(query)
                
                # Bước 4: Fuzzy matching với từng node
                candidates = []
                
                for record in result:
                    node_name = record["node_name"]
                    labels = record["labels"]
                    
                    if not node_name:
                        continue
                    
                    # Xác định node type
                    node_type = 'unknown'
                    if 'PERSON' in labels:
                        node_type = 'person'
                    elif 'FILM' in labels:
                        node_type = 'film'
                    
                    # Fuzzy matching
                    score, matched, match_type = fuzzy_match_node_cypher(
                        entity=entity,
                        node_name=node_name,
                        node_type=node_type,
                        expected_types=expected_types,
                        threshold=type_threshold
                    )
                    
                    if matched:
                        candidates.append({
                            'name': node_name,
                            'type': node_type,
                            'score': score,
                            'match_type': match_type,
                            'original_entity': entity
                        })
                
                # Bước 5: Lấy best match
                if candidates:
                    # Sort by score
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    best_match = candidates[0]
                    
                    linked_entities.append(best_match)
                    
                    if debug:
                        print(f"  ✓ Best match: {best_match['name']} (score: {best_match['score']}, type: {best_match['type']}, match: {best_match['match_type']})")
                        if len(candidates) > 1:
                            print(f"  [INFO] Found {len(candidates)} total candidates")
                            for c in candidates[:3]:
                                print(f"    → {c['name']} (score: {c['score']}, {c['match_type']})")
                else:
                    if debug:
                        print(f"  ✗ No match found (threshold: {type_threshold})")
    
    return linked_entities


# ==================== 4. GRAPH ROUTING ====================

def route_graph_query(linked_entities, question, intent, debug=False):
    """
    Điều hướng query - Sử dụng các hàm Cypher đã import
    
    Args:
        linked_entities: List of dicts with 'name', 'type', 'score'
        question: Câu hỏi
        intent: Dict với 'intent' và 'confidence'
        debug: Debug mode
    """
    intent_type = intent['intent']
    
    if not linked_entities:
        return {"status": "error", "data": None, "message": "Không tìm thấy thực thể liên quan."}

    # Lấy tên entity đầu tiên
    entity_name_1 = linked_entities[0]['name']
    
    # --- NHÓM 1: CÁC HÀM 1-HOP CƠ BẢN ---
    
    if intent_type == "get_movies_by_actor":
        data = graph_query_movies_by_actor(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy {len(data)} phim của {entity_name_1}", 
            "entity_name": entity_name_1
        }
        
    elif intent_type == "get_actors_of_movie":
        data = graph_query_actors_of_movie(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy {len(data)} diễn viên trong phim {entity_name_1}", 
            "entity_name": entity_name_1
        }
        
    elif intent_type == "get_general_info":
        data = graph_query_node_info(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data,
            "message": f"Tìm thấy thông tin về {entity_name_1}",
            "entity_name": entity_name_1
        }

    elif intent_type == "get_director_of_movie":
        data = graph_query_director_of_movie(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy đạo diễn của {entity_name_1}", 
            "entity_name": entity_name_1
        }

    elif intent_type == "get_movies_by_director":
        data = graph_query_movies_by_director(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy phim của đạo diễn {entity_name_1}", 
            "entity_name": entity_name_1
        }

    # --- NHÓM 2: CÁC HÀM ATTRIBUTE ---

    elif intent_type == "get_same_school":
        data = graph_query_same_schoolmates(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy bạn học của {entity_name_1}", 
            "entity_name": entity_name_1
        }

    elif intent_type == "get_same_location":
        data = graph_query_same_location(entity_name_1, debug=debug)
        return {
            "status": "success", "data": data, 
            "message": f"Tìm thấy người cùng quê với {entity_name_1}", 
            "entity_name": entity_name_1
        }

    # --- NHÓM 3: CÁC HÀM 2 ENTITY ---

    elif intent_type in ["get_common_movies", "get_relationship_path"]:
        if len(linked_entities) < 2:
            return {"status": "error", "data": None, "message": "Câu hỏi này cần ít nhất 2 người/phim để so sánh."}
             
        entity_name_2 = linked_entities[1]['name']
        
        if intent_type == "get_common_movies":
            data = graph_query_common_movies(entity_name_1, entity_name_2, debug=debug)
            return {
                "status": "success", "data": data, 
                "message": f"Phim chung của {entity_name_1} và {entity_name_2}", 
                "entity_name": f"{entity_name_1} và {entity_name_2}"
            }
            
        elif intent_type == "get_relationship_path":
            data = graph_query_shortest_path(entity_name_1, entity_name_2, debug=debug)
            return {
                "status": "success", "data": data, 
                "message": f"Mối quan hệ giữa {entity_name_1} và {entity_name_2}", 
                "entity_name": f"{entity_name_1} và {entity_name_2}"
            }
        
    else:
        return {"status": "error", "data": None, "message": "Không xác định được ý định của câu hỏi."}


# ==================== 5. FORMATTER ====================

def format_graph_data_strictly(graph_data, intent_type, entity_name=None):  
    if not graph_data:
        return "KHÔNG TÌM THẤY thông tin liên quan trong dữ liệu."

    # --- XỬ LÝ CHO INTENT: GET_GENERAL_INFO ---
    if intent_type == "get_general_info":
        if isinstance(graph_data, dict):
            props = graph_data.get('properties', graph_data)
            
            key_map = {
                'info_birth_name': 'Tên khai sinh',
                'info_birth_date': 'Năm sinh',
                'info_occupation': 'Nghề nghiệp',
                'info_spouse': 'Vợ/Chồng',
                'info_relatives': 'Người thân',
                'info_education': 'Học vấn',
                'info_height': 'Chiều cao',
                'info_nationality': 'Quốc tịch',
                'info_birth_place': 'Nơi sinh'
            }
            
            info_list = []
            name = props.get('info_name') or props.get('name') or entity_name
            info_list.append(f"Tên: {name}")

            for k, v in props.items():
                if k in key_map and v:
                    clean_v = str(v).replace('((', '').replace('))', '').replace('*', ',')
                    info_list.append(f"{key_map[k]}: {clean_v}")
            
            return "THÔNG TIN HỒ SƠ: " + ". ".join(info_list) + "."
        
        return f"THÔNG TIN: {str(graph_data)}"

    # --- XỬ LÝ RELATIONSHIP PATH ---
    if intent_type == "get_relationship_path":
        if isinstance(graph_data, dict) and 'description' in graph_data:
            return f"THÔNG TIN QUAN HỆ: {graph_data['description']}."
        return "Không tìm thấy đường đi kết nối."

    # --- XỬ LÝ LIST ---
    if isinstance(graph_data, list):
        items_str = ", ".join([str(item) for item in graph_data])
    else:
        items_str = str(graph_data)
    
    templates = {
        "get_movies_by_actor": f"DANH SÁCH PHIM: Diễn viên {entity_name} đóng các phim: {items_str}.",
        "get_actors_of_movie": f"DANH SÁCH DIỄN VIÊN: Phim {entity_name} có diễn viên: {items_str}.",
        "get_movies_by_director": f"DANH SÁCH PHIM: Đạo diễn {entity_name} làm phim: {items_str}.",
        "get_director_of_movie": f"THÔNG TIN ĐẠO DIỄN: Đạo diễn phim {entity_name} là: {items_str}.",
        "get_same_school": f"DANH SÁCH: Người học cùng trường {entity_name}: {items_str}.",
        "get_same_location": f"DANH SÁCH: Người cùng quê {entity_name}: {items_str}.",
        "get_common_movies": f"DANH SÁCH: {entity_name} đóng chung phim: {items_str}."
    }
    return templates.get(intent_type, f"THÔNG TIN: {items_str}")


# ==================== 6. GENERATION ====================

def llm_paraphrase_graphrag(model_pack, formatted_sentence, question, use_finetuned=False, debug=False):
    """Viết lại câu trả lời - Bắt buộc model PHẢI dùng đúng data"""
    model, tokenizer = model_pack
    
    if "KHÔNG TÌM THẤY" in formatted_sentence:
        return "Xin lỗi, hiện tại tôi chưa có thông tin đầy đủ về câu hỏi này trong dữ liệu."

   # ===== PROMPT ĐƠN GIẢN CHO MODEL NHỎ =====
    if use_finetuned:
        system_prompt = "Bạn là trợ lý trả lời câu hỏi từ dữ liệu."
        user_prompt = f"""Dữ liệu: {formatted_sentence}

Câu hỏi: {question}

Trả lời ngắn gọn dựa trên dữ liệu:"""
    else:
        # Không dùng system prompt với model nhỏ
        user_prompt = f"""{formatted_sentence}

Hỏi: {question}
Trả lời:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.3,  # Giảm temperature để model ít "sáng tạo"
            repetition_penalty=1.0,  # Bỏ penalty để tránh lặp
            do_sample=False , # Tắt sampling với model nhỏ,
            num_beams=2 ,         # Dùng beam search
            top_k=20,
            top_p=0.85
        )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    response = re.sub(r'^[:\-\s]+', '', response)
    
    # Kiểm tra nếu model trả lời sai (phủ nhận data)
    if "không" in response.lower() and "không" not in formatted_sentence.lower():
        # Model đang hallucinate - trả về câu trả lời trực tiếp từ data
        if debug:
            print(f"[WARNING] Model hallucination detected! Using direct answer from data.")
        # Trích xuất trực tiếp từ formatted_sentence
        if "DANH SÁCH" in formatted_sentence:
            # Lấy phần sau dấu hai chấm
            parts = formatted_sentence.split(":")
            if len(parts) >= 2:
                response = parts[-1].strip().rstrip('.')
        else:
            response = formatted_sentence
    
    if debug:
        print(f"\n[LLM Input Data]: {formatted_sentence}")
        print(f"[LLM Output]: {response}")
        
    return response


# ==================== 7. PIPELINE CHÍNH ====================

def get_answer(question, model_pack, use_finetuned=False, debug=False):
    """
    Pipeline chính - KHÔNG CẦN GRAPH NỮA, dùng Cypher trực tiếp
    """
    print(f"\n{'='*60}\nQUESTION: {question}")
    
    # Bước 1: Trích xuất entities
    entities = extract_entity_from_sentences(question)
    
    if not entities:
        return "Không tìm thấy tên riêng."
    if debug:
        print(f"[1] Entities: {entities}")
    
    # Bước 2: Phát hiện intent
    intent = detect_intent(question)
    if debug:
        print(f"[2] Intent: {intent['intent']}")
    
    # Bước 3: Entity linking với Cypher
    linked_entities = entity_linking_cypher(question)
    if not linked_entities:
        return "Không tìm thấy thực thể phù hợp trong Graph."
    
    if debug:
        print(f"[3] Linked Entities: {linked_entities}")
    
    # Bước 4: Query graph với Cypher
    g_res = route_graph_query(linked_entities, question, intent, debug=debug)
    if g_res['status'] == 'error':
        return g_res['message']
    if debug:
        print(f"[4] Graph Data: {g_res['data']}")
    
    # Bước 5: Format và generate
    formatted = format_graph_data_strictly(g_res['data'], intent['intent'], g_res.get('entity_name'))
    final = llm_paraphrase_graphrag(model_pack, formatted, question, use_finetuned, debug)
    
    print(f"ANSWER: {final}\n{'='*60}\n")
    return final


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("\n>>> INITIALIZING SYSTEM...")
    
    
    print("✓ Sử dụng Cypher queries trực tiếp từ Neo4j")

    # Load Model
    try:
        llm_pack = load_llm_model(use_finetuned=args.use_finetuned)
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        exit(1)

    # Test
    test_questions = [
        "Trấn Thành đóng phim gì?",
        "Trấn Thành và Ninh Dương Lan Ngọc có quan hệ gì?",
        "Trấn Thành và Hari Won đóng chung phim nào?",
        "Ai cùng trường với Ninh Dương Lan Ngọc?",
        "Đạo diễn của phim Bố Già là ai?",
        "Victor Vũ làm đạo diễn phim nào?",
    ]
    
    for q in test_questions:
        get_answer(q, llm_pack, use_finetuned=True, debug=True)
    
    
    
    # Đóng kết nối Neo4j khi kết thúc
    close_driver()
    print("\n Neo4j driver closed.")