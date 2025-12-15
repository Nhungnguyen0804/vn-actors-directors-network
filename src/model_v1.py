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
    from src.chatbot.entity_linking_node import (
        normalize_text,
        entity_linking_graph
    )
    
    from src.nlp.ner import extract_entity_from_sentences
    
    from src.chatbot.graph_query import (
        build_query_from_relationships,
        query_flexible,
        RELATIONSHIPS,
        
        graph_query_movies_by_actor,
        graph_query_actors_of_movie,
        graph_query_movies_by_director,
        graph_query_director_of_movie,
        graph_query_same_schoolmates,
        graph_query_same_location,
        graph_query_common_movies,
        
        graph_query_director_of_actor_movies,
        graph_query_actors_in_director_movies,
        graph_query_spouse_movies,
        graph_query_schoolmate_movies,
        graph_query_common_directors,
        graph_query_coactor_network,
        graph_query_actor_collaboration_history,
        
        graph_query_shortest_path,
        graph_query_node_info,
        graph_query_with_planner,
        
        close_driver,
    )
    
except ImportError as e:
    print(f"Warning: Could not import from chatbot modules - {e}")


# ==================== 1. LOAD LLM ====================

def load_llm_model(lora_path="./outputs_graphrag_lora", fine_tune=False):
    if not os.path.exists("offload_weights"):
        os.makedirs("offload_weights")

    print("Dang khoi tao model...")
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        print("Dang tai Base Model (Qwen 0.5B)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            offload_folder="offload_weights",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        if not fine_tune:
            print("Che do Base Model - Khong su dung LoRA adapter")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            print("Tai thanh cong Base Model!")
            return model, tokenizer

        print(f"Dang tai Tokenizer tu thu muc LoRA: {lora_path}...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
        
        print("Dang gan Adapter (LoRA) vao Base Model...")
        model = PeftModel.from_pretrained(model, lora_path, offload_folder="offload_weights")
        
        print("Tai thanh cong! Model dang chay voi LoRA adapter.")
        return model, tokenizer

    except Exception as e:
        print(f"\nError: Loi tai model: {e}")
        sys.exit(1)


# ==================== 2. MULTI-QUERY COMPOSER ====================

def compose_multi_queries(question, linked_entities, debug=False):
    """
    Phát hiện và thực thi NHIỀU queries trong 1 câu hỏi
    CHỈ áp dụng khi có 2+ sub-queries (multi-query thật sự)
    """
    q_lower = normalize_text(question)
    
    # CRITICAL: Chỉ xử lý PERSON entities
    if not linked_entities or linked_entities[0].get('type') != 'person':
        return None
    
    # CRITICAL: Detect "và" để xác định multi-query
    if ' va ' not in q_lower and ' và ' not in q_lower:
        return None
    
    sub_queries = []
    
    # Detect base entity (spouse, self, etc.)
    has_spouse_query = bool(re.search(r'\b(vo|chong|ba\s+xa|ong\s+xa)\b', q_lower))
    
    # 1. Identity query: "là ai" / "tên gì"
    if re.search(r'\b(la\s+ai|ten\s+la\s+gi|ten\s+gi|ten\s+cua)\b', q_lower):
        if has_spouse_query:
            sub_queries.append({
                "type": "identity",
                "relationships": ["PERSON_SPOUSE"],
                "description": "Tên"
            })
    
    # 2. Birth date: "sinh năm" / "năm sinh"
    if re.search(r'\b(sinh\s+nam|nam\s+sinh|sinh|tuoi|bao\s+nhieu\s+tuoi)\b', q_lower):
        if has_spouse_query:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_DATE"],
                "description": "Năm sinh"
            })
        else:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_BIRTH_DATE"],
                "description": "Năm sinh"
            })
    
    # 3. Birth place: "quê" / "ở đâu"
    if re.search(r'\b(que|o\s+dau|noi\s+sinh|sinh\s+ra|sinh\s+o)\b', q_lower):
        if has_spouse_query:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_PLACE"],
                "description": "Quê quán"
            })
        else:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_BIRTH_PLACE"],
                "description": "Quê quán"
            })
    
    # 4. Movies: "đóng phim gì"
    if re.search(r'\b(dong|phim|tham\s+gia|dien\s+trong)\b', q_lower):
        if has_spouse_query:
            sub_queries.append({
                "type": "edge",
                "relationships": ["PERSON_SPOUSE", "PERSON_ACTED_IN_FILM"],
                "description": "Phim đóng"
            })
        else:
            sub_queries.append({
                "type": "edge",
                "relationships": ["PERSON_ACTED_IN_FILM"],
                "description": "Phim đóng"
            })
    
    # 5. Occupation: "nghề nghiệp"
    if re.search(r'\b(nghe\s+nghiep|lam\s+nghe|nghe|cong\s+viec)\b', q_lower):
        if has_spouse_query:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_SPOUSE", "PERSON_OCCUPATION"],
                "description": "Nghề nghiệp"
            })
        else:
            sub_queries.append({
                "type": "property",
                "relationships": ["PERSON_OCCUPATION"],
                "description": "Nghề nghiệp"
            })
    
    # CRITICAL: Chỉ xử lý khi có 2+ sub-queries (multi-query thật sự)
    if len(sub_queries) < 2:
        return None
    
    if debug:
        print(f"\n[COMPOSER] Detected {len(sub_queries)} sub-queries:")
        for sq in sub_queries:
            print(f"  - {sq['description']}: {sq['relationships']}")
    
    # Execute queries
    entity_name = linked_entities[0]['node_name']
    start_label = "PERSON"
    results = {}
    
    for sq in sub_queries:
        relationships = sq["relationships"]
        desc = sq["description"]
        
        try:
            step_result = build_query_from_relationships(
                entity_name,
                start_label,
                relationships,
                limit=20,
                debug=False,
                return_steps=True
            )
            
            if step_result and "final_result" in step_result:
                final = step_result["final_result"]
                
                if isinstance(final, list):
                    if len(final) == 1:
                        results[desc] = final[0]
                    elif len(final) <= 5:
                        results[desc] = ", ".join(str(x) for x in final)
                    else:
                        results[desc] = ", ".join(str(x) for x in final[:5]) + f" (va {len(final)-5} phim khac)"
                else:
                    results[desc] = final
                
                if debug:
                    print(f"  ✓ {desc}: {results[desc]}")
        
        except Exception as e:
            if debug:
                print(f"  ✗ {desc}: Error - {e}")
            continue
    
    return results if results else None


def format_composed_results(results, entity_name):
    """Format kết quả từ nhiều queries"""
    if not results:
        return None
    
    parts = []
    
    if "Tên" in results:
        parts.append(f"la {results['Tên']}")
    
    if "Năm sinh" in results:
        parts.append(f"sinh nam {results['Năm sinh']}")
    
    if "Quê quán" in results:
        parts.append(f"que o {results['Quê quán']}")
    
    if "Nghề nghiệp" in results:
        parts.append(f"lam nghe {results['Nghề nghiệp']}")
    
    if "Phim đóng" in results:
        parts.append(f"dong phim {results['Phim đóng']}")
    
    if not parts:
        return None
    
    return "THONG TIN: " + ", ".join(parts) + "."


# ==================== 3. INTENT DETECTION ====================

def detect_intent(question, num_entities=1):
    """
    Phát hiện intent bằng regex
    
    CRITICAL RULE: Nếu num_entities >= 2, CHỈ cho phép intent 2-hop trở lên
    """
    q_lower = normalize_text(question)
    
    patterns = {
        # === PROPERTY QUERIES (HIGHEST PRIORITY) ===
        "get_spouse_birthdate": [r'\b(vo|chong).+(sinh|nam\s+sinh)\b'],
        "get_spouse_birthplace": [r'\b(vo|chong).+(que|o\s+dau)\b'],
        "get_director_birthdate": [r'\b(dao\s+dien).+(phim).+(sinh|nam\s+sinh)\b'],
        "get_actor_birthdate": [r'\b(dien\s+vien).+(phim).+(sinh|nam\s+sinh)\b'],
        
        # === COLLECT PROPERTY QUERIES ===
        "get_actor_film_genres": [
            r'\b(the\s+loai).+(phim).+(dong|tham\s+gia)\b',
            r'\b(dong|tham\s+gia).+(the\s+loai|loai)\s+(phim)\b',
        ],
        "get_director_film_genres": [r'\b(dao\s+dien).+(the\s+loai|loai)\s+(phim)\b'],
        
        # === MULTI-HOP (2-HOP+) - CHECK BEFORE BASIC ===
        "get_spouse_movies": [
            r'\b(vo|chong|ba\s+xa|ong\s+xa).+(dong|phim|tham\s+gia)\b',
            r'\bphim\s+(cua|nao).+(vo|chong)\b'
        ],
        
        "get_director_of_actor_movies": [
            r'\b(dao\s+dien).+(phim)\s+(cua|ma)\b',
            r'\b(ai|nguoi\s+nao)\s+(dao\s+dien).+(phim).+(cua)\b',
        ],
        
        "get_common_directors": [r'\b(dao\s+dien)\s+(chung|cung|nao)\b'],
        "get_coactor_network": [r'\b(ban\s+dien|dong\s+phim\s+voi)\b'],
        "get_collaboration_history": [r'\b(lich\s+su|qua\s+trinh)\s+(hop\s+tac)\b'],
        "get_actors_in_director_movies": [r'\b(ai|dien\s+vien)\s+(dong|tham\s+gia).+(phim).+(dao\s+dien)\b'],
        "get_schoolmate_movies": [r'\b(ban\s+hoc|hoc\s+sinh).+(dong|phim)\b'],
        
        # === INTERSECTION (2-HOP+) ===
        "get_common_movies": [
            r'\b(dong|tham\s+gia)\s+(chung|cung)\b.*\b(phim)\b',
            r'\bphim\s+(chung|cung)\b',
        ],
        
        # === BASIC 1-HOP (ONLY ALLOWED IF num_entities == 1) ===
        "get_director_of_movie": [
            r'\b(ai|nguoi\s+nao)\s+(dao\s+dien|chi\s+dao)\s+(phim)\b',
            r'\b(dao\s+dien|chu\s+dao)\s+(cua\s+)?(phim)\b',
        ],
        "get_movies_by_director": [r'\b(dao\s+dien).+(lam|chi\s+dao|phim)\s+(nao|gi)\b'],
        "get_movies_by_actor": [
            r'\b(dong|tham\s+gia|vai)\s+(phim|trong)\b',
            r'\bphim\s+(cua|nao|gi)\b',
        ],
        "get_actors_of_movie": [r'\b(ai|dien\s+vien|cast)\s+(dong|vai|tham\s+gia)\b'],
        "get_same_school": [
            r'\b(hoc|ban\s+hoc|ban)\s+(cung|chung)\b',
            r'\bcung\s+(truong|hoc)\b',
        ],
        "get_same_location": [r'\b(cung|dong)\s+(que|huong|noi)\b'],
        
        # === OTHER ===
        "get_relationship_path": [r'\b(quan\s+he|lien\s+quan|ket\s+noi)\b'],
        "get_general_info": [r'\b(la\s+ai|la\s+gi)\b', r'\bthong\s+tin\b'],
    }
    
    # Define which intents are 1-hop (not allowed for 2+ entities)
    ONE_HOP_INTENTS = {
        "get_movies_by_actor",
        "get_actors_of_movie", 
        "get_movies_by_director",
        "get_director_of_movie",
        "get_same_school",
        "get_same_location",
        "get_general_info"
    }
    
    # CRITICAL: Priority order - Multi-hop BEFORE basic
    priority_order = [
        # Properties (highest)
        ("get_spouse_birthdate", 0.98), ("get_spouse_birthplace", 0.98),
        ("get_director_birthdate", 0.95), ("get_actor_birthdate", 0.95),
        ("get_actor_film_genres", 0.98), ("get_director_film_genres", 0.98),
        
        # Multi-hop (check BEFORE basic)
        ("get_spouse_movies", 0.95),
        ("get_director_of_actor_movies", 0.9),
        ("get_common_directors", 0.9),
        ("get_collaboration_history", 0.9),
        ("get_actors_in_director_movies", 0.9),
        ("get_schoolmate_movies", 0.9),
        ("get_coactor_network", 0.85),
        
        # Intersection (2-hop+)
        ("get_common_movies", 0.9),
        
        # Basic 1-hop (ONLY if num_entities == 1)
        ("get_director_of_movie", 0.85),
        ("get_movies_by_director", 0.8),
        ("get_movies_by_actor", 0.8),
        ("get_actors_of_movie", 0.8),
        ("get_same_school", 0.85),
        ("get_same_location", 0.85),
        
        # Other
        ("get_relationship_path", 0.75),
        ("get_general_info", 0.7)
    ]
    
    for intent_type, confidence in priority_order:
        # CRITICAL: Skip 1-hop intents if we have 2+ entities
        if num_entities >= 2 and intent_type in ONE_HOP_INTENTS:
            continue
        
        if intent_type in patterns:
            for pattern in patterns[intent_type]:
                if re.search(pattern, q_lower, re.IGNORECASE):
                    return {"intent": intent_type, "confidence": confidence}
    
    # CRITICAL: If 2+ entities but no valid intent found, force intersection
    if num_entities >= 2:
        return {"intent": "get_common_movies", "confidence": 0.7}
    
    return {"intent": "unknown", "confidence": 0.5}


# ==================== 4. INTENT MAPPING ====================

INTENT_TO_RELATIONSHIPS = {
    "get_movies_by_actor": {"relationships": ["PERSON_ACTED_IN_FILM"], "start_label": "PERSON", "requires_two_entities": False},
    "get_actors_of_movie": {"relationships": ["FILM_HAS_ACTOR"], "start_label": "FILM", "requires_two_entities": False},
    "get_movies_by_director": {"relationships": ["PERSON_DIRECTED_FILM"], "start_label": "PERSON", "requires_two_entities": False},
    "get_director_of_movie": {"relationships": ["FILM_HAS_DIRECTOR"], "start_label": "FILM", "requires_two_entities": False},
    "get_same_school": {"relationships": ["PERSON_SAME_SCHOOL"], "start_label": "PERSON", "requires_two_entities": False},
    "get_same_location": {"relationships": ["PERSON_SAME_LOCATION"], "start_label": "PERSON", "requires_two_entities": False},
    
    "get_spouse_birthdate": {"relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_DATE"], "start_label": "PERSON", "requires_two_entities": False},
    "get_spouse_birthplace": {"relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_PLACE"], "start_label": "PERSON", "requires_two_entities": False},
    "get_director_birthdate": {"relationships": ["FILM_HAS_DIRECTOR", "PERSON_BIRTH_DATE"], "start_label": "FILM", "requires_two_entities": False, "fallback_intent": "get_director_of_movie"},
    "get_actor_birthdate": {"relationships": ["FILM_HAS_ACTOR", "PERSON_BIRTH_DATE"], "start_label": "FILM", "requires_two_entities": False, "fallback_intent": "get_actors_of_movie"},
    
    "get_actor_film_genres": {"relationships": ["PERSON_ACTED_IN_FILM", "FILM_GENRE"], "start_label": "PERSON", "requires_two_entities": False, "fallback_intent": "get_movies_by_actor"},
    "get_director_film_genres": {"relationships": ["PERSON_DIRECTED_FILM", "FILM_GENRE"], "start_label": "PERSON", "requires_two_entities": False, "fallback_intent": "get_movies_by_director"},
    
    "get_spouse_movies": {"relationships": ["PERSON_SPOUSE", "PERSON_ACTED_IN_FILM"], "start_label": "PERSON", "requires_two_entities": False},
    "get_director_of_actor_movies": {"special": "custom", "query_func": graph_query_director_of_actor_movies, "requires_two_entities": False, "fallback_intent": "get_movies_by_actor"},
    "get_actors_in_director_movies": {"special": "custom", "query_func": graph_query_actors_in_director_movies, "requires_two_entities": False, "fallback_intent": "get_movies_by_director"},
    "get_schoolmate_movies": {"special": "custom", "query_func": graph_query_schoolmate_movies, "requires_two_entities": False, "fallback_intent": "get_same_school"},
    
    "get_common_movies": {"special": "intersection", "query_func": graph_query_common_movies, "requires_two_entities": True},
    "get_common_directors": {"special": "custom", "query_func": graph_query_common_directors, "requires_two_entities": True},
    "get_collaboration_history": {"special": "custom", "query_func": graph_query_actor_collaboration_history, "requires_two_entities": True},
    "get_coactor_network": {"special": "custom", "query_func": graph_query_coactor_network, "requires_two_entities": False},
    "get_relationship_path": {"special": "custom", "query_func": graph_query_shortest_path, "requires_two_entities": True},
    "get_general_info": {"special": "custom", "query_func": graph_query_node_info, "requires_two_entities": False}
}


# ==================== 5. QUERY ROUTER ====================

def route_graph_query_dynamic(linked_entities, question, intent, debug=False):
    """Dynamic query routing với multi-query support"""
    
    num_entities = len(linked_entities)
    
    # === CRITICAL CHECK: If 2+ entities, ensure intent is 2-hop+ ===
    if num_entities >= 2:
        intent_type = intent['intent']
        config = INTENT_TO_RELATIONSHIPS.get(intent_type, {})
        
        # Check if intent requires 2 entities OR is multi-hop
        is_valid_2entity_intent = (
            config.get("requires_two_entities", False) or
            config.get("special") in ["intersection", "custom"] or
            len(config.get("relationships", [])) >= 2
        )
        
        if not is_valid_2entity_intent:
            if debug:
                print(f"[ERROR] Intent '{intent_type}' is 1-hop but we have {num_entities} entities!")
                print(f"[FORCING] Switching to 'get_common_movies' for 2-entity query")
            
            # Force switch to intersection query
            intent = {"intent": "get_common_movies", "confidence": 0.7}
            intent_type = "get_common_movies"
    
    # === STEP 1: Try multi-query composer (CHỈ cho multi-query thật sự) ===
    composed = compose_multi_queries(question, linked_entities, debug=debug)
    if composed:
        entity_name = linked_entities[0]['node_name']
        formatted = format_composed_results(composed, entity_name)
        if formatted:
            if debug:
                print(f"[COMPOSER SUCCESS] Multi-query detected and processed")
            return {"status": "success", "data": composed, "message": "Multi-query success", "entity_name": entity_name, "formatted": formatted}
    
    # === STEP 2: Standard routing (cho single queries) ===
    intent_type = intent['intent']
    if not linked_entities:
        return {"status": "error", "data": None, "message": "Khong tim thay thuc the."}
    if intent_type not in INTENT_TO_RELATIONSHIPS:
        return {"status": "error", "data": None, "message": f"Intent {intent_type} chua duoc dinh nghia."}
    
    config = INTENT_TO_RELATIONSHIPS[intent_type]
    entity_name_1 = linked_entities[0]['node_name']
    
    if config.get("requires_two_entities", False):
        if num_entities < 2:
            return {"status": "error", "data": None, "message": "Can 2 thuc the."}
        entity_name_2 = linked_entities[1]['node_name']
    
    # Custom functions
    if config.get("special") == "custom":
        query_func = config["query_func"]
        if config.get("requires_two_entities"):
            data = query_func(entity_name_1, entity_name_2, debug=debug)
            entity_display = f"{entity_name_1} va {entity_name_2}"
        else:
            data = query_func(entity_name_1, debug=debug)
            entity_display = entity_name_1
            if not data and config.get("fallback_intent") and num_entities == 1:
                if debug: print(f"[FALLBACK] {config['fallback_intent']}")
                fallback_intent = {"intent": config["fallback_intent"], "confidence": 0.8}
                return route_graph_query_dynamic(linked_entities, question, fallback_intent, debug)
        return {"status": "success", "data": data, "message": "Query success", "entity_name": entity_display}
    
    # Intersection
    if config.get("special") == "intersection":
        if num_entities < 2:
            return {"status": "error", "data": None, "message": "Can 2 thuc the cho intersection query."}
        entity_name_2 = linked_entities[1]['node_name']
        data = config["query_func"](entity_name_1, entity_name_2, debug=debug)
        return {"status": "success", "data": data, "message": f"Found {len(data)} results", "entity_name": f"{entity_name_1} va {entity_name_2}"}
    
    # Standard queries
    relationships = config.get("relationships")
    if not relationships:
        return {"status": "error", "data": None, "message": "Invalid config."}
    
    start_label = config["start_label"]
    if debug:
        print(f"\n[ROUTING] Intent: {intent_type}, Relationships: {relationships}, Start: {entity_name_1} ({start_label})")
    
    if len(relationships) == 1:
        data = build_query_from_relationships(entity_name_1, start_label, relationships, limit=20, debug=debug)
    else:
        data = query_flexible(entity_name_1, start_label, relationships, debug=debug)
        if not data and config.get("fallback_intent") and num_entities == 1:
            if debug: print(f"[FALLBACK] {config['fallback_intent']}")
            fallback_intent = {"intent": config["fallback_intent"], "confidence": 0.8}
            return route_graph_query_dynamic(linked_entities, question, fallback_intent, debug)
    
    return {"status": "success", "data": data, "message": f"Found {len(data) if isinstance(data, list) else 1} results", "entity_name": entity_name_1}


# ==================== 6. FORMATTER ====================

def format_graph_data_dynamic(graph_data, intent_type, entity_name=None):
    """Format data động"""
    if not graph_data:
        return "KHONG TIM THAY thong tin."
    
    # Dict
    if isinstance(graph_data, dict):
        if 'properties' in graph_data or 'info_name' in graph_data:
            props = graph_data.get('properties', graph_data)
            key_map = {'info_birth_name': 'Ten khai sinh', 'info_birth_date': 'Nam sinh', 'info_occupation': 'Nghe nghiep', 'info_spouse': 'Vo/Chong', 'info_education': 'Hoc van', 'info_birth_place': 'Noi sinh'}
            info_list = [f"Ten: {props.get('info_name') or props.get('name') or entity_name}"]
            for k, v in props.items():
                if k in key_map and v:
                    info_list.append(f"{key_map[k]}: {str(v).replace('((', '').replace('))', '')}")
            return "THONG TIN: " + ". ".join(info_list) + "."
        if 'description' in graph_data:
            return f"QUAN HE: {graph_data['description']}."
    
    # List
    if isinstance(graph_data, list):
        if not graph_data:
            return "KHONG TIM THAY ket qua."
        
        # Check if list contains dict (complex structure)
        if isinstance(graph_data[0], dict):
            first_item = graph_data[0]
            
            # Case 1: Director of actor's movies
            if 'director' in first_item and 'film' in first_item:
                items = [f"{item['film']} (dao dien: {item['director']})" for item in graph_data]
                return f"DANH SACH DAO DIEN: {', '.join(items)}."
            
            # Case 2: Actors in director's movies
            elif 'actor' in first_item and 'film' in first_item:
                items = [f"{item['actor']} (phim: {item['film']})" for item in graph_data]
                return f"DANH SACH DIEN VIEN: {', '.join(items)}."
            
            # Case 3: Schoolmate movies
            elif 'schoolmate' in first_item and 'film' in first_item:
                items = [f"{item['schoolmate']}: {item['film']}" for item in graph_data]
                return f"DANH SACH PHIM: {', '.join(items)}."
            
            # Case 4: Coactor network (with distance)
            elif 'name' in first_item and 'distance' in first_item:
                items = [f"{item['name']} (khoang cach {item['distance']} buoc)" for item in graph_data]
                return f"MANG LUOI BAN DIEN: {', '.join(items)}."
            
            # Case 5: Collaboration history (with year)
            elif 'year' in first_item and 'film' in first_item:
                items = [f"{item['film']} ({item.get('year', 'N/A')}) - dao dien {item.get('director', 'N/A')}" for item in graph_data]
                return f"LICH SU HOP TAC: {', '.join(items)}."
            
            # Case 6: Common directors
            elif 'films_with_actor1' in first_item or 'films_with_actor2' in first_item:
                items = [item['director'] for item in graph_data]
                return f"DANH SACH DAO DIEN CHUNG: {', '.join(items)}."
            
            # Default: try to extract any meaningful field
            else:
                items = []
                for item in graph_data:
                    fields = [f"{k}: {v}" for k, v in item.items() if v and k not in ['id', 'type']]
                    if fields:
                        items.append(", ".join(fields[:2]))
                
                if items:
                    return f"DANH SACH: {' | '.join(items)}."
        
        items_str = ", ".join([str(item) for item in graph_data])
        templates = {
            "get_movies_by_actor": f"PHIM: {items_str}.",
            "get_actors_of_movie": f"DIEN VIEN: {items_str}.",
            "get_movies_by_director": f"PHIM: {items_str}.",
            "get_director_of_movie": f"DAO DIEN: {items_str}.",
            "get_common_movies": f"PHIM CHUNG: {items_str}.",
            "get_spouse_movies": f"PHIM: {items_str}.",
        }
        return templates.get(intent_type, f"KET QUA: {items_str}.")
    
    return f"THONG TIN: {str(graph_data)}."


# ==================== 7. LLM GENERATION ====================

def llm_paraphrase_graphrag(model_pack, formatted_sentence, question, use_finetuned=False, debug=False):
    """Viết lại câu trả lời tự nhiên"""
    model, tokenizer = model_pack
    
    if "KHONG TIM THAY" in formatted_sentence:
        return "Xin loi, hien tai toi chua co thong tin day du ve cau hoi nay trong du lieu."

    user_prompt = f"""{formatted_sentence}

Hoi: {question}
Tra loi:"""

    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=128, temperature=0.1, repetition_penalty=1.0, do_sample=False, num_beams=1)
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    response = re.sub(r'^[:\-\s]+', '', response)
    
    # Anti-hallucination
    if "không" in response.lower() and "không" not in formatted_sentence.lower():
        if debug: print(f"[WARNING] Model hallucination detected!")
        if "DANH SACH" in formatted_sentence or "PHIM" in formatted_sentence:
            parts = formatted_sentence.split(":")
            if len(parts) >= 2:
                response = parts[-1].strip().rstrip('.')
        else:
            response = formatted_sentence
    
    if debug:
        print(f"\n[LLM Input]: {formatted_sentence}")
        print(f"[LLM Output]: {response}")
    
    return response


# ==================== 8. MAIN PIPELINE ====================

def get_answer(question, model_pack, use_finetuned=False, debug=False):
    """Pipeline chính - HỖ TRỢ MULTI-QUERIES và ĐẢM BẢO 2-ENTITY = 2-HOP+"""
    print(f"\n{'='*60}\nQUESTION: {question}")
    
    # 1. Extract entities
    entities = extract_entity_from_sentences(question)
    if not entities:
        return "Khong tim thay ten rieng."
    if debug: print(f"[1] Entities: {entities}")
    
    # 2. Entity linking
    linked_entities = entity_linking_graph(question)
    if not linked_entities:
        return "Khong tim thay thuc the phu hop trong Graph."
    if debug: print(f"[2] Linked: {linked_entities}")
    
    # 3. Detect intent (PASS num_entities to enforce 2-hop rule)
    num_entities = len(linked_entities)
    intent = detect_intent(question, num_entities=num_entities)
    if debug: 
        print(f"[3] Intent: {intent['intent']} (num_entities={num_entities})")
        if num_entities >= 2:
            print(f"    [2-ENTITY MODE] Only 2-hop+ queries allowed")
    
    # 4. Dynamic query routing
    g_res = route_graph_query_dynamic(linked_entities, question, intent, debug=debug)
    if g_res['status'] == 'error':
        return g_res['message']
    if debug: print(f"[4] Graph Data: {g_res['data']}")
    
    # 5. Format & Generate
    if 'formatted' in g_res:
        formatted = g_res['formatted']
    else:
        formatted = format_graph_data_dynamic(g_res['data'], intent['intent'], g_res.get('entity_name'))
    
    final = llm_paraphrase_graphrag(model_pack, formatted, question, use_finetuned, debug)
    
    print(f"ANSWER: {final}\n{'='*60}\n")
    return final


# ==================== 9. TEST CASES ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetuned", action="store_true", help="Use fine-tuned LoRA model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GRAPHRAG CHATBOT - 2-ENTITY = 2-HOP+ ENFORCED")
    print("="*60)

    # Load Model
    try:
        llm_pack = load_llm_model(fine_tune=args.use_finetuned)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        exit(1)

    # Test queries
    test_questions = [
        # === BASIC 1-HOP QUERIES (1 entity) ===
        "Trấn Thành đóng phim gì?",
        "Ai đạo diễn phim Bố Già?",
        "Hari Won học cùng trường với ai?",
        
        # === 2-HOP PROPERTY QUERIES (1 entity) ===
        "Vợ của Trấn Thành sinh năm nao?",
        "Vợ Trấn Thành quê ở đâu?",
        "Đạo diễn phim Bố Già sinh năm nào?",
        
        # === 2-HOP EDGE QUERIES (1 entity) ===
        "Vợ của Trấn Thành đóng phim gì?",
        "Trấn Thành đóng thể loại phim gì?",
        
        # === MULTI-QUERY (3+ properties in 1 question) ===
        "Vợ của Trấn Thành là ai và sinh năm bao nhiêu và quê ở đâu?",
        "Hari Won sinh năm nào và quê ở đâu?",
        "Vợ Trấn Thành tên gì, sinh năm nào?",
        
        # === 2-ENTITY QUERIES (MUST BE 2-HOP+) ===
        "Trấn Thành và Hari Won đóng chung phim nào?",
        "Trấn Thành và Hari Won đóng phim gì?",  # Should find COMMON movies
        "Phim của Trấn Thành và Hari Won?",  # Should find COMMON movies
        
          ]
    
    print("\n" + "="*60)
    print("RUNNING TEST QUERIES")
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
    print("ALL TESTS COMPLETED")
    print("Neo4j driver closed.")
    print("="*60)