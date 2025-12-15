import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Them duong dan cha de import module
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== IMPORT MODULES ====================
try:
    from src.chatbot.entity_linking_node import (
        normalize_text,
        entity_linking_graph
    )
    
    from src.nlp.ner import extract_entity_from_sentences
    
    # IMPORT DYNAMIC QUERY BUILDER
    from src.chatbot.graph_query import (
        # Core dynamic query functions
        build_query_from_relationships,
        query_flexible,
        RELATIONSHIPS,
        
        # Convenience wrappers
        graph_query_movies_by_actor,
        graph_query_actors_of_movie,
        graph_query_movies_by_director,
        graph_query_director_of_movie,
        graph_query_same_schoolmates,
        graph_query_same_location,
        graph_query_common_movies,
        
        # Multi-hop functions
        graph_query_director_of_actor_movies,
        graph_query_actors_in_director_movies,
        graph_query_spouse_movies,
        graph_query_schoolmate_movies,
        graph_query_common_directors,
        graph_query_coactor_network,
        graph_query_actor_collaboration_history,
        
        # Special queries
        graph_query_shortest_path,
        graph_query_node_info,
        graph_query_with_planner,
        
        # Utilities
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


# ==================== 2. INTENT TO RELATIONSHIPS MAPPING ====================

INTENT_TO_RELATIONSHIPS = {
    # === BASIC 1-HOP QUERIES ===
    "get_movies_by_actor": {
        "relationships": ["PERSON_ACTED_IN_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_actors_of_movie": {
        "relationships": ["FILM_HAS_ACTOR"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_movies_by_director": {
        "relationships": ["PERSON_DIRECTED_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_director_of_movie": {
        "relationships": ["FILM_HAS_DIRECTOR"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_same_school": {
        "relationships": ["PERSON_SAME_SCHOOL"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_same_location": {
        "relationships": ["PERSON_SAME_LOCATION"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    # === PROPERTY QUERIES ===
    "get_birthdate_spouse": {
        "special": "planner",
        "requires_two_entities": False
    },
    
    # === EDGE -> PROPERTY QUERIES ===
    "get_director_birthdate": {
        "relationships": ["FILM_HAS_DIRECTOR", "PERSON_BIRTH_DATE"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_actor_birthdate": {
        "relationships": ["FILM_HAS_ACTOR", "PERSON_BIRTH_DATE"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_spouse_birthdate": {
        "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_DATE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_spouse_birthplace": {
        "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_PLACE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    # === EDGE -> LIST PROPERTY (COLLECT) ===
    "get_actor_film_genres": {
        "relationships": ["PERSON_ACTED_IN_FILM", "FILM_GENRE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_director_film_genres": {
        "relationships": ["PERSON_DIRECTED_FILM", "FILM_GENRE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    # === MULTI-HOP QUERIES ===
    "get_spouse_movies": {
        "relationships": ["PERSON_SPOUSE", "PERSON_ACTED_IN_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_director_of_actor_movies": {
        "special": "custom",
        "query_func": graph_query_director_of_actor_movies,
        "requires_two_entities": False
    },
    
    "get_actors_in_director_movies": {
        "special": "custom",
        "query_func": graph_query_actors_in_director_movies,
        "requires_two_entities": False
    },
    
    "get_schoolmate_movies": {
        "special": "custom",
        "query_func": graph_query_schoolmate_movies,
        "requires_two_entities": False
    },
    
    # === INTERSECTION QUERIES (2 entities) ===
    "get_common_movies": {
        "special": "intersection",
        "query_func": graph_query_common_movies,
        "requires_two_entities": True
    },
    
    "get_common_directors": {
        "special": "custom",
        "query_func": graph_query_common_directors,
        "requires_two_entities": True
    },
    
    "get_collaboration_history": {
        "special": "custom",
        "query_func": graph_query_actor_collaboration_history,
        "requires_two_entities": True
    },
    
    "get_coactor_network": {
        "special": "custom",
        "query_func": graph_query_coactor_network,
        "requires_two_entities": False
    },
    
    "get_relationship_path": {
        "special": "custom",
        "query_func": graph_query_shortest_path,
        "requires_two_entities": True
    },
    
    "get_general_info": {
        "special": "custom",
        "query_func": graph_query_node_info,
        "requires_two_entities": False
    }
}


# ==================== 3. INTENT DETECTION ====================

# ==================== CẬP NHẬT DETECT_INTENT VỚI PATTERN TỐT HƠN ====================

def detect_intent(question):
    """Phat hien intent bang regex - CẢI THIỆN PHÂN BIỆT 1-HOP vs 2-HOP"""
    q_lower = normalize_text(question)
    
    patterns = {
        # === COLLECT PROPERTY QUERIES (check truoc) ===
        "get_actor_film_genres": [
            r'\b(the\s+loai).+(phim).+(dong|tham\s+gia)\b',
            r'\b(dong|tham\s+gia).+(the\s+loai|loai)\s+(phim)\b',
            r'\b(cac|nhung)\s+(the\s+loai)\b',
        ],
        
        "get_director_film_genres": [
            r'\b(the\s+loai).+(phim).+(dao\s+dien)\b',
            r'\b(dao\s+dien).+(the\s+loai|loai)\s+(phim)\b',
        ],
        
        # === PROPERTY QUERIES (check truoc) ===
        "get_spouse_birthdate": [
            r'\b(vo|chong).+(sinh|nam\s+sinh)\b',
        ],
        
        "get_spouse_birthplace": [
            r'\b(vo|chong).+(que|o\s+dau|sinh\s+ra)\b',
        ],
        
        "get_director_birthdate": [
            r'\b(dao\s+dien).+(phim).+(sinh|nam\s+sinh)\b',  # "đạo diễn phim X sinh năm nào"
        ],
        
        "get_actor_birthdate": [
            r'\b(dien\s+vien).+(phim).+(sinh|nam\s+sinh)\b',  # "diễn viên phim X sinh năm nào"
        ],
        
        "get_birthdate_spouse": [
            r'\bsinh\s+(nam|ngay|thang)\s+(nao|bao\s+nhieu)\b',
            r'\bnam\s+sinh\b',
            r'\bngay\s+sinh\b',
            r'\bque\s+(quan|o\s+dau)\b',
            r'\b(vo|chong|ba\s+xa|ong\s+xa)\s+(cua|la)\b(?!.+(sinh|que|o\s+dau))',
        ],
        
        # === MULTI-HOP QUERIES ===
        "get_spouse_movies": [
            r'\b(vo|chong|ba\s+xa|ong\s+xa).+(dong|phim)\b',
            r'\bphim\s+(cua|nao).+(vo|chong)\b'
        ],
        
        "get_director_of_actor_movies": [
            # QUAN TRỌNG: Chỉ match khi có "phim của NGƯỜI" (actor)
            r'\b(dao\s+dien).+(phim)\s+(cua|ma).+(?=\b(tran\s+thanh|hari\s+won|[\w\s]+)\b)',  # "đạo diễn phim của Trấn Thành"
            r'\b(ai|nguoi\s+nao)\s+(dao\s+dien).+(phim).+(cua)\b',  # "ai đạo diễn phim của X"
        ],
        
        "get_common_directors": [
            r'\b(dao\s+dien)\s+(chung|cung|nao)\b',
            r'\b(lam\s+viec|hop\s+tac)\s+voi\s+(dao\s+dien)\b',
        ],
        
        "get_coactor_network": [
            r'\b(ban\s+dien|dong\s+phim\s+voi)\b',
            r'\b(ai|nguoi\s+nao)\s+(dong\s+chung|tung\s+dong)\b'
        ],
        
        "get_collaboration_history": [
            r'\b(lich\s+su|qua\s+trinh)\s+(hop\s+tac)\b',
            r'\b(tung|da)\s+(hop\s+tac|dong\s+chung)\s+(trong|phim|nao)\b',
        ],
        
        "get_actors_in_director_movies": [
            r'\b(ai|dien\s+vien)\s+(dong|tham\s+gia).+(phim).+(dao\s+dien)\b',
        ],
        
        "get_schoolmate_movies": [
            r'\b(ban\s+hoc|hoc\s+sinh).+(dong|phim)\b',
        ],
        
        # === BASIC 1-HOP (ƯU TIÊN CAO HƠN) ===
        "get_director_of_movie": [
            # Pattern rõ ràng: "đạo diễn phim TÊN_PHIM"
            r'\b(ai|nguoi\s+nao)\s+(dao\s+dien|chi\s+dao)\s+(phim)\b',  # "ai đạo diễn phim"
            r'\b(dao\s+dien|chu\s+dao)\s+(cua\s+)?(phim)\b',  # "đạo diễn phim" hoặc "đạo diễn của phim"
            r'\bphim\s+.+\s+(do|boi|cua)\s+(ai|nguoi\s+nao)\s+(dao\s+dien)\b',  # "phim X do ai đạo diễn"
        ],
        
        "get_movies_by_director": [
            r'\b(dao\s+dien).+(lam|chi\s+dao|phim)\s+(nao|gi)\b',
            r'\bphim\s+(cua|nao).+(dao\s+dien)\b',
        ],
        
        "get_movies_by_actor": [
            r'\b(dong|tham\s+gia|vai)\s+(phim|trong)\b',
            r'\bphim\s+(cua|nao|gi)\b',
        ],
        
        "get_actors_of_movie": [
            r'\b(ai|dien\s+vien|cast)\s+(dong|vai|tham\s+gia)\b',
        ],
        
        "get_same_school": [
            r'\b(hoc|ban\s+hoc|ban)\s+(cung|chung)\b',
            r'\bcung\s+(truong|hoc)\b',
        ],
        
        "get_same_location": [
            r'\b(cung|dong)\s+(que|huong|noi)\b',
        ],
        
        # === INTERSECTION QUERIES - PHIM CHUNG (2 entities) ===
        "get_common_movies": [
            r'\b(dong|tham\s+gia)\s+(chung|cung)\b.*\b(phim)\b',
            r'\bphim\s+(chung|cung)\b',
            r'\b(cung|chung)\s+(dong|tham\s+gia)\b',
            r'\b(co|da)\s+(dong|hop\s+tac)\s+(chung|cung|voi)\b.*\b(phim)\b',
            r'\b(hai|2)\s+(nguoi|ng).+(dong|tham\s+gia)\s+(chung|cung)\b',
            r'\b(va|voi)\b.+(dong\s+chung|tham\s+gia\s+chung)\b',
        ],
        
        "get_relationship_path": [
            r'\b(quan\s+he|lien\s+quan|ket\s+noi)\b',
            r'\bmoi\s+(quan\s+he)\b'
        ],
        
        "get_general_info": [
            r'\b(la\s+ai|la\s+gi)\b',
            r'\bthong\s+tin\b',
        ]
    }
    
    # Priority order - 1-HOP TRƯỚC 2-HOP
    priority_order = [
        # Collect properties (highest priority)
        ("get_actor_film_genres", 0.98),
        ("get_director_film_genres", 0.98),
        
        # Property chains (high priority)
        ("get_spouse_birthdate", 0.98),
        ("get_spouse_birthplace", 0.98),
        ("get_director_birthdate", 0.95),
        ("get_actor_birthdate", 0.95),
        
        # === 1-HOP QUERIES (ƯU TIÊN CAO) ===
        ("get_director_of_movie", 0.95),  # TĂNG ƯU TIÊN
        ("get_movies_by_director", 0.9),
        ("get_movies_by_actor", 0.9),
        ("get_actors_of_movie", 0.85),
        ("get_same_school", 0.95),
        ("get_same_location", 0.95),
        
        # INTERSECTION - PHIM CHUNG
        ("get_common_movies", 0.95),
        
        # === MULTI-HOP (ƯU TIÊN THẤP HƠN 1-HOP) ===
        ("get_spouse_movies", 0.9),  # GIẢM ƯU TIÊN
        ("get_director_of_actor_movies", 0.85),  # GIẢM ƯU TIÊN
        ("get_common_directors", 0.9),
        ("get_collaboration_history", 0.9),
        ("get_actors_in_director_movies", 0.9),
        ("get_schoolmate_movies", 0.9),
        ("get_coactor_network", 0.85),
        
        # Properties
        ("get_birthdate_spouse", 0.85),
        
        # Others
        ("get_relationship_path", 0.85),
        ("get_general_info", 0.7)
    ]
    
    for intent_type, confidence in priority_order:
        if intent_type in patterns:
            for pattern in patterns[intent_type]:
                if re.search(pattern, q_lower, re.IGNORECASE):
                    return {"intent": intent_type, "confidence": confidence}
    
    return {"intent": "unknown", "confidence": 0.5}


# ==================== THÊM FALLBACK MECHANISM ====================

def route_graph_query_dynamic(linked_entities, question, intent, debug=False):
    """
    Dieu huong query DONG - CÓ FALLBACK MECHANISM
    Nếu query N-hop thất bại → tự động thử (N-1)-hop
    CHỈ ÁP DỤNG FALLBACK CHO 1-ENTITY QUERIES
    """
    intent_type = intent['intent']
    
    if not linked_entities:
        return {"status": "error", "data": None, "message": "Khong tim thay thuc the lien quan."}

    if intent_type not in INTENT_TO_RELATIONSHIPS:
        return {"status": "error", "data": None, "message": f"Intent {intent_type} chua duoc dinh nghia."}
    
    config = INTENT_TO_RELATIONSHIPS[intent_type]
    entity_name_1 = linked_entities[0]['node_name']
    num_entities = len(linked_entities)
    
    # Check neu can 2 entities
    if config.get("requires_two_entities", False):
        if num_entities < 2:
            return {"status": "error", "data": None, "message": "Cau hoi nay can it nhat 2 thuc the."}
        entity_name_2 = linked_entities[1]['node_name']
    
    # === XU LY SPECIAL CASES ===
    
    # PLANNER (for property queries)
    if config.get("special") == "planner":
        data = graph_query_with_planner(question, entity_name_1, entity_type="PERSON", debug=debug)
        return {
            "status": "success",
            "data": data,
            "message": f"Query thanh cong",
            "entity_name": entity_name_1
        }
    
    # CUSTOM FUNCTIONS
    if config.get("special") == "custom":
        query_func = config["query_func"]
        
        if config.get("requires_two_entities"):
            data = query_func(entity_name_1, entity_name_2, debug=debug)
            entity_display = f"{entity_name_1} va {entity_name_2}"
            # 2-entity queries KHÔNG có fallback - nếu thất bại thì thất bại
        else:
            data = query_func(entity_name_1, debug=debug)
            entity_display = entity_name_1
            
            # === FALLBACK: CHỈ cho 1-entity queries ===
            if not data and config.get("fallback_intent") and num_entities == 1:
                if debug:
                    print(f"[FALLBACK] 1-entity query thất bại, thử intent: {config['fallback_intent']}")
                fallback_intent = {"intent": config["fallback_intent"], "confidence": 0.8}
                return route_graph_query_dynamic(linked_entities, question, fallback_intent, debug)
        
        return {
            "status": "success",
            "data": data,
            "message": f"Query thanh cong",
            "entity_name": entity_display
        }
    
    # INTERSECTION
    if config.get("special") == "intersection":
        query_func = config["query_func"]
        data = query_func(entity_name_1, entity_name_2, debug=debug)
        # 2-entity intersection queries KHÔNG có fallback
        return {
            "status": "success",
            "data": data,
            "message": f"Tim thay {len(data) if isinstance(data, list) else 1} ket qua",
            "entity_name": f"{entity_name_1} va {entity_name_2}"
        }
    
    # === XU LY STANDARD QUERIES (using relationships) ===
    relationships = config.get("relationships")
    if not relationships:
        return {"status": "error", "data": None, "message": "Config khong hop le."}
    
    start_label = config["start_label"]
    
    if debug:
        print(f"\n[DYNAMIC ROUTING]")
        print(f"  Intent: {intent_type}")
        print(f"  Relationships: {relationships}")
        print(f"  Start: {entity_name_1} ({start_label})")
    
    # Execute query using dynamic builder
    if len(relationships) == 1:
        # Simple 1-hop
        rel_key = relationships[0]
        
        if rel_key == "PERSON_ACTED_IN_FILM":
            data = graph_query_movies_by_actor(entity_name_1, debug=debug)
        elif rel_key == "FILM_HAS_ACTOR":
            data = graph_query_actors_of_movie(entity_name_1, debug=debug)
        elif rel_key == "PERSON_DIRECTED_FILM":
            data = graph_query_movies_by_director(entity_name_1, debug=debug)
        elif rel_key == "FILM_HAS_DIRECTOR":
            data = graph_query_director_of_movie(entity_name_1, debug=debug)
        elif rel_key == "PERSON_SAME_SCHOOL":
            data = graph_query_same_schoolmates(entity_name_1, debug=debug)
        elif rel_key == "PERSON_SAME_LOCATION":
            data = graph_query_same_location(entity_name_1, debug=debug)
        else:
            data = build_query_from_relationships(
                entity_name_1, start_label, relationships, limit=20, debug=debug
            )
    else:
        # Multi-hop - use flexible query
        data = query_flexible(entity_name_1, start_label, relationships, debug=debug)
        
        # === FALLBACK MECHANISM: CHỈ cho 1-entity queries ===
        if not data and len(relationships) > 1 and config.get("fallback_intent") and num_entities == 1:
            if debug:
                print(f"[FALLBACK] Multi-hop 1-entity query trả về rỗng")
                print(f"[FALLBACK] Thử query 1-hop: {config['fallback_intent']}")
            
            fallback_intent = {"intent": config["fallback_intent"], "confidence": 0.8}
            return route_graph_query_dynamic(linked_entities, question, fallback_intent, debug)
    
    return {
        "status": "success",
        "data": data,
        "message": f"Tim thay {len(data) if isinstance(data, list) else 1} ket qua",
        "entity_name": entity_name_1
    }


# ==================== CẬP NHẬT INTENT MAPPING VỚI FALLBACK ====================

INTENT_TO_RELATIONSHIPS = {
    # === BASIC 1-HOP QUERIES ===
    "get_movies_by_actor": {
        "relationships": ["PERSON_ACTED_IN_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_actors_of_movie": {
        "relationships": ["FILM_HAS_ACTOR"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_movies_by_director": {
        "relationships": ["PERSON_DIRECTED_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_director_of_movie": {
        "relationships": ["FILM_HAS_DIRECTOR"],
        "start_label": "FILM",
        "requires_two_entities": False
    },
    
    "get_same_school": {
        "relationships": ["PERSON_SAME_SCHOOL"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_same_location": {
        "relationships": ["PERSON_SAME_LOCATION"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    # === PROPERTY QUERIES ===
    "get_birthdate_spouse": {
        "special": "planner",
        "requires_two_entities": False
    },
    
    # === EDGE -> PROPERTY QUERIES ===
    "get_director_birthdate": {
        "relationships": ["FILM_HAS_DIRECTOR", "PERSON_BIRTH_DATE"],
        "start_label": "FILM",
        "requires_two_entities": False,
        "fallback_intent": "get_director_of_movie"  # Fallback về 1-hop
    },
    
    "get_actor_birthdate": {
        "relationships": ["FILM_HAS_ACTOR", "PERSON_BIRTH_DATE"],
        "start_label": "FILM",
        "requires_two_entities": False,
        "fallback_intent": "get_actors_of_movie"  # Fallback về 1-hop
    },
    
    "get_spouse_birthdate": {
        "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_DATE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_spouse_birthplace": {
        "relationships": ["PERSON_SPOUSE", "PERSON_BIRTH_PLACE"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    # === EDGE -> LIST PROPERTY (COLLECT) ===
    "get_actor_film_genres": {
        "relationships": ["PERSON_ACTED_IN_FILM", "FILM_GENRE"],
        "start_label": "PERSON",
        "requires_two_entities": False,
        "fallback_intent": "get_movies_by_actor"  # Fallback về phim
    },
    
    "get_director_film_genres": {
        "relationships": ["PERSON_DIRECTED_FILM", "FILM_GENRE"],
        "start_label": "PERSON",
        "requires_two_entities": False,
        "fallback_intent": "get_movies_by_director"  # Fallback về phim
    },
    
    # === MULTI-HOP QUERIES ===
    "get_spouse_movies": {
        "relationships": ["PERSON_SPOUSE", "PERSON_ACTED_IN_FILM"],
        "start_label": "PERSON",
        "requires_two_entities": False
    },
    
    "get_director_of_actor_movies": {
        "special": "custom",
        "query_func": graph_query_director_of_actor_movies,
        "requires_two_entities": False,
        "fallback_intent": "get_movies_by_actor"  # Fallback về phim của actor
    },
    
    "get_actors_in_director_movies": {
        "special": "custom",
        "query_func": graph_query_actors_in_director_movies,
        "requires_two_entities": False,
        "fallback_intent": "get_movies_by_director"  # Fallback về phim của đạo diễn
    },
    
    "get_schoolmate_movies": {
        "special": "custom",
        "query_func": graph_query_schoolmate_movies,
        "requires_two_entities": False,
        "fallback_intent": "get_same_school"  # Fallback về bạn học
    },
    
    # === INTERSECTION QUERIES (2 entities) ===
    "get_common_movies": {
        "special": "intersection",
        "query_func": graph_query_common_movies,
        "requires_two_entities": True
    },
    
    "get_common_directors": {
        "special": "custom",
        "query_func": graph_query_common_directors,
        "requires_two_entities": True
    },
    
    "get_collaboration_history": {
        "special": "custom",
        "query_func": graph_query_actor_collaboration_history,
        "requires_two_entities": True
    },
    
    "get_coactor_network": {
        "special": "custom",
        "query_func": graph_query_coactor_network,
        "requires_two_entities": False
    },
    
    "get_relationship_path": {
        "special": "custom",
        "query_func": graph_query_shortest_path,
        "requires_two_entities": True
    },
    
    "get_general_info": {
        "special": "custom",
        "query_func": graph_query_node_info,
        "requires_two_entities": False
    }
}



# ==================== 5. FORMATTER ====================

def format_graph_data_dynamic(graph_data, intent_type, entity_name=None):
    """Format data - tu dong xu ly tat ca cac loai data"""
    
    if not graph_data:
        return "KHONG TIM THAY thong tin lien quan trong du lieu."
    
    # === XU LY DICT (General Info, Relationship Path) ===
    if isinstance(graph_data, dict):
        # General info
        if 'properties' in graph_data or 'info_name' in graph_data:
            props = graph_data.get('properties', graph_data)
            key_map = {
                'info_birth_name': 'Ten khai sinh',
                'info_birth_date': 'Nam sinh',
                'info_occupation': 'Nghe nghiep',
                'info_spouse': 'Vo/Chong',
                'info_relatives': 'Nguoi than',
                'info_education': 'Hoc van',
                'info_height': 'Chieu cao',
                'info_nationality': 'Quoc tich',
                'info_birth_place': 'Noi sinh'
            }
            
            info_list = []
            name = props.get('info_name') or props.get('name') or entity_name
            info_list.append(f"Ten: {name}")
            
            for k, v in props.items():
                if k in key_map and v:
                    clean_v = str(v).replace('((', '').replace('))', '').replace('*', ',')
                    info_list.append(f"{key_map[k]}: {clean_v}")
            
            return "THONG TIN HO SO: " + ". ".join(info_list) + "."
        
        # Relationship path
        if 'description' in graph_data:
            return f"THONG TIN QUAN HE: {graph_data['description']}."
    
    # === XU LY LIST ===
    if isinstance(graph_data, list):
        if not graph_data:
            return "KHONG TIM THAY ket qua nao."
        
        # Neu list chua dict (complex structure)
        if isinstance(graph_data[0], dict):
            first_item = graph_data[0]
            
            if 'director' in first_item and 'film' in first_item:
                items = [f"{item['film']} (dao dien: {item['director']})" for item in graph_data]
                items_str = ", ".join(items)
                return f"DANH SACH DAO DIEN: {items_str}."
            
            elif 'schoolmate' in first_item and 'film' in first_item:
                items = [f"{item['schoolmate']}: {item['film']}" for item in graph_data]
                items_str = ", ".join(items)
                return f"DANH SACH PHIM: {items_str}."
            
            elif 'actor' in first_item and 'film' in first_item:
                items = [f"{item['actor']} (phim: {item['film']})" for item in graph_data]
                items_str = ", ".join(items)
                return f"DANH SACH DIEN VIEN: {items_str}."
            
            elif 'name' in first_item and 'distance' in first_item:
                items = [f"{item['name']} (khoang cach {item['distance']} buoc)" for item in graph_data]
                items_str = ", ".join(items)
                return f"MANG LUOI BAN DIEN: {items_str}."
            
            elif 'year' in first_item:
                items = [f"{item['film']} ({item['year'] or 'N/A'}) - dao dien {item['director']}" for item in graph_data]
                items_str = ", ".join(items)
                return f"LICH SU HOP TAC: {items_str}."
            
            elif 'films_with_actor1' in first_item:
                # Common directors
                items = [item['director'] for item in graph_data]
                items_str = ", ".join(items)
                return f"DANH SACH DAO DIEN CHUNG: {items_str}."
        
        # Simple list of strings
        items_str = ", ".join([str(item) for item in graph_data])
        
        # Generic templates
        templates = {
            "get_movies_by_actor": f"DANH SACH PHIM: Dien vien {entity_name} dong cac phim: {items_str}.",
            "get_actors_of_movie": f"DANH SACH DIEN VIEN: Phim {entity_name} co dien vien: {items_str}.",
            "get_movies_by_director": f"DANH SACH PHIM: Dao dien {entity_name} lam phim: {items_str}.",
            "get_director_of_movie": f"THONG TIN DAO DIEN: Dao dien phim {entity_name} la: {items_str}.",
            "get_same_school": f"DANH SACH: Nguoi hoc cung truong {entity_name}: {items_str}.",
            "get_same_location": f"DANH SACH: Nguoi cung que {entity_name}: {items_str}.",
            "get_common_movies": f"DANH SACH: {entity_name} dong chung phim: {items_str}.",
            "get_spouse_movies": f"DANH SACH PHIM: Vo/chong cua {entity_name} dong: {items_str}.",
        }
        
        return templates.get(intent_type, f"THONG TIN: {items_str}")
    
    # === XU LY STRING (Property values) ===
    return f"THONG TIN: {str(graph_data)}"


# ==================== 6. GENERATION ====================

def llm_paraphrase_graphrag(model_pack, formatted_sentence, question, use_finetuned=False, debug=False):
    """Viet lai cau tra loi"""
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
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.1,
            repetition_penalty=1.0,
            do_sample=False,
            num_beams=1,
            
        )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    response = re.sub(r'^[:\-\s]+', '', response)
    
    # Anti-hallucination check
    if "không" in response.lower() and "không" not in formatted_sentence.lower():
        if debug:
            print(f"[WARNING] Model hallucination detected!")
        if "DANH SACH" in formatted_sentence:
            parts = formatted_sentence.split(":")
            if len(parts) >= 2:
                response = parts[-1].strip().rstrip('.')
        else:
            response = formatted_sentence
    
    if debug:
        print(f"\n[LLM Input]: {formatted_sentence}")
        print(f"[LLM Output]: {response}")
    
    return response


# ==================== 7. PIPELINE CHINH ====================

def get_answer(question, model_pack, use_finetuned=False, debug=False):
    """Pipeline chinh - SU DUNG DYNAMIC QUERY BUILDER"""
    print(f"\n{'='*60}\nQUESTION: {question}")
    
    # 1. Extract entities
    entities = extract_entity_from_sentences(question)
    if not entities:
        return "Khong tim thay ten rieng."
    if debug:
        print(f"[1] Entities: {entities}")
    
    # 2. Detect intent
    intent = detect_intent(question)
    if debug:
        print(f"[2] Intent: {intent['intent']}")
    
    # 3. Entity linking
    linked_entities = entity_linking_graph(question)
    if not linked_entities:
        return "Khong tim thay thuc the phu hop trong Graph."
    if debug:
        print(f"[3] Linked: {linked_entities}")
    
    # 4. DYNAMIC QUERY ROUTING
    g_res = route_graph_query_dynamic(linked_entities, question, intent, debug=debug)
    if g_res['status'] == 'error':
        return g_res['message']
    if debug:
        print(f"[4] Graph Data: {g_res['data']}")
    
    # 5. Format & Generate
    formatted = format_graph_data_dynamic(g_res['data'], intent['intent'], g_res.get('entity_name'))
    final = llm_paraphrase_graphrag(model_pack, formatted, question, use_finetuned, debug)
    
    print(f"ANSWER: {final}\n{'='*60}\n")
    return final


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("\n>>> INITIALIZING DYNAMIC QUERY SYSTEM...")
    print("Su dung Dynamic Relationship-Based Query Builder")

    # Load Model
    try:
        llm_pack = load_llm_model(fine_tune=False)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        exit(1)

    # Test queries
    test_questions = [
        # Basic
        "Trấn Thành đóng phim gì?",
        "Ai đạo diễn phim Bố Già?",
        
        # Properties
        "Vợ của Trấn Thành là ai?",
        "Trấn Thành sinh năm bao nhiêu?",
        
        # Property -> Property chains
        "Vợ của Trấn Thành sinh năm nào?",
        "Vợ của Trấn Thành quê ở đâu?",
        
        # Edge -> Property chains
        "Đạo diễn phim Bố Già sinh năm nào?",
        
        # Edge -> List Property (COLLECT)
        "Ninh Dương Lan Ngọc đã đóng các thể loại phim gì?",
        "Trấn Thành đạo diễn những thể loại phim nào?",
        
        # Multi-hop
        "Ai đạo diễn các phim của Trấn Thành?",
        "Vợ của Trấn Thành đóng phim gì?",
        "Bạn học của Trấn Thành đóng phim nào?",
        
        # Intersection
        "Trấn Thành và Hari Won đóng chung phim nào?",
        "Trấn Thành và Hari Won làm việc với đạo diễn nào?",
        
        # Network
        "Ai đã đóng phim với Trấn Thành?",
    ]
    
    for q in test_questions:
        get_answer(q, llm_pack, use_finetuned=args.use_finetuned, debug=True)
    
    close_driver()
    print("\nNeo4j driver closed.")