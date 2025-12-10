import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import re

# ==================== IMPORTS ====================

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from load_graph import load_graphs
    from chatbot.entity_linking_node import (
        extract_entities,
        get_best_match,
        normalize_text, 
        entity_linking
    )
    from chatbot.extract_entities_from_question import VIETNAMESE_STOPWORDS
    from chatbot.graph_query import (
        graph_query_movies_by_actor,
        graph_query_actors_of_movie,
        graph_query_common_movies,
        graph_query_collaborations,
        graph_query_node_info,
        get_node_name,
        get_node_type, 
        graph_query_actor_via_collaboration,
        graph_query_indirect_collaboration,
        graph_query_movie_chain,
        graph_query_actor_via_movie, 
        graph_query_movie_via_actor
    )
except ImportError as e:
    print(f"⚠ Warning: Could not import from chatbot modules - {e}")


# ==================== ENTITY EXTRACTION ====================

def filter_entities(entities_list):
    """Lọc bỏ các từ không phải entity thật"""
    filtered = []
    for entity in entities_list:
        entity_clean = normalize_text(entity).lower()
        if entity_clean not in VIETNAMESE_STOPWORDS and len(entity_clean) > 1:
            filtered.append(entity)
    return filtered


# ==================== INTENT DETECTION ====================

def detect_intent(question):
    """Phát hiện ý định của câu hỏi"""
    question_norm = normalize_text(question).lower()
    
    intents = {
        'actor_movies': {
            'keywords': ['dong phim', 'phim nao', 'tham gia phim', 'co mat trong phim'],
            'pattern': ['dong phim gi', 'dong nhung phim nao', 'tham gia phim nao'],
            'priority': 3
        },
        'movie_actors': {
            'keywords': ['dien vien', 'cast', 'dong chinh', 'vai dien'],
            'pattern': ['phim co ai', 'dien vien nao', 'ai dong phim', 'co nhung dien vien nao'],
            'priority': 3
        },
        'common_movies': {
            'keywords': ['dong chung', 'phim chung', 'cung dong', 'chung'],
            'pattern': ['dong chung phim nao', 'cung dong trong phim', 'phim chung'],
            'priority': 5
        },
        'collaboration': {
            'keywords': ['hop tac', 'cung lam viec', 'dong vien', 'co su tham gia'],
            'pattern': ['ai hop tac', 'hop tac voi ai', 'cung lam viec voi ai'],
            'priority': 4
        },
        'info': {
            'keywords': ['la ai', 'thong tin', 'chi tiet', 've'],
            'pattern': ['la ai', 'thong tin ve', 'cho biet ve'],
            'priority': 2
        },
        'actor_via_collaboration': {
            'keywords': ['nhung ai khac', 'hop tac voi nhung ai', 'ngoai ai'],
            'pattern': ['nhung ai khac', 'hop tac voi nhung ai', 'ngoai ai', 'lam viec voi nhung ai khac'],
            'priority': 6
        },  
        'indirect_collaboration': {
            'keywords': ['hop tac giua', 'cau noi', 'dong phim giua', 'co lien ket voi', 'lien ket'],
            'pattern': ['hop tac giua', 'cau noi', 'dong phim giua', 'co lien ket voi', 'lien ket voi ca'],
            'priority': 5
        },
        'movie_chain': {
            'keywords': ['chuoi phim', 'ket noi phim', 'ai lan ai'],
            'pattern': ['chuoi phim giua', 'ket noi phim giua', 'co tham gia giua'],
            'priority': 2
        },
        'actor_via_movie': {
            'keywords': ['ngoai', 'khac', 'con ai'],
            'pattern': ['ngoai', 'con ai khac', 'ai khac dong'],
            'priority': 3
        },
        'movie_via_actor': {
            'keywords': ['phim khac', 'phim nao khac'],
            'pattern': ['phim khac', 'phim nao khac', 'con phim nao'],
            'priority': 3
        }
    }
    
    intent_scores = {}
    
    for intent_name, intent_info in intents.items():
        score = 0
        keywords = intent_info['keywords']
        priority = intent_info.get('priority', 1)
        
        for pattern in intent_info['pattern']:
            if pattern in question_norm:
                score += 10
        
        for kw in keywords:
            if kw in question_norm:
                score += 3
        
        score *= priority
        
        if score > 0:
            intent_scores[intent_name] = score
    
    # Special rules
    if 'nhung ai khac' in question_norm or 'hop tac voi nhung ai' in question_norm:
        intent_scores['actor_via_collaboration'] = intent_scores.get('actor_via_collaboration', 0) + 100
    
    if 'lien ket voi ca' in question_norm or 'hop tac giua' in question_norm:
        intent_scores['indirect_collaboration'] = intent_scores.get('indirect_collaboration', 0) + 100
    
    if not intent_scores:
        return {
            'intent': 'unknown',
            'keywords': [],
            'confidence': 0.0
        }
    
    best_intent = max(intent_scores.items(), key=lambda x: x[1])
    
    return {
        'intent': best_intent[0],
        'keywords': [],
        'confidence': min(best_intent[1] / 50.0, 1.0)
    }


# ==================== ENTITY LINKING ====================

def link_entities_to_graph(entities_list, graph, question, debug=False):
    """Linking entities với graph nodes"""
    
    if not entities_list or not graph:
        return {}
    
    entities_filtered = filter_entities(entities_list)
    
    if debug:
        print(f"   Entities sau khi lọc: {entities_filtered}")
    
    if not entities_filtered:
        return {}
    
    linked_results = entity_linking(entities_filtered, graph, question=question, 
                                    top_k=5, threshold=70, debug=debug)
    
    linked = {}
    for entity, candidates in linked_results.items():
        if not candidates:
            continue
        
        best_match = get_best_match(candidates)
        
        if best_match:
            linked[entity] = best_match['node_id']
            if debug:
                node_name = get_node_name(graph, best_match['node_id'])
                node_type = get_node_type(graph, best_match['node_id'])
                print(f"   [LINKED] '{entity}' -> '{node_name}' (ID: {best_match['node_id']}, Type: {node_type})")
    
    return linked


# ==================== GRAPH QUERY ROUTER ====================

def route_graph_query(entities_dict, graph_bipartite, graph_collab, question, intent, debug=False):
    """Điều hướng đến hàm graph query phù hợp với graph thích hợp"""
    
    if debug:
        print(f"\n[ROUTER] Intent: {intent['intent']}")
        print(f"[ROUTER] Linked entities: {entities_dict}")
    
    if not entities_dict:
        return {
            'status': 'error',
            'data': None,
            'message': 'Không tìm thấy entity nào khớp với cơ sở dữ liệu.'
        }
    
    intent_type = intent['intent']
    entity_names = list(entities_dict.keys())
    entity_ids = list(entities_dict.values())
    
    if intent_type == 'actor_movies':
        if len(entity_ids) > 0:
            actor_id = entity_ids[0]
            actor_name = entity_names[0]
            try:
                movies = graph_query_movies_by_actor(graph_bipartite, actor_id, get_names=True, debug=debug)
                return {
                    'status': 'success',
                    'data': movies,
                    'message': f"Phim của {actor_name}",
                    'entity_name': actor_name
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f"Lỗi khi query: {str(e)}"
                }
    
    elif intent_type == 'movie_actors':
        if len(entity_ids) > 0:
            movie_id = entity_ids[0]
            movie_name = entity_names[0]
            try:
                actors = graph_query_actors_of_movie(graph_bipartite, movie_id, get_names=True, debug=debug)
                return {
                    'status': 'success',
                    'data': actors,
                    'message': f"Diễn viên của {movie_name}",
                    'entity_name': movie_name
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f"Lỗi khi query: {str(e)}"
                }
    
    elif intent_type == 'common_movies':
        if len(entity_ids) >= 2:
            actor1_id, actor2_id = entity_ids[0], entity_ids[1]
            actor1_name, actor2_name = entity_names[0], entity_names[1]
            try:
                common = graph_query_common_movies(graph_collab, actor1_id, actor2_id, debug=debug)
                return {
                    'status': 'success',
                    'data': common,
                    'message': f"Phim chung của {actor1_name} và {actor2_name}",
                    'entity_name': f"{actor1_name} và {actor2_name}"
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f"Lỗi khi query: {str(e)}"
                }
        else:
            return {
                'status': 'error',
                'data': None,
                'message': 'Cần ít nhất 2 diễn viên để tìm phim chung.'
            }
    
    elif intent_type == 'info':
        if len(entity_ids) > 0:
            node_id = entity_ids[0]
            node_name = entity_names[0]
            try:
                info = graph_query_node_info(graph_bipartite, node_id, debug=False)
                return {
                    'status': 'success',
                    'data': info,
                    'message': f"Thông tin về {node_name}",
                    'entity_name': node_name
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f"Lỗi khi query: {str(e)}"
                }
    
    elif intent_type == 'collaboration':
        if len(entity_ids) > 0:
            actor_id = entity_ids[0]
            actor_name = entity_names[0]
            try:
                collaborators = graph_query_collaborations(graph_collab, actor_id, get_names=True, debug=debug)
                return {
                    'status': 'success',
                    'data': collaborators,
                    'message': f"Diễn viên hợp tác với {actor_name}",
                    'entity_name': actor_name
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f"Lỗi khi query: {str(e)}"
                }
    
    elif intent_type == 'actor_via_movie':
        # Cần: 1 phim + 1 diễn viên (hoặc chỉ 1 phim)
        if len(entity_ids) >= 1:
            movie_ref = entity_names[0]
            actor_ref = entity_names[1] if len(entity_names) >= 2 else None
            try:
                others = graph_query_actor_via_movie(
                    graph_bipartite, movie_ref, exclude_actor=actor_ref, debug=debug
                )
                msg = f"Diễn viên khác trong phim {movie_ref}"
                if actor_ref:
                    msg += f" (ngoài {actor_ref})"
                return {
                    'status': 'success',
                    'data': others,
                    'message': msg,
                    'entity_name': movie_ref
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'movie_via_actor':
        # Cần: 1 diễn viên + 1 phim (hoặc chỉ 1 diễn viên)
        if len(entity_ids) >= 1:
            actor_ref = entity_names[0]
            movie_ref = entity_names[1] if len(entity_names) >= 2 else None
            try:
                others = graph_query_movie_via_actor(
                    graph_bipartite, actor_ref, exclude_movie=movie_ref, debug=debug
                )
                msg = f"Phim khác của {actor_ref}"
                if movie_ref:
                    msg += f" (ngoài {movie_ref})"
                return {
                    'status': 'success',
                    'data': others,
                    'message': msg,
                    'entity_name': actor_ref
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'actor_via_collaboration':
        # Cần: 2 diễn viên
        if len(entity_ids) >= 2:
            actor1 = entity_names[0]
            actor2 = entity_names[1]
            try:
                others = graph_query_actor_via_collaboration(
                    graph_collab, actor2, exclude_actor=actor1, debug=debug
                )
                return {
                    'status': 'success',
                    'data': others,
                    'message': f"Diễn viên khác hợp tác với {actor2} (ngoài {actor1})",
                    'entity_name': actor2
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'indirect_collaboration':
        # Cần: 2 diễn viên
        if len(entity_ids) >= 2:
            actor1 = entity_names[0]
            actor2 = entity_names[1]
            try:
                bridges = graph_query_indirect_collaboration(
                    graph_collab, actor1, actor2, debug=debug
                )
                return {
                    'status': 'success',
                    'data': bridges,
                    'message': f"Diễn viên cầu nối giữa {actor1} và {actor2}",
                    'entity_name': f"{actor1} và {actor2}"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'movie_chain':
        # Cần: 1 diễn viên + 1 phim
        if len(entity_ids) >= 2:
            actor_ref = entity_names[0]
            movie_ref = entity_names[1]
            try:
                common = graph_query_movie_chain(
                    graph_bipartite, actor_ref, movie_ref, debug=debug
                )
                return {
                    'status': 'success',
                    'data': common,
                    'message': f"Phim chung của {actor_ref} với diễn viên từ {movie_ref}",
                    'entity_name': actor_ref
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    else:
        return {
            'status': 'error',
            'data': None,
            'message': f'Intent "{intent_type}" không được hỗ trợ.'
        }


# ==================== LLM MODEL ====================

def load_llm_model(model_path="Qwen/Qwen2.5-0.5B-Instruct"):
    """Tải model"""
    print(f"[INFO] Đang tải model {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    print("[INFO] Tải model thành công!")
    return model, tokenizer


# ==================== FORMAT DATA ====================

def format_graph_data_strictly(graph_data, intent_type, entity_name=None):
    """Format dữ liệu thành câu hoàn chỉnh"""
    
    if isinstance(graph_data, list):
        if len(graph_data) == 0:
            return "KHÔNG TÌM THẤY DỮ LIỆU TRONG CƠ SỞ DỮ LIỆU"
        
        items_str = ", ".join(graph_data)
        
        if intent_type == 'actor_movies':
            return f"THÔNG TIN: {entity_name} đã tham gia các phim sau: {items_str}"
        
        elif intent_type == 'movie_actors':
            return f"THÔNG TIN: Phim {entity_name} có các diễn viên: {items_str}"
        
        elif intent_type == 'common_movies':
            return f"THÔNG TIN: {entity_name} đã cùng đóng các phim: {items_str}"
        
        elif intent_type == 'collaboration':
            return f"THÔNG TIN: {entity_name} đã hợp tác với các diễn viên: {items_str}"
        
        elif intent_type == 'actor_via_collaboration':
            return f"THÔNG TIN: Các diễn viên khác hợp tác với {entity_name}: {items_str}"
        
        elif intent_type == 'indirect_collaboration':
            return f"THÔNG TIN: Các diễn viên cầu nối với {entity_name} là: {items_str}"
        
        elif intent_type == 'movie_chain':
            return f"THÔNG TIN: Chuỗi phim kết nối với {entity_name} bao gồm: {items_str}"
        
        elif intent_type == 'actor_via_movie':
            return f"THÔNG TIN: Các diễn viên khác trong phim {entity_name}: {items_str}"
        
        elif intent_type == 'movie_via_actor':
            return f"THÔNG TIN: Các phim khác của {entity_name}: {items_str}"
        
        else:
            return f"THÔNG TIN: Danh sách kết quả: {items_str}"
    
    elif isinstance(graph_data, dict):
        core_info = graph_data
        if 'attributes' in graph_data:
            core_info = graph_data['attributes'].get('info', graph_data)
        elif 'info' in graph_data:
            core_info = graph_data['info']
        
        key_map = {
            'name': 'Tên', 
            'birth_name': 'Tên thật',
            'birth_date': 'Ngày sinh', 
            'birth_place': 'Nơi sinh',
            'occupation': 'Nghề nghiệp', 
            'spouse': 'Vợ/Chồng', 
            'nationality': 'Quốc tịch',
            'height': 'Chiều cao', 
            'education': 'Học vấn',
            'active_years': 'Năm hoạt động'
        }
        
        facts = []
        for k, v in core_info.items():
            if k in key_map and v:
                clean_v = str(v).replace('*', '').replace('((', '(').replace('))', ')')
                facts.append(f"{key_map[k]} là {clean_v}")
        
        if not facts:
            return "KHÔNG CÓ THÔNG TIN CHI TIẾT"
        
        facts_str = ". ".join(facts)
        return f"THÔNG TIN: {facts_str}."
    
    else:
        return "KHÔNG TÌM THẤY DỮ LIỆU"


# ==================== LLM PARAPHRASE ====================

def llm_paraphrase_only(model_pack, formatted_sentence, question, debug=False):
    """LLM paraphrase câu đã format"""
    
    model, tokenizer = model_pack
    
    if "KHÔNG TÌM THẤY" in formatted_sentence or "KHÔNG CÓ THÔNG TIN" in formatted_sentence:
        return "Không tìm thấy thông tin trong cơ sở dữ liệu."
    
    system_prompt = """Bạn là trợ lý AI. Nhiệm vụ duy nhất: VIẾT LẠI câu thông tin bằng cách diễn đạt tự nhiên hơn.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ sử dụng thông tin có trong câu "THÔNG TIN" bên dưới
2. KHÔNG ĐƯỢC thêm: năm, đạo diễn, thể loại, đánh giá, hay BẤT KỲ chi tiết nào không có trong câu gốc
3. KHÔNG ĐƯỢC bỏ bất kỳ tên nào trong danh sách
4. CHỈ thay đổi cách diễn đạt cho tự nhiên hơn
5. Giữ nguyên TẤT CẢ tên người, tên phim trong câu gốc"""

    user_prompt = f"""CÂU THÔNG TIN GỐC:
{formatted_sentence}

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

YÊU CẦU: Viết lại câu thông tin trên cho tự nhiên, KHÔNG thêm/bớt thông tin."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "assistant" in response.lower():
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    response = response.strip()
    response = re.sub(r'^[:\-\s]+', '', response)
    
    if debug:
        print(f"\n[LLM DEBUG]")
        print(f"   Input sentence: {formatted_sentence}")
        print(f"   Output response: {response}")
    
    return response


# ==================== VALIDATION ====================

def validate_response(response, formatted_sentence, graph_data, debug=False):
    """Kiểm tra response có tuân thủ không"""
    
    if debug:
        print(f"\n[VALIDATION]")
    
    if isinstance(graph_data, list):
        required_items = graph_data
    elif isinstance(graph_data, dict):
        required_items = []
        core_info = graph_data.get('attributes', {}).get('info', graph_data.get('info', graph_data))
        for v in core_info.values():
            if v and isinstance(v, str):
                required_items.append(str(v))
    else:
        required_items = []
    
    if not required_items:
        return True, response
    
    response_norm = normalize_text(response).lower()
    
    missing_items = []
    for item in required_items:
        item_norm = normalize_text(str(item)).lower()
        if item_norm not in response_norm:
            missing_items.append(item)
    
    hallucination_keywords = [
        'năm', 'rating', 'giải thưởng',
        'nổi tiếng', 'xuất sắc', 'blockbuster', 'doanh thu'
    ]
    
    found_hallucinations = []
    for kw in hallucination_keywords:
        if kw in response.lower():
            found_hallucinations.append(kw)
    
    if missing_items or found_hallucinations:
        if debug:
            print(f"   [ERROR] Validation failed!")
            if missing_items:
                print(f"     Missing items: {missing_items}")
            if found_hallucinations:
                print(f"     Hallucinations: {found_hallucinations}")
        
        fallback = formatted_sentence.replace("THÔNG TIN: ", "").replace("THÔNG TIN:", "")
        return False, fallback
    
    if debug:
        print(f"   [OK] Validation passed")
    
    return True, response


# ==================== MAIN PIPELINE ====================

def get_answer(question, G_bipartite, G_collab, model_pack, debug=False):
    """Pipeline chính"""
    
    print(f"\n{'='*100}")
    print(f"QUESTION: {question}")
    print(f"{'='*100}")
    
    print(f"\n[STEP 1] Extracting entities...")
    entities_list = extract_entities(question)
    print(f"[OK] Extracted: {entities_list}")
    
    if not entities_list:
        return "Không tìm thấy entity nào trong câu hỏi."
    
    print(f"\n[STEP 2] Detecting intent...")
    intent = detect_intent(question)
    print(f"[OK] Intent: {intent['intent']} (confidence: {intent['confidence']:.0%})")
    
    print(f"\n[STEP 3] Linking entities to graph...")
    linked_entities = link_entities_to_graph(entities_list, G_bipartite, question, debug=debug)
    print(f"[OK] Linked: {linked_entities}")
    
    if not linked_entities:
        return "Không tìm thấy thông tin trong cơ sở dữ liệu."
    
    print(f"\n[STEP 4] Querying graph...")
    graph_result = route_graph_query(linked_entities, G_bipartite, G_collab, question, intent, debug=debug)
    print(f"[OK] Query result: {graph_result['message']}")
    
    if graph_result['status'] == 'error':
        return f"{graph_result['message']}"
    
    print(f"\n[STEP 5] Formatting graph data into complete sentence...")
    formatted_sentence = format_graph_data_strictly(
        graph_result['data'], 
        intent['intent'],
        graph_result.get('entity_name')
    )
    print(f"[OK] Formatted: {formatted_sentence[:100]}...")
    
    print(f"\n[STEP 6] LLM paraphrasing...")
    paraphrased = llm_paraphrase_only(
        model_pack,
        formatted_sentence,
        question,
        debug=debug
    )
    
    print(f"\n[STEP 7] Validating response...")
    is_valid, final_answer = validate_response(
        paraphrased,
        formatted_sentence,
        graph_result['data'],
        debug=debug
    )
    
    if not is_valid:
        print(f"[WARN] Validation failed - using formatted sentence as fallback")
    
    print(f"\n{'='*100}")
    print(f"ANSWER: {final_answer}")
    print(f"{'='*100}\n")
    
    return final_answer


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("CHATBOT PIPELINE - LLM PARAPHRASE ONLY (NO HALLUCINATION)")
    print("="*100)
    
    print("\n[INIT] Loading graphs...")
    try:
        G_actor_collab, G_bipartite = load_graphs()
        print("[OK] Both graphs loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading graphs: {e}")
        exit(1)
    
    print("\n[INIT] Loading LLM model...")
    try:
        llm_pack = load_llm_model("Qwen/Qwen2.5-0.5B-Instruct")
        print("[OK] LLM model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading LLM: {e}")
        exit(1)
    
    # Test questions
    test_questions = [
        "Phim nào có sự tham gia của cả Hoàng Sơn (diễn viên) và Nam Thư?"
    ]
    
    # Run pipeline
    for q in test_questions:
        try:
            answer = get_answer(q, G_bipartite, G_actor_collab, llm_pack, debug=True)
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*100 + "\n")