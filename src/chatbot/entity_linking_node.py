from thefuzz import fuzz
import unicodedata
import re
from underthesea import pos_tag, word_tokenize

from src.chatbot.extract_entities_from_question import extract_entities, normalize_text

# ==================== CONSTANTS ====================

ENTITY_TYPE_HINTS = {
    'phim': 'film',
    'bộ phim': 'film',
    'bộ': 'film',
    'tập': 'film',
    'chiếu': 'film',
    'xem': 'film',
    'flim': 'film',
    'đóng': 'person',
    'diễn viên': 'person',
    'diễn': 'person',
    'đạo diễn': 'person',
    'sản xuất': 'person',
    'dao dien': 'person',
    'san xuat': 'person',
    'dien vien': 'person',
}

FILM_KEYWORDS = {'phim', 'bộ', 'tập', 'chiếu', 'xem', 'flim', 'lịch chiếu', 'rating', 'sao'}
PERSON_KEYWORDS = {'diễn viên', 'đạo diễn', 'sản xuất', 'đóng', 'dien vien', 'dao dien', 'san xuat'}




def get_tokens(text):
    
    text_norm = normalize_text(text)
    return text_norm.split()


def tokens_match_ordered(entity_tokens, node_tokens):
    """
    Kiểm tra xem tokens của entity có khớp đúng thứ tự với node không
    
    
    """
    # Exact token match 
    if entity_tokens == node_tokens:
        return True
    
    #  prefix/suffix của node tokens
    node_str = ' '.join(node_tokens)
    entity_str = ' '.join(entity_tokens)
    
    if entity_str in node_str:
        # Kiểm tra xem có đúng thứ tự không
        entity_idx = node_str.find(entity_str)
        if entity_idx == 0 or node_str[entity_idx - 1] == ' ':
            if entity_idx + len(entity_str) == len(node_str) or node_str[entity_idx + len(entity_str)] == ' ':
                return True
    
    return False


def tokens_match_unordered(entity_tokens, node_tokens):
    """
    Kiểm tra xem tokens có khớp (không quan tâm thứ tự)
    
   
    """
    return set(entity_tokens) == set(node_tokens)



def detect_entity_type_from_context(entity, question):
    """
    Nhận diện loại entity từ ngữ cảnh trong câu hỏi
    """
    question_norm = normalize_text(question)
    entity_norm = normalize_text(entity)
    
    try:
        entity_idx = question_norm.index(entity_norm)
    except ValueError:
        entity_idx = -1
    
    context_window = question_norm.lower()
    
    type_scores = {
        'film': 0,
        'person': 0,
    }
    
    # === FILM INDICATORS ===
    for kw in ['phim', 'bo', 'tap', 'chieu', 'xem', 'lat mat', 'dien', 'dong']:
        if kw in context_window:
            type_scores['film'] += 1
    
    if 'phim' in context_window:
        type_scores['film'] += 5
    
    # === PERSON INDICATORS ===
    for kw in ['dien vien', 'dao dien', 'san xuat', 'cung', 'vs', 'va', 'hop tac', 'la ai', 'ai', 'dong', 'cua']:
        if kw in context_window:
            type_scores['person'] += 1
    
    if 'dien vien' in context_window or 'dao dien' in context_window:
        type_scores['person'] += 5
    
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    
    result = [t[0] for t in sorted_types if t[1] > 0]
    if not result:
        result = ['person', 'film']
    
    return result




def fuzzy_match_node(entity, node_name, node_type=None, expected_types=None, threshold=80):
    """
    So khớp mờ giữa entity và tên node
    
    Ưu tiên:
    1. Exact match (không dấu) + đúng thứ tự
    2. Exact match (không dấu) + sai thứ tự → PENALTY
    3. Substring match + đúng thứ tự
    4. Fuzzy ratio
    """
    entity_norm = normalize_text(entity)
    node_norm = normalize_text(node_name)
    
    entity_tokens = get_tokens(entity)
    node_tokens = get_tokens(node_name)
    
    
    base_score = 0
    match_type = None
    
    # === LEVEL 1: EXACT TOKEN MATCH ===
    if entity_norm == node_norm:
        # Perfect match
        base_score = 100
        match_type = "exact_ordered"
    
    
    # === LEVEL 2: TOKEN MATCH (CHECK ORDER) ===
    elif tokens_match_ordered(entity_tokens, node_tokens):
        # Tokens khớp, đúng thứ tự
        base_score = 95
        match_type = "tokens_ordered"
    
    elif tokens_match_unordered(entity_tokens, node_tokens):
        # Tokens khớp, sai thứ tự
        base_score = 70  # giam diem 
        match_type = "tokens_unordered"
    
    # substring match
    elif entity_norm in node_norm:
        base_score = 85
        match_type = "substring_entity_in_node"
    
    elif node_norm in entity_norm:
        base_score = 80
        match_type = "substring_node_in_entity"
    
    # fuzzy ratio
    else:
        score_ratio = fuzz.ratio(entity_norm, node_norm)
        score_token = fuzz.token_sort_ratio(entity_norm, node_norm)
        
        # Chọn lower score (để tránh boost false positive)
        base_score = min(score_ratio, score_token)
        match_type = "fuzzy"
    
    
    score = base_score
    
    if expected_types and node_type:
        if node_type in expected_types:
            type_rank = expected_types.index(node_type)
            if type_rank == 0:
                score += 15  # Boost nhưng không quá
            else:
                score += 8
        else:
            score -= 10  
    
    matched = score >= threshold
    
    return score, matched, match_type


# ==================== FUZZY MATCHING FOR LINKING ====================

def exact_match_entity(entity_name, graph, debug=False):
    """
    Tìm exact match trong graph (không fuzzy)
    
    Args:
        entity_name (str): Tên entity
        graph: NetworkX graph
        debug (bool): Debug mode
    
    Returns:
        dict hoặc None: Node match hoặc None
    """
    
    if not graph or not entity_name:
        return None
    
    entity_norm = normalize_text(entity_name)
    
    # Duyệt qua tất cả nodes
    for node_id in graph.nodes():
        node_name = get_node_name(graph, node_id)
        node_norm = normalize_text(node_name)
        
        # EXACT MATCH (100%)
        if entity_norm == node_norm:
            if debug:
                print(f"  [EXACT] Found: {node_name} (100% match)")
            
            return {
                'node_id': node_id,
                'node_name': node_name,
                'score': 100,
                'type': graph.nodes[node_id].get('type'),
                'match_type': 'exact'
            }
    
    return None


def fuzzy_match_entity(entity_name, graph, question="", threshold=70, debug=False):
    """
    Matching entity với nodes trong graph sử dụng fuzzy matching
    
    ✅ CẢI TIẾN: Sử dụng logic từ fuzzy_match_node()
    - Level 1: Exact token match
    - Level 2: Token match (check order)
    - Level 3: Substring match
    - Level 4: Fuzzy ratio (tránh false positive bằng min score)
    - Type boosting
    
    Args:
        entity_name (str): Tên entity cần linking
        graph: NetworkX graph
        question (str): Câu hỏi (để xác định context type)
        threshold (int): Ngưỡng matching (0-100)
        debug (bool): Debug mode
    
    Returns:
        list: Danh sách candidates
    """
    
    if not graph or not entity_name:
        return []
    
    #  BƯỚC 1: DETECT EXPECTED TYPE FROM CONTEXT
    if question:
        expected_types = detect_entity_type_from_context(entity_name, question)
    else:
        expected_types = ['person', 'film']
    
    if debug:
        print(f"  [TYPE] Expected types: {expected_types}")
    
    entity_norm = normalize_text(entity_name)
    entity_tokens = get_tokens(entity_name)
    
    node_candidates = {}
    node_types = {}
    
    # ✅ BƯỚC 2: LỌC NODES THEO EXPECTED TYPE
    for node_id in graph.nodes():
        node_name = get_node_name(graph, node_id)
        node_type = get_node_type(graph, node_id)
        
        # CHỈ giữ lại nodes có type khớp
        if node_type in expected_types and node_name and node_name.strip():
            node_candidates[node_name] = node_id
            node_types[node_name] = node_type
    
    if not node_candidates:
        if debug:
            print(f"  ❌ No nodes found with expected types: {expected_types}")
        return []
    
    if debug:
        print(f"  [FILTER] Filtered {len(node_candidates)} nodes by type")
    
    # ✅ BƯỚC 3: XỬ LÝ THRESHOLD TỪ TỪNG TYPE
    if 'film' in expected_types and len(expected_types) == 1:
        type_threshold = 85  # 🔥 FILM: Threshold cao
    else:
        type_threshold = threshold  # Person hoặc mixed: 70
    
    if debug:
        print(f"  [THRESHOLD] Using threshold: {type_threshold}")
    
    # ✅ BƯỚC 4: MATCHING TỪNG NODE VỚI LOGIC fuzzy_match_node
    results = []
    
    for node_name, node_id in node_candidates.items():
        node_type = node_types[node_name]
        
        # Gọi fuzzy_match_node để tính score
        score, matched, match_type = fuzzy_match_node(
            entity=entity_name,
            node_name=node_name,
            node_type=node_type,
            expected_types=expected_types,
            threshold=type_threshold  # Sử dụng type_threshold
        )
        
        if matched:
            results.append({
                'node_id': node_id,
                'node_name': node_name,
                'score': score,
                'type': node_type,
                'match_type': match_type
            })
    
    if debug and results:
        print(f"  [FUZZY] Matched {len(results)} candidates with threshold {type_threshold}:")
        for r in results[:5]:
            print(f"    → {r['node_name']} (score: {r['score']}, type: {r['type']}, match: {r['match_type']})")
    elif debug:
        print(f"  ❌ No matches above threshold {type_threshold}")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def get_node_name(graph, node_id):
    """Lấy tên hiển thị của node"""
    if node_id not in graph.nodes():
        return str(node_id)
    
    node_data = graph.nodes[node_id]
    name = (
        node_data.get('name') or
        node_data.get('title') or
        node_data.get('full_name') or
        str(node_id)
    )
    return name


def get_node_type(graph, node_id):
    """Lấy loại node"""
    if node_id not in graph.nodes():
        return None
    return graph.nodes[node_id].get('type')


# ==================== ENTITY LINKING MAIN FUNCTION ====================

def entity_linking(entities_list, graph, question="", top_k=3, threshold=70, debug=False):
    """
    Linking danh sách entities với các nodes trong graph
    
    STRATEGY:
    1. Thử EXACT MATCH trước
    2. Nếu không có → thử FUZZY MATCH
    3. Chỉ lấy tốt nhất (top 1)
    
    Args:
        entities_list (list): Danh sách entities
        graph: NetworkX graph
        question (str): Câu hỏi gốc
        top_k (int): Số candidates tối đa
        threshold (int): Ngưỡng fuzzy matching
        debug (bool): Debug mode
    
    Returns:
        dict: {entity_name: [candidates]}
    """
    
    if not entities_list or not graph:
        return {}
    
    linked_results = {}
    
    for entity in entities_list:
        if debug:
            print(f"\n[LINKING] Entity: '{entity}'")
        
        # ✅ STEP 1: EXACT MATCH
        exact_match = exact_match_entity(entity, graph, debug=debug)
        
        if exact_match:
            # Đã tìm thấy exact match → không fuzzy nữa
            linked_results[entity] = [exact_match]
            if debug:
                print(f"  ✅ Exact match found - skipping fuzzy matching")
            continue
        
        # ✅ STEP 2: FUZZY MATCH (chỉ nếu không có exact)
        if debug:
            print(f"  No exact match - trying fuzzy matching...")
        
        candidates = fuzzy_match_entity(entity, graph, question=question, threshold=threshold, debug=debug)
        
        if candidates:
            linked_results[entity] = candidates[:top_k]
        else:
            if debug:
                print(f"  ❌ No match found (threshold: {threshold})")
            linked_results[entity] = []
    
    return linked_results


def get_best_match(candidates):
    """
    Lấy candidate tốt nhất (score cao nhất)
    
    Args:
        candidates (list): Danh sách candidates
    
    Returns:
        dict hoặc None: Candidate tốt nhất
    """
    if not candidates or len(candidates) == 0:
        return None
    
    # Sort by score, descending
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return sorted_candidates[0]


# ==================== UPDATED entity_linking_question ====================

def entity_linking_question(question, graph=None, threshold=70, debug=False):
    """
    Trích xuất + linking entities từ câu hỏi
    
    Args:
        question (str): Câu hỏi
        graph: NetworkX graph (optional)
        threshold: Ngưỡng fuzzy matching
        debug: Debug mode
    
    Returns:
        dict: {
            'entities': list,
            'linked': {entity: [candidates]}
        }
    """
    
    entities = extract_entities(question)
    
    result = {
        'entities': entities,
        'linked': {}
    }
    
    # Nếu có graph, thực hiện linking
    if graph:
        linked = entity_linking(entities, graph, question=question, 
                               top_k=5, threshold=threshold, debug=debug)
        result['linked'] = linked
    
    return result


