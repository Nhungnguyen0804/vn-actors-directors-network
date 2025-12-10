from collections import Counter, defaultdict,OrderedDict
import re
from underthesea import ner as uts_ner
from underthesea import pos_tag as uts_pos_tag
from underthesea import word_tokenize as uts_word_tokenize
import unicodedata
import json
from src.constant import WIKI_ENRICHMENT, BIPARTITE_JSON
from src.data_prep.load_graph import load_graph,load_bipartite_graph_and_nodes
from src.nlp.text_utils import normalize_text_for_nlp,normalize_entity_name,normalize_type

# B-XXX = Begin entity (bắt đầu 1 thực thể)
# I-XXX = Inside entity (các từ tiếp theo trong cùng thực thể)
# ====================================
B = load_graph(BIPARTITE_JSON)
B, person_list, film_list = load_bipartite_graph_and_nodes(B)

# ====================================
def load_jsonl_to_dict(path, name = "name"):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            name = obj.get(name)
            if name:
                result[name] = obj
    return result



wiki_enrich = load_jsonl_to_dict(WIKI_ENRICHMENT)
# ===========================================
# lọc khỏi vb tập từ ko quan trọng
DEFAULT_STOPWORDS = {
    "và", "là", "của", "cho", "với", "trong", "một", "những", "các", "được",
    "đó", "này", "khi", "đã", "tại", "về", "như", "vẫn", "để", "cũng", "bị",
    "ra", "theo", "vào", "hay", "nhưng", "vì", "do", "nên", "còn", "thì"
}


# =====================================================================
# LAYER 1: BASE NER (UNDERTHESEA)

def ner_raw_underthesea(text):
    """
    Tầng 1 — chạy NER gốc từ Underthesea.
    Output: list [(token, ner_tag)]

    
    - PER → B-PER
    - ORG → B-ORG
    - LOC → B-LOC
    - O  → O
    - B-xxx / I-xxx giữ nguyên

    ============================================================
    Chạy NER bằng underthesea.
    Input: text (str)
    Output: List[Tuple[str, str]] ==> list [(token, tag)] như underthesea trả về (chưa nhóm BIO)
    note: underthesea.ner trả về list (word, tag) với tag có thể là 'B-ORG', 'I-ORG', 'O', ...
    """
    if not text or not text.strip():
        return []
    # underthesea.ner yêu cầu đầu vào là string và sẽ tự tách token
    ner_out = uts_ner(text)
    # Đảm bảo trả về list of (token, tag) và loại token rỗng
    clean_output = []

    # (word, pos_tag, chunk_tag, ner_tag)
    for item in ner_out:
        # Underthesea format: (word, pos, chunk, ner)
        if len(item) >= 4:
            word, pos, chunk, ner_tag = item[0], item[1], item[2], item[3]
        else:
            continue
        
        token = str(word).strip()
        tag = str(ner_tag).strip().upper()
        
        if not token:
            continue
        
        # Giữ nguyên tag từ Underthesea
        # O, B-LOC, I-LOC, B-PER, I-PER, B-ORG, I-ORG
        bio_tag = tag if tag else "O"
        
        clean_output.append((token, bio_tag))
    
    # do underthesea nghĩ rằng phim là loc => fix
    MEDIA_PREFIX = {"phim", "truyện", "bài", "bài hát", "ca khúc", "phim ảnh", "bộ phim"}

    fixed = []
    for w, t in clean_output:
        if w.lower() in MEDIA_PREFIX and t.startswith("B-LOC"):
            fixed.append((w, "O"))
        else:
            fixed.append((w, t))
    return fixed

def canonical_title(name: str) -> str:
    """
    Chuẩn hoá.
    Xoá tất cả mọi thứ trong ngoặc (bao gồm cả ngoặc và nội dung bên trong).
    """
    # raw = name

    # Xoá tất cả mọi thứ trong ngoặc (bao gồm cả ngoặc)
    name = re.sub(r"\(.*?\)", "", name)

    # Thu gọn khoảng trắng
    name = " ".join(name.split())
    return name

def build_map(input_list):
    """
    Tạo map canonical -> list các tiêu đề gốc.
    Bảo toàn phim/người trùng tên.
    """
    res_map = {}
    for title in input_list:
        if not title or not title.strip():
            continue

        key = canonical_title(title)
        # if key != title:
        #     print('test build map: ',title, '==>' ,key)
        if key not in res_map:
            res_map[key] = [ title.strip() ]
        else:
            res_map[key].append(title.strip())


    return res_map


def ner_override_graph(raw_tokens, person_list, film_list):
    """
    Tầng 2 — override tag dựa trên graph bipartite.
    Override NER tags dựa trên bipartite graph.
    Sử dụng longest match để xử lý tên nhiều từ.
    Input: list [(token, tag_raw)]
    Output: list [(token, tag_graph_fixed)]
    """
    '''
    raw_tokens: 
        Tầng 1 (raw BIO):
        [('Trấn Thành', 'O'), ('đóng', 'O'), ('trong', 'O'), ('phim', 'O'), ('Bố Già', 'I-LOC')]
    person_list: ['Trấn Thành', 'Ninh Dương Lan Ngọc', 'Ngô Thanh Vân', 'Hồng Đào', 'Kiều Minh Tuấn', 'Victor Vũ', 
    film_list: ['Nhà bà Nữ', 'Hai Phượng', 'Tèo em', 'Mùi ngò gai', 'Cổng mặt trời (phim truyền hình)', 
    film_set: ... 'Những công dân tập thể', 'Mắt biếc', 'Một lần đi bụi', 'Con đường sáng', 'Bước nhảy hoàn vũ', 'Ngôi nhà trong hẻm', 'Bố già', 'Hello cô Ba', 'Những người đã hết thời', 'Đảo của dân ngụ cư'
    person_set: {'Công Hậu', 'Nikki Dương Nhật Vi', 'Trương Minh Cường', 'Nguyệt Nhi', 'Trí Tuệ', 'Nguyên Trinh', 'Thân Thanh Giang', 'Khoa', 'Nguyễn Thị Tuyết', 'Thanh Ngọc', 'Hoàng Trinh diễn viên', 'Thành Trí', 'Tiến Thành', 'Hoàng Mèo', 'Long Điền', 'Linh Chi', 'Minh Luân', 'Thanh Thúy', 'Kathy Tiên', 'Thảo Quyên', 'Quách Ngọc Tuyên', 'Isaac ca sĩ',
    
    output hàm: [('Trấn Thành', 'B-PER'), ('đóng', 'B-PER'), ('trong', 'O'), ('phim', 'O'), ('Bố Già', 'I-LOC')]

    print(normalize("Bố già") ) bố già
    '''
    # Chuẩn hóa lookup: tạo set 
      # ----- PERSON/ FILM MAP + SET -----
    person_map = build_map(person_list)
    film_map = build_map(film_list)
   
    person_set = set(normalize_text_for_nlp(k) for k in person_map.keys())
    film_set   = set(normalize_text_for_nlp(k) for k in film_map.keys())

    # thêm cả bản đầy đủ (bố già (phim 2021)) vào set (trc đó set chỉ có bố già)
    for key, arr in film_map.items():
        for full in arr:
            film_set.add(normalize_text_for_nlp(full)) 
    
    for k, arr in person_map.items():
        for full in arr: 
            person_set.add(normalize_text_for_nlp(full))
    
    out = []
    i = 0
    
    while i < len(raw_tokens):
        matched = False
        best_match_length = 0
        best_match_type = None

        # Thử match từ dài nhất (5 từ) xuống 1 từ
        for length in range(min(5, len(raw_tokens) - i), 0, -1):
            phrase_tokens = [raw_tokens[i + j][0] for j in range(length)]
            phrase = " ".join(phrase_tokens)
            phrase_norm = normalize_text_for_nlp(phrase)
            # Bỏ qua nếu phrase rỗng hoặc là noise
            if not phrase_norm:
                continue
            # Check person match
            if phrase_norm in person_set:
                best_match_length = length
                best_match_type = "PER"
                matched = True
                break  # Tìm được match dài nhất, dừng ngay
            
            # Check film match
            if phrase_norm in film_set:
                best_match_length = length
                best_match_type = "FILM"
                matched = True
                break
        
        # Nếu match được, gán tag mới
        if matched and best_match_length > 0:
            phrase_tokens = [raw_tokens[i + j][0] for j in range(best_match_length)]
            for j in range(best_match_length):
                prefix = "B" if j == 0 else "I"
                out.append((phrase_tokens[j], f"{prefix}-{best_match_type}"))
            i += best_match_length
        else:
            # Không match → giữ nguyên tag gốc
            out.append(raw_tokens[i])
            i += 1
    
    return out


# ====================================
# LAYER 3: WIKI ENRICHMENT
# Keywords để detect entity type từ Wiki
PERSON_KEYWORDS = ["diễn viên", "đạo diễn", "ca sĩ", "nghệ sĩ", "mc", "nhà sản xuất"]
FILM_KEYWORDS = ["phim", "bộ phim", "tác phẩm điện ảnh"]

def detect_entity_type_from_wiki(entity_text, wiki_enrich):
    """
    Phát hiện loại entity từ Wikipedia summary.
    
    Returns: "PER" | "FILM" | None
    """
    entity_norm = normalize_text_for_nlp(entity_text)
    if entity_norm not in wiki_enrich:
        return None
    
    wiki_data = wiki_enrich[entity_norm]
    summary = str(wiki_data.get("summary", "")).lower()
    if not summary:
        return None
    # Check PERSON (ưu tiên cao hơn)
    if any(kw in summary for kw in PERSON_KEYWORDS):
        return "PER"
    
    # Check FILM
    if any(kw in summary for kw in FILM_KEYWORDS):
        return "FILM"
    
    return None
    

def ner_override_wiki(tokens_after_graph, wiki_enrich_norm):
    """
    Tầng 3 — override NER dựa vào wiki enrichment.
    Override NER tags dựa trên Wikipedia enrichment.
    Sử dụng longest match để xử lý tên nhiều từ.

    Input:
        - tokens_after_graph: [(token, tag)]
        - wiki_enrich: dict JSONL đã load thành dict
    Output:
        [(token, final_tag)]

    """
    if not wiki_enrich_norm:
        return tokens_after_graph
    # # Chuẩn hóa wiki dict
    # wiki_norm = {clean_token(k).lower(): v for k, v in wiki_enrich_norm.items()}

    out = []
    i=0
    while i < len(tokens_after_graph):
        matched = False
        best_match_length = 0
        best_match_type = None
        # Thử match từ dài nhất xuống 1 từ
        for length in range(min(5, len(tokens_after_graph) - i), 0, -1):
            phrase_tokens = [tokens_after_graph[i + j][0] for j in range(length)]
            phrase = " ".join(phrase_tokens)
            phrase_norm = normalize_text_for_nlp(phrase).lower()
            
            if not phrase_norm:
                continue

            entity_type = detect_entity_type_from_wiki(phrase, wiki_enrich_norm)
                
            if entity_type:
                best_match_length = length
                best_match_type = entity_type
                matched = True
                break   
        
        # Nếu match được, gán tag mới
        if matched and best_match_length > 0:
            phrase_tokens = [tokens_after_graph[i + j][0] for j in range(best_match_length)]
            for j in range(best_match_length):
                prefix = "B" if j == 0 else "I"
                out.append((phrase_tokens[j], f"{prefix}-{best_match_type}"))
            i += best_match_length
        else:
            # Không match → giữ nguyên
            out.append(tokens_after_graph[i])
            i += 1
    
    return out

# ====================================
# ENTITY EXTRACTION (BIO → ENTITIES)

# TẦNG 1
# extract entities từ BIO tags
# -đọc chuỗi BIO
# -nhóm token
# -convert thành entity chuẩn hóa
# ner_output: List[Tuple[str, str]]
# trả List[Tuple[str, str]]
def extract_entities_from_bio(ner_output):
    """
    Gom các token theo BIO tags thành entities hoàn chỉnh.
    Entity type được chuẩn hóa: PER, ORG, LOC, FILM, O
    ----------------------------------------------
    Nhận vào output của underthesea.ner (list (token, tag))
    -> nhóm các token theo BIO thành entity đầy đủ, trả về list (entity_text, TAG_SIMPLE)
    TAG_SIMPLE là 'LOC' / 'ORG' / 'PER' (chuẩn hóa dạng ngắn) hoặc original tag nếu không nhận dạng BIO.

    biến output thô của tokenizer/NER thành danh sách entity sạch, gọn, chuẩn
    """
    entities = []
    current_tokens = []
    current_tag = None  # 'B-LOC' -> convert to 'LOC'
    for token, tag in ner_output:
        token = str(token).strip()
        tag = str(tag).strip().upper()
        
        if not token:
            continue
        
        # Tag = O → kết thúc entity hiện tại
        if tag == "O":
            if current_tokens:
                entities.append((" ".join(current_tokens), current_tag))
                current_tokens = []
                current_tag = None
            continue

        # Parse BIO tag
        if tag.startswith("B-"):
            # Flush entity cũ nếu có
            if current_tokens:
                entities.append((" ".join(current_tokens), current_tag))

            # Bắt đầu entity mới
            base_tag = tag[2:]  # B-PER → PER
            current_tokens = [token]
            current_tag = base_tag

        elif tag.startswith("I-"):
            base_tag = tag[2:]  # I-PER → PER
            
            # Nếu I-tag khớp với current tag → tiếp tục entity
            if current_tag == base_tag:
                current_tokens.append(token)
            else:
                # I-tag không khớp → bắt đầu entity mới
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_tag))
                current_tokens = [token]
                current_tag = base_tag
        
        else:
            # Tag không phải B-/I-/O → xử lý như B-
            if current_tokens:
                entities.append((" ".join(current_tokens), current_tag))
            current_tokens = [token]
            current_tag = tag
    
    # Flush entity cuối cùng
    if current_tokens:
        entities.append((" ".join(current_tokens), current_tag))
    
    return entities


# ROLE EXTRACTION
# detect ROLE
ROLE_MAP = {
    # ---- ACTOR ----
    "diễn viên": "actor",
    "nghệ sĩ": "actor",
    
    # ---- DIRECTOR ----
    "đạo diễn": "director",
    
    # ---- PRODUCER ----
    "nhà sản xuất": "producer",

    # ---- SCREENWRITER ----
    "biên kịch": "screenwriter",

    # ---- MC / HOST ----
    "mc": "mc",
    "m.c": "mc",
    "người dẫn chương trình": "mc",
    "dẫn chương trình": "mc",
    "host": "mc",

    # ---- COMEDIAN ----
    "hài": "comedian",
    "nghệ sĩ hài": "comedian",

    # ---- FILMMAKER ----
    "nhà làm phim": "filmmaker",
    "movie maker": "filmmaker",
    "film maker": "filmmaker",

    # ---- SINGER ----
    "ca sĩ": "singer",
}

# --- TRÍCH XUẤT TỪ NGỮ CẢNH ---
def extract_roles_from_context(text, entity_name, window_chars=40):
    """
    Tìm role dựa trên từ khóa xuất hiện xung quanh entity trong câu gốc.
    Hỗ trợ tìm cả trước (prefix) và sau (suffix).
    """
    if not text or not entity_name:
        return []
    
    text_norm = normalize_text_for_nlp(text)
    entity_norm = normalize_text_for_nlp(entity_name)
    
    roles = set()
    
    # Tìm vị trí entity trong câu
    start_idx = text_norm.find(entity_norm)
    if start_idx == -1:
        return []
    
    end_idx = start_idx + len(entity_norm)
    
    # Lấy vùng văn bản xung quanh (trước và sau entity)
    # Ví dụ: "... [Diễn viên chính] Trấn Thành..." hoặc "...Galaxy Studio [sản xuất]..."
    window_start = max(0, start_idx - window_chars)
    window_end = min(len(text_norm), end_idx + window_chars)
    
    context_snippet = text_norm[window_start:window_end]
    
    # Quét Role Map trong vùng context này
    for keyword, role in ROLE_MAP.items():
        # Dùng regex \b để tránh bắt nhầm (ví dụ tránh bắt 'nam' trong 'nam nam')
        # Nhưng tiếng Việt từ ghép nên check in string đơn giản thường hiệu quả hơn
        if keyword in context_snippet:
            roles.add(role)
            
    return sorted(list(roles))

def extract_roles_from_graph(person_name, bipartite_graph):
    # Trích xuất vai trò từ bipartite graph.
    if not bipartite_graph:
        return []

    person_norm = normalize_text_for_nlp(person_name)
    roles = set()
    # --- Tìm node theo info["name"], không phải key ---
    node_key = None
    for key in bipartite_graph.nodes:
        if normalize_text_for_nlp(key) == person_norm:
            node_key = key
            break

    if not node_key:
        return []  
    
    # --- Lấy OCCUPATION ---
    # LẤY OCCUPATION TỪ NODES VÀ MAP SANG ENGLISH
    node_data = bipartite_graph.nodes[node_key]
    person_info = node_data.get("info", {})
    occupation = person_info.get("occupation", "")
    if occupation:
        # occupation là chuỗi phân tách bằng dấu phẩy
        for item in occupation.split(","):
            item = item.strip().lower()
            if item:
                # Map sang English nếu khớp ROLE_MAP
                mapped = next((role for kw, role in ROLE_MAP.items() if kw in item), item)
                roles.add(mapped)

    # BỔ SUNG TỪ ROLE TRONG EDGES

    for neighbor, info in bipartite_graph[node_key].items():
        if not isinstance(info, dict):
            continue
        role_value = info.get("role", "").lower()
        if not role_value:
            continue
        # if role_value in ["family", "relative"]:
        #     continue
        # Map sang English chuẩn
        roles.add(role_value)
    return sorted(list(roles))
   

def extract_roles_from_wiki(person_name, wiki_enrich):
    # Trích xuất vai trò từ Wikipedia summary
    if not wiki_enrich:
        return []
    person_norm = normalize_text_for_nlp(person_name)
    roles = set()

    # Tìm key tương ứng
    entry = None
    for key, val in wiki_enrich.items():
        if normalize_text_for_nlp(key) == person_norm:
            entry = val
            break
    if not entry:
        return []
    summary = entry.get("summary", "").lower()
    for keyword, role in ROLE_MAP.items():
        if keyword in summary:
            roles.add(role)

    return sorted(list(roles))

    
# TỔNG HỢP ROLE TỪ WIKI LẪN GRAPH  
def extract_all_roles(text, person_name, bipartite_graph, wiki_enrich):
    # Thêm tham số 'text' vào đầu vào để chạy Context Extraction
   
    # Kết hợp roles từ cả graph và wiki
    roles_graph = extract_roles_from_graph(person_name, bipartite_graph)
    roles_wiki  = extract_roles_from_wiki(person_name, wiki_enrich)
    # Từ Ngữ cảnh (Câu văn hiện tại) 
    roles_context = extract_roles_from_context(text, person_name)
    # Gộp tất cả (Set để loại trùng)
    all_roles = set(roles_graph + roles_wiki + roles_context)
    return sorted(all_roles)

# MAIN NER PIPELINE
# fix TỔ CHỨC
ORG_HINTS = ["studio", "company", "pictures", "production", "entertainment", "corp", "ltd"]

def simple_org_fix(entity):
    name = entity["name"].lower()
    for hint in ORG_HINTS:
        if hint in name:
            entity["type"] = "ORG"
            return entity
    return entity

# COMBINED NER
def run_combine_ner(text, person_list, film_list, wiki_enrich, bipartite_graph):
    # Pipeline NER hoàn chỉnh 3 tầng.
    if not text or not text.strip():
        return []

    # Layer 1: Base NER
    stage1  = ner_raw_underthesea(text)
    
    # Layer 2: Graph override
    stage2 = ner_override_graph(stage1, person_list, film_list)
    
    # Layer 3: Wiki override
    # wiki_enrich_lower = { k.lower(): k for k in wiki_enrich }
    wiki_enrich_norm = { normalize_text_for_nlp(k): v for k, v in wiki_enrich.items() }
    stage3 = ner_override_wiki(stage2, wiki_enrich_norm)

    # Cuối cùng: gom lại theo BIO để ra entity
    entities = extract_entities_from_bio(stage3)
    # Enrich với roles cho PERSON
    results = []

    for entity_name, entity_type in entities:
        if entity_type == "O":
            continue
        result = {
            "name": entity_name,
            "type": entity_type,
            "roles": []
        }
        # Extract roles nếu là PERSON
        if entity_type == "PER":
            result["roles"] = extract_all_roles( 
                text, 
                entity_name, 
                bipartite_graph, 
                wiki_enrich_norm
            )
        
        results.append(result)
    results = [simple_org_fix(ent) for ent in results]
    return results
     

# DEBUG & TESTING

# test 
def debug_ner_pipeline(text):
    print("==== INPUT ====")
    print(text)

    stage1 = ner_raw_underthesea(text)
    print("\nTầng 1 (raw BIO):")
    print(stage1)
    
    stage2 = ner_override_graph(stage1, person_list, film_list)
    print("\nTầng 2 (graph override):")
    print(stage2)

    stage3 = ner_override_wiki(stage2, wiki_enrich)
    print("\nTầng 3 (wiki override):")
    print(stage3)

    final_entities = extract_entities_from_bio(stage3)
    print("\nEntities cuối cùng (sau BIO grouping):")
    print(final_entities)

print('*****************************************************************')
print('TEST 3 TẦNG NER')
debug_ner_pipeline("Trấn Thành đóng trong phim Bố Già cùng các diễn viên khác như ninh dương lan ngọc, kiều minh tuấn")
print('*****************************************************************')


def extract_location_entities(entities):
    return [normalize_entity_name(e["name"]) for e in entities if e["type"] == "LOC"]

def extract_person_entities(entities):
    return [normalize_entity_name(e["name"]) for e in entities if e["type"] == "PER"]

def extract_org_entities(entities):
    return [normalize_entity_name(e["name"]) for e in entities if e["type"] == "ORG"]

def extract_film_entities(entities):
    return [normalize_entity_name(e["name"]) for e in entities if e["type"] == "FILM"]


def tokenize_and_pos_tag(text):
    """
    vào: text: str
    ra: List[Tuple[str, str]]
    Dùng underthesea.pos_tag: trả về list [(word, pos_tag)].
    Tokenize và gán POS-tag bằng underthesea.pos_tag.
    input: text: Câu hoặc đoạn văn bản cần phân tích.
    output: List[(token, pos_tag)]:
        + Danh sách các token và nhãn từ loại tương ứng
    
    underthesea.pos_tag tự tokenize nội bộ
    trả về kết quả raw ở dạng (token, tag) đã được strip()   
    """
    if not text or not text.strip():
        return []
    
    # normalize Unicode để tránh lỗi dấu
    text_norm = unicodedata.normalize("NFC", text.strip())

    try:
        pos_out = uts_pos_tag(text_norm)
    except Exception:
        return [] 
    
    results = []
    for token, tag in pos_out:
        tok = str(token).strip()
        tg = str(tag).strip().upper()

        # Bỏ các token vô nghĩa
        if not tok:
            continue
        if len(tok) == 1 and tok in ",.!?;:-_\"'()[]{}*/":
            continue

        # Normalize token 
        tok_norm = unicodedata.normalize("NFC", tok)

        results.append((tok_norm, tg))
    return results


def extract_keywords_from_pos(
    pos_output,
    films, persons, orgs, locations,
    min_len = 2,
    stopwords = None
):
    """
    Tìm từ khóa dựa vào pos tags:
    - Chọn từ có pos bắt đầu bằng 'N' (danh từ) hoặc 'A' (tính từ) thường đại diện cho topics.
    - Lọc stopwords, ký tự không phải chữ, và từ ngắn (< min_len).
    - Trả về danh sách từ (không trùng), giữ thứ tự xuất hiện theo tần suất giảm dần; nếu bằng nhau → theo thứ tự xuất hiện
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    stopwords = set(w.lower() for w in stopwords)

    # OrderedDict để ghi nhớ thứ tự xuất hiện đầu tiên
    order = OrderedDict()
    counter = Counter()
   
    for word, pos in pos_output:
        # chuẩn hóa unicode + loại khoảng trắng
        token = unicodedata.normalize("NFKC", word).strip()
        if not token:
            continue
        pos_u = pos.upper()

        if not (pos_u.startswith("N") or pos_u.startswith("A")):
            continue

        # keyword extraction không cần giữ nguyên tên riêng, không cần bảo tồn viết hoa, cũng không cần entity format
        token_clean = normalize_text_for_nlp(token)
        if not token_clean: continue 

        # loại stopwords
        if token_clean in stopwords: continue

        # loại từ 1 ký tự
        if len(token_clean) < min_len:continue

        # loại token rác: chỉ toàn số, toàn punctuation, toàn ký tự không phải chữ
        # nhưng ko loại từ có dấu gạch (-), dấu nháy (')
        if re.fullmatch(r"^[\W\d_]+$", token_clean): continue

        # save thứ tự xuất hiện (nếu chưa có)
        if token_clean not in order: order[token_clean] = None
        counter[token_clean] += 1

    # Sort theo tần suất giảm → nếu bằng nhau, theo thứ tự xuất hiện
    keywords = sorted(counter.keys(), key=lambda k: (-counter[k], list(order.keys()).index(k)))

    # Loại token thuộc thực thể NER (film, person, org, loc)
    all_ner = set([*films, *persons, *orgs, *locations])
    all_ner_norm = {unicodedata.normalize("NFKC", x).lower().strip() for x in all_ner}

    # Tách tất cả tokens trong entity đa từ (vd: "bố già" → {"bố", "già"})
    ner_subtokens = set()
    for ent in all_ner_norm:
        for t in ent.split():
            ner_subtokens.add(t.strip())

    # Loại token nếu nó là 1 phần của NER nhiều từ
    clean_keywords = []
    for kw in keywords:
        kw_norm = normalize_text_for_nlp(kw)
        if kw_norm in ner_subtokens:
            continue  # loại "bố", "già", "thành", "lan", "ngọc"...
        clean_keywords.append(kw)

    keywords = clean_keywords


    return keywords 
  
        

def extract_top_keywords(text, k= 10,stopwords = DEFAULT_STOPWORDS ):
    """
    Đếm tần suất các token (token hóa bằng underthesea.word_tokenize), lọc stopwords và punctuation.
    Trả về top k keywords cùng tần suất: [(keyword, count), ...]
    """
    if not text or not text.strip():
        return []
    
    # chuẩn hóa unicode trước khi tách từ
    text = unicodedata.normalize("NFKC", text)

    stopwords = {unicodedata.normalize("NFKC", w).lower().strip() for w in stopwords}

    # token hóa (underthesea.word_tokenize giữ token tiếng việt tốt)
    tokens = uts_word_tokenize(text)
    cleaned = []
    for tok in tokens:
        # chuẩn hóa token
        w = unicodedata.normalize("NFKC", tok)
        w = normalize_text_for_nlp(w)

        if not w:
            continue
        if w in stopwords:
            continue
        if len(w) < 2:
            continue
        # loại token rác (toàn số/ký hiệu), nhưng cho phép từ có '-' hoặc ' hoặc /
        # giữ trấn-thành
        if re.fullmatch(r"^[\W\d_]+$", w):
            continue

        cleaned.append(w)

        tok2 = normalize_text_for_nlp(tok)
        if not tok2:
            continue
        if tok2 in stopwords:
            continue
        # loại bỏ token chỉ số/punctuation
        if re.fullmatch(r"[\d\W_]+", tok2):
            continue
        if len(tok2) < 2:
            continue
        cleaned.append(tok2)

    freq = Counter(cleaned)
    return freq.most_common(k)


def generate_topic_nodes(keywords,top_n = None):
    """
    Từ iterable keywords (string), tạo nodes dạng (name, 'Topic').
    Nếu top_n được truyền vào thì lấy top_n đầu (tiền đề: keywords đã sắp theo tần suất).
    Trả về danh sách tuple (topic_name, 'Topic').

    TOPIC: mặc định để bỏ những keyword không thuộc loại NER nào khác.
    label dự phòng trong hệ thống phân loại node.
    """
    # đảm bảo iterable → list (để cắt top_n)
    kws = list(keywords)
    if top_n is not None:
        kws = kws[:top_n]
  
    nodes = []
    seen = set()
    for item in kws:
        # hỗ trợ dạng (keyword, count)
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            k = item[0]
        else:
            k = item

        if not isinstance(k, str): k = str(k) 

        # chuẩn hóa unicode + strip
        name = unicodedata.normalize("NFKC", k).strip()
        if not name:continue
            
        # loại các token rác: toàn ký hiệu hoặc toàn số
        if re.fullmatch(r"[\W\d_]+", name): continue
        # collapse spaces
        name_clean  = re.sub(r"\s+", " ", name).lower()
        # chuẩn hóa key kiểm tra trùng, ko dùng để hiển thị
        keynorm = name_clean.lower()
        if keynorm in seen:
            continue # bước loại trùng, nếu trùng thì k làm bc sau
        seen.add(keynorm)
        
        nodes.append((name_clean, "Topic"))
    return nodes

#======================================================
def add_node(name_raw, typ_raw,nodes,seen):
    name = normalize_entity_name(name_raw)
    if not name: 
        return nodes, seen

    typ = normalize_type(typ_raw, default="Node")

    key = name.lower()
    if key in seen: 
        return nodes, seen
    seen.add(key)
    nodes.append((name, typ))
    return nodes,seen


def create_new_nodes(films, persons,orgs,locations,topics):
    nodes= []
    seen = set()
    # --- Films ---
    for f in (films or []):
        nodes, seen = add_node(f, "Film", nodes, seen)

    # --- Locations ---
    for loc in (locations or []):
        nodes, seen =add_node(loc, "Location", nodes,seen)

    
    # --- Topics ---
    for item in (topics or []):
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            name = item[0]
            type_or_freq = item[1] if len(item) > 1 else "Topic"

            # nếu t[1] là số ⇒ hiểu là freq ⇒ set type = Topic
            if isinstance(type_or_freq, (int, float)):
                nodes, seen =add_node(name, "Topic",nodes,seen)
            else:
                nodes,seen=add_node(name, type_or_freq,nodes,seen)
        else:
            nodes,seen=add_node(item, "Topic",nodes,seen)

    # --- Persons ---
    for p in (persons or []):
        nodes,seen=add_node(p, "Person",nodes,seen)

    # --- Organizations ---
    for o in (orgs or []):
        nodes,seen=add_node(o, "Organization",nodes,seen)

    return nodes

def pipeline_extract_nodes_from_summary(summary_text,person_list, film_list, wiki_enrich, B,top_k_keywords=5,stopwords=None):
    """
    Pipeline đầy đủ trích xuất các node từ một đoạn summary.

    Các bước xử lý:
    1) Chạy NER để lấy:
       - Location
       - Person
       - Organization
    2) POS tagging → chọn keyword dạng danh từ/tính từ.
    3) Lấy top_k_keywords làm Topic nodes.
    4) Gộp toàn bộ thành danh sách node chuẩn hóa dạng:
       [(name, type), ...]

    Tham số:
    - summary_text: đoạn văn cần phân tích.
    - top_k_keywords: số lượng Topic mong muốn.
    - stopwords: bộ stopwords tùy chỉnh (nếu None → dùng mặc định).

    Trả về:
    - Danh sách node không trùng (name, type).
    """

    # --- Trường hợp input rỗng ---
    if not summary_text or not summary_text.strip():
        return []
    # --- Bước 1: NER ---
    ner_out = run_combine_ner(summary_text,person_list, film_list, wiki_enrich, B)
    films    = extract_film_entities(ner_out)
    persons  = extract_person_entities(ner_out)
    orgs     = extract_org_entities(ner_out)
    locations = extract_location_entities(ner_out)
    


    # --- Bước 2: POS tagging & keyword extraction ---
    pos_out = tokenize_and_pos_tag(summary_text)
    keywords_by_pos = extract_keywords_from_pos(pos_out,
                                                films, persons, orgs, locations,
                                                stopwords=stopwords)
    # Giới hạn số lượng chủ đề
    # Lấy top k theo pos (nếu enumerate)
    top_keywords = keywords_by_pos[:top_k_keywords]
    # --- Bước 3: Tạo Topic nodes ---
    topic_nodes = generate_topic_nodes(top_keywords)
    # --- Bước 4: Gộp tất cả node --
    new_nodes = create_new_nodes(films=films,
                                 persons=persons, 
                                 orgs=orgs,
                                 locations =locations, 
                                 topics=topic_nodes
                                 )
    return new_nodes



if __name__ == "__main__":
    
    sample = "Bố Già là một phim điện ảnh chủ đề gia đình, hài kịch, bối cảnh tại TP.HCM. Diễn viên chính: Trấn Thành, Ninh Dương Lan Ngọc, kiều minh tuấn. Bộ phim do Galaxy Studio sản xuất."

    from underthesea import ner



    ner_raw = ner(sample)


    combine_ner = run_combine_ner(sample,person_list, film_list, wiki_enrich, B)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("NER raw:", combine_ner)
#     print("Locations:", extract_location_entities(combine_ner))
#     print("Films:", extract_film_entities(combine_ner))
#     print("Persons:", extract_person_entities(combine_ner))
#     print("Orgs:", extract_org_entities(combine_ner))
#     print("POS:", tokenize_and_pos_tag(sample))
#     print("Keywords by POS:", extract_keywords_from_pos(tokenize_and_pos_tag(sample), 
#           extract_film_entities(combine_ner), 
#           extract_person_entities(combine_ner),
#           extract_org_entities(combine_ner), 
#           extract_location_entities(combine_ner) 
#           ))

#     print("Top tokens:", extract_top_keywords(sample, k=10))
#     print("Generated topic nodes:", generate_topic_nodes(["gia đình", "hài kịch", "phim"]))
#     print("Final nodes:", pipeline_extract_nodes_from_summary(sample,person_list, film_list, wiki_enrich, B))

