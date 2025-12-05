from collections import Counter, defaultdict,OrderedDict
import re
from underthesea import ner as uts_ner
from underthesea import pos_tag as uts_pos_tag
from underthesea import word_tokenize as uts_word_tokenize
import unicodedata

# loại bỏ dấu câu trước khi xử lý văn bản
RE_PUNCT = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\\'\"!#\$%&\(\)\*\+,\-./:;<=>\?@\[\]^_`{\|}~…“”‘’—·]+")

# chuẩn hóa khoảng trắng
RE_SPACE = re.compile(r"\s+")


# lọc khỏi vb tập từ ko quan trọng
DEFAULT_STOPWORDS = {
    "và", "là", "của", "cho", "với", "trong", "một", "những", "các", "được",
    "đó", "này", "khi", "đã", "tại", "về", "như", "vẫn", "để", "cũng", "bị",
    "ra", "theo", "vào", "hay", "nhưng", "vì", "do", "nên", "còn", "thì"
}

# vào str -> trả str
def clean_token(tok):
    tok = tok.strip()
    tok = RE_PUNCT.sub(" ", tok)
    tok = RE_SPACE.sub(" ", tok)
    return tok.strip()


# extract entities từ BIO tags
# -đọc chuỗi BIO
# -nhóm token
# -convert thành entity chuẩn hóa
# ner_output: List[Tuple[str, str]]
# trả List[Tuple[str, str]]
def extract_entities_from_bio(ner_output):
    """
    Nhận vào output của underthesea.ner (list (token, tag))
    -> nhóm các token theo BIO thành entity đầy đủ, trả về list (entity_text, TAG_SIMPLE)
    TAG_SIMPLE là 'LOC' / 'ORG' / 'PER' (chuẩn hóa dạng ngắn) hoặc original tag nếu không nhận dạng BIO.

    biến output thô của tokenizer/NER thành danh sách entity sạch, gọn, chuẩn
    """
    entities = []
    cur_tokens = []
    cur_tag = None  # 'B-LOC' -> convert to 'LOC'
    for token, tag in ner_output:
        token = token.strip()
        if not token:
            continue

        # underthesea trả về 'O' hoặc 'B-LOC' / 'I-LOC' hoặc 'LOC' — handle cả 2
        # Kết thúc entity hiện tại khi gặp 'O'
        if tag == 'O' or tag.upper() == 'O':
            # flush current
            if cur_tokens:
                entities.append((" ".join(cur_tokens), cur_tag))
                cur_tokens = []
                cur_tag = None
            continue

        # Chuẩn hóa định dạng tag BIO → lấy tag chính
        tag_u = tag.upper()
        if tag_u.startswith('B-') or tag_u.startswith('I-'):
            base_tag = tag_u.split('-', 1)[1] # B-LOC → LOC 
        else:
            base_tag = tag_u  # LOC / PER / ORG

        # Bắt đầu entity mới
        if tag_u.startswith('B-') or (not cur_tokens):
            if cur_tokens:
                entities.append((" ".join(cur_tokens), cur_tag))
            cur_tokens = [token]
            cur_tag = base_tag

        # Tiếp tục entity (I-) hoặc tag giống nhau
        else:  
            # Nếu tag thay đổi → đóng entity cũ và mở mới
            if cur_tag is None or base_tag != cur_tag:
                if cur_tokens:
                    entities.append((" ".join(cur_tokens), cur_tag))
                cur_tokens = [token]
                cur_tag = base_tag
            else:
                cur_tokens.append(token)
    # Flush entity cuối cùng
    if cur_tokens:
        entities.append((" ".join(cur_tokens), cur_tag))

    # Chuẩn hóa tên tag (LOC, ORG, PERSON)
    normalized = []
    for ent_text, tag in entities:
        tag_norm = tag
        if tag is None:
            tag_norm = 'O'
        else:
            t = tag.upper()
            if 'LOC' in t:
                tag_norm = 'LOC'
            elif 'PER' in t or 'PERSON' in t or 'NAME' in t:
                tag_norm = 'PERSON'
            elif 'ORG' in t:
                tag_norm = 'ORG'
            else:
                tag_norm = t
        normalized.append((clean_token(ent_text), tag_norm))
    return normalized



def run_ner(text):
    """
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
    for w, pos, chunk, tag in ner_out:
        token = str(w).strip()
        ner_tag  = str(tag).strip()
        if token:            # Bỏ token rỗng
            clean_output.append((token, ner_tag ))

    return clean_output

# ner_output: List[Tuple[str, str]]
# target_tag: str
def extract_entities_by_tag(ner_output,target_tag):
    """
    Từ output của run_ner (list (token, tag)), trả về danh sách (đã gộp entity)
    Loại tag trả về: target_tag
    """
    grouped = extract_entities_from_bio(ner_output)
    results  = []
    seen = set() # để xử lý trùng
    for ent_text, tag in grouped:
        if tag != target_tag: continue
        ent_clean = ent_text.strip()
        if not ent_clean: continue 

        # loại rác: dấu đơn, dấu câu, 1 ký tự
        if len(ent_clean) < 2: continue

        # chuẩn hóa key để kiểm tra trùng lặp
        norm_key = ent_clean.lower()
        if norm_key not in seen:
            seen.add(norm_key)
            results .append(ent_clean)
    
    return results 

def extract_location_entities(ner_output):
    return extract_entities_by_tag(ner_output, "LOC")
def extract_org_entities(ner_output):
    return extract_entities_by_tag(ner_output, "ORG")
def extract_person_entities(ner_output):
    return extract_entities_by_tag(ner_output, "PERSON")


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

        # Normalize token (bạn có thể tùy chỉnh)
        tok_norm = unicodedata.normalize("NFC", tok)

        results.append((tok_norm, tg))
    return results


def extract_keywords_from_pos(
    pos_output,
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

        token_clean = clean_token(token).lower()
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
        w = clean_token(w).lower()

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

        tok2 = clean_token(tok).lower()
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
        if isinstance(item, (tuple, list)) and len(item) >= 1:k = item
        else: k = item
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

def normalize_name(x) :
    """Chuẩn hóa tên node: unicode, strip, collapse spaces."""
    if not isinstance(x, str): x = str(x)

    x = unicodedata.normalize("NFKC", x).strip()
    if not x: return ""

    # loại token toàn số hoặc toàn ký tự đặc biệt
    if re.fullmatch(r"[\W\d_]+", x):return ""

    x = re.sub(r"\s+", " ", x)  
    return x

def normalize_type(t, default):
    """Chuẩn hóa type: viết hoa chữ đầu."""
    if not t: return default
    t = str(t).strip()
    if not t: return default
    return t[0].upper() + t[1:]

def add_node(name_raw, typ_raw,nodes,seen):
    name = normalize_name(name_raw)
    if not name: 
        return nodes, seen

    typ = normalize_type(typ_raw, default="Node")

    key = name.lower()
    if key in seen: 
        return nodes, seen
    seen.add(key)
    nodes.append((name, typ))
    return nodes,seen


def create_new_nodes(locations,topics,persons,orgs):
    nodes= []
    seen = set()

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

def pipeline_extract_nodes_from_summary(summary_text,top_k_keywords=5,stopwords=None):
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
    ner_out = run_ner(summary_text)
    locations  = extract_location_entities(ner_out)
    persons = extract_person_entities(ner_out)
    orgs = extract_org_entities(ner_out)

    # --- Bước 2: POS tagging & keyword extraction ---
    pos_out = tokenize_and_pos_tag(summary_text)
    keywords_by_pos = extract_keywords_from_pos(pos_out, stopwords=stopwords)
    # Giới hạn số lượng chủ đề
    # Lấy top k theo pos (nếu enumerate)
    top_keywords = keywords_by_pos[:top_k_keywords]
    # --- Bước 3: Tạo Topic nodes ---
    topic_nodes = generate_topic_nodes(top_keywords)
    # --- Bước 4: Gộp tất cả node --
    new_nodes = create_new_nodes(locations =locations, 
                                 topics=topic_nodes, 
                                 persons=persons, 
                                 orgs=orgs)
    return new_nodes



if __name__ == "__main__":
    sample = "Bố Già là một phim điện ảnh chủ đề gia đình, hài kịch, bối cảnh tại TP.HCM. Diễn viên chính: Trấn Thành, Ninh Dương Lan Ngọc. Bộ phim do Galaxy Studio sản xuất."

    from underthesea import ner



    ner_raw = ner(sample)

    print("NER OUTPUT RAW:")

    for idx, item in enumerate(ner_raw, 1):
        print(f"{idx}. {item}  (len={len(item)})")

    print("NER raw:", run_ner(sample))
    print("Locations:", extract_location_entities(run_ner(sample)))
    print("Persons:", extract_person_entities(run_ner(sample)))
    print("Orgs:", extract_org_entities(run_ner(sample)))
    print("POS:", tokenize_and_pos_tag(sample))
    print("Keywords by POS:", extract_keywords_from_pos(tokenize_and_pos_tag(sample)))
    print("Top tokens:", extract_top_keywords(sample, k=10))
    print("Generated topic nodes:", generate_topic_nodes(["gia đình", "hài kịch", "phim"]))
    print("Final nodes:", pipeline_extract_nodes_from_summary(sample))