from src.nlp.ner import person_list, film_list, wiki_enrich, run_combine_ner
from src.data_prep.load_graph import B, G_collab 
from src.nlp.text_utils import normalize_entity, normalize_entity_name,remove_footnotes,split_text_into_sentences
import re

import json
def load_jsonl_to_dict(path, key_field="entity"):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            key = obj.get(key_field)
            if key:
                result[key] = obj

    return result


import networkx as nx

def inspect_structure(data, label="INPUT", indent=0):
    prefix = " " * indent
    print(f"\n{prefix}=== Inspecting: {label} ===")

    # ==============================
    # CASE 1 — NetworkX Graph
    # ==============================
    if isinstance(data, nx.Graph):
        print(f"{prefix}Detected: NetworkX Graph")
        print(f"{prefix}- num_nodes = {data.number_of_nodes()}")
        print(f"{prefix}- num_edges = {data.number_of_edges()}")

        # ---- Inspect node attributes ----
        if data.number_of_nodes() > 0:
            any_node = next(iter(data.nodes))
            attrs = data.nodes[any_node]
            print(f"{prefix}- Node example: '{any_node}'")
            print(f"{prefix}  Node attribute keys: {list(attrs.keys())}")
        else:
            print(f"{prefix}- No node found to inspect attributes")

        # ---- Inspect edge attributes ----
        if data.number_of_edges() > 0:
            any_u, any_v = next(iter(data.edges))
            attrs = data.get_edge_data(any_u, any_v)
            print(f"{prefix}- Edge example: {any_u} -> {any_v}")
            print(f"{prefix}  Edge attribute keys: {list(attrs.keys())}")
        else:
            print(f"{prefix}- No edge found to inspect attributes")

        return

    # ==============================
    # CASE 2 — dict
    # ==============================
    if isinstance(data, dict):
        print(f"{prefix}Detected: dict with {len(data)} keys")
        for key, value in data.items():
            print(f"{prefix}- {key}: type={type(value).__name__}")

            # sub dict → print sub-keys
            if isinstance(value, dict):
                print(f"{prefix}   child keys: {list(value.keys())}")

            # list of dict → check first element
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                print(f"{prefix}   list item keys: {list(value[0].keys())}")

        return

    # ==============================
    # CASE 3 — list of dict
    # ==============================
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        print(f"{prefix}Detected: list[{len(data)}] of dict")
        print(f"{prefix}- item keys: {list(data[0].keys())}")
        return

    # ==============================
    # OTHER TYPE
    # ==============================
    print(f"{prefix}Type={type(data).__name__} — nothing to inspect.")



import networkx as nx
from rapidfuzz import fuzz

def find_existing_node(G, raw_name):
    """EL: exact → lowercase → fuzzy"""
    if not raw_name:
        return None

    name = normalize_entity_name(raw_name)
    name_low = name.lower()

    # 1) exact
    if name in G:
        return name

    # 2) lowercase
    for n in G.nodes:
        if n.lower() == name_low:
            return n

    # 3) fuzzy > 90
    best = None
    best_score = 0
    for n in G.nodes:
        score = fuzz.ratio(name_low, n.lower())
        if score > 90 and score > best_score:
            best = n
            best_score = score

    return best

def add_custom_node(G, raw_name, typ="person"):
    """Thêm node mới theo đúng format chuẩn."""
    name = normalize_entity_name(raw_name)
    if not name:
        return None

    # tìm node sẵn có trước
    exist = find_existing_node(G, name)
    if exist:
        return exist

    # tạo node mới đúng cấu trúc GRAPH_B
    G.add_node(name, type=typ, info={})
    return name


def add_person_node(G, raw_name):
    return add_custom_node(G, raw_name, "person")

def add_film_node(G, raw_name):
    return add_custom_node(G, raw_name, "film")

def add_location_node(G, raw_name):
    return add_custom_node(G, raw_name, "location")

def add_topic_node(G, raw_name):
    return add_custom_node(G, raw_name, "topic")

def add_work_edge(G, src, role, character, dst):
    """
    Edge diễn viên → tác phẩm (film).
    role / character đúng theo format GRAPH_B.
    """
    G.add_edge(
        src,
        dst,
        relation_category="PER_WORK",
        role=role,
        character=character
    )
 

def add_relation_edge(G, src_raw, rel_type, dst_raw):
    """
    Person–Person edges (from RE result).
    """
    # tìm node hiện có trước
    src = find_existing_node(G, src_raw)
    if not src:
        src = add_person_node(G, src_raw)

    dst = find_existing_node(G, dst_raw)
    if not dst:
        dst = add_person_node(G, dst_raw)

    # lấy data cũ nếu đã có edge
    data = G.get_edge_data(src, dst, default={})

    data["relation_category"] = "PER_PER"

    # accumulate relation types
    types = set(data.get("relation_types", []))
    types.add(rel_type)
    data["relation_types"] = list(types)

    # tăng weight
    data["weight"] = data.get("weight", 0) + 1

    G.add_edge(src, dst, **data)
    return src, dst


def update_graph_with_new_nodes(G, new_nodes):
    """
    new_nodes = [(raw_name, type), ...]
    """
    for name, typ in new_nodes:
        add_custom_node(G, name, typ)


def update_graph_with_relations(G, relations):
    """
    relations = [{"subject": "...", "relation": "...", "object": "..."}]
    """
    for r in relations:
        s = r.get("subject")
        o = r.get("object")
        t = r.get("relation")
        if not (s and o and t):
            continue
        add_relation_edge(G, s, t, o)


def ensure_unique_node(G, candidate_set):
    votes = {}
    for raw in candidate_set:
        name = normalize_entity_name(raw)
        linked = find_existing_node(G, name)
        if linked:
            votes[linked] = votes.get(linked, 0) + 1

    if not votes:
        return None

    return max(votes, key=votes.get)


def save_graph_json(G, path):
    data = nx.node_link_data(G,edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

def update_graph():  
    re_res_dict = load_jsonl_to_dict("data/re_results.jsonl", "entity")
    inspect_structure(G_collab, "G_COLLAB")
    inspect_structure(B, "GRAPH_B")
    print('======================================')
    node = "Trấn Thành"
    print(node, "=>", B.nodes[node])
    print(type(re_res_dict))        # <class 'dict'>
    print(re_res_dict.keys())
    # print(re_res_dict)
    print(re_res_dict['Trấn Thành'].keys())
    B_updated = B.copy()
    G_collab_updated = G_collab.copy()
    print(find_existing_node(G_collab, "Trấn Thành"))
    save_graph_json(B_updated, "data/updated/B.json")
    save_graph_json(G_collab_updated, "data/updated/G_collab.json")

# =============================================
import re
from typing import List, Dict, Set

GENRE_KEYWORDS: Dict[str, List[str]] = {
    "hai": [
        "hài",
        "hài hước",
        "hề",
        "comedy",
        "hài kịch"
    ],
    "chinh_kich": [
        "chính kịch",
        "kịch",
        "drama"
    ],
    "tinh_cam": [
        "tình cảm",
        "romantic",
        "romance"
    ],
    "lang_man": [
        "lãng mạn",
        "lãng mạng"  # typo phổ biến
    ],
    "co_trang": [
        "cổ trang",
        "thời trang cổ",
        "cung đình"
    ],
    "lich_su": [
        "lịch sử",
        "history",
        "historical"
    ],
    "tam_ly": [
        "tâm lý",
        "tâm linh",
        "psychology"
    ],
    "xa_hoi": [
        "xã hội",
        "social"
    ],
    "chien_tranh": [
        "chiến tranh",
        "war",
        "quân sự"
    ],
    "kinh_di": [
        "kinh dị",
        "horror",
        "ma",
        "thriller"
    ],
    "hanh_dong": [
        "hành động",
        "action",
        
    ],
    "vo_thuat": ["võ thuật",
        "kung fu"],
    
    "tai_lieu": [
        "tài liệu",
        "documentary"
    ],
    "hoat_hinh": [
        "hoạt hình",
        "animation",
        "cartoon"
    ],
    "vien_tuong": [
        "viễn tưởng",
        "khoa học viễn tưởng",
        "sci-fi",
        "science fiction"
    ],
    "than_thoai": [
        "thần thoại",
        "huyền thoại",
        "mythology"
    ],
    "am_nhac": [
        "âm nhạc",
        "musical",
        "music"
    ],
    "gia_dinh": [
        "gia đình",
        "family"
    ],
    "hinh_su": [
        "hình sự",
        "tội phạm",
        "crime"
    ],

    "dien_anh":['điện ảnh', 'phim điện ảnh'],
    "truyen_hinh":['truyền hình', 'phim truyền hình']
}


GENRE_LABEL_MAP = {
    "hai": "hài",
    "chinh_kich": "chính kịch",
    "tinh_cam": "tình cảm",
    "lang_man": "lãng mạn",
    "co_trang": "cổ trang",
    "lich_su": "lịch sử",
    "tam_ly": "tâm lý",
    "xa_hoi": "xã hội",
    "chien_tranh": "chiến tranh",
    "kinh_di": "kinh dị",
    "hanh_dong": "hành động",
    "vo_thuat": "võ thuật",
    "tai_lieu": "tài liệu",
    "hoat_hinh": "hoạt hình",
    "vien_tuong": "viễn tưởng",
    "than_thoai": "thần thoại",
    "am_nhac": "âm nhạc",
    "gia_dinh": "gia đình",
    "hinh_su": "hình sự",
    "dien_anh": "điện ảnh",
    "truyen_hinh": "truyền hình",
}

def normalize_text(text: str) -> str:
    """Chuẩn hóa text"""
    text = text.lower()
    # Chuẩn hóa gạch nối
    text = re.sub(r"[–—−]", "-", text)
    # Loại bỏ dấu ngoặc, dấu chấm phẩy
    text = re.sub(r"[()\",.:;]", " ", text)
    # Gộp khoảng trắng
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_genre_candidates(text: str) -> List[str]:
    """
    Tìm tất cả cụm có thể là thể loại trong text
    """
    text_norm = normalize_text(text)
    candidates = []
    
    # Pattern 1: "thuộc thể loại X"
    pattern1 = r"thuộc thể loại\s+([^.\n]+?)(?=\s+ra mắt|\s+của|\s+năm|\.|\n|$)"
    matches = re.findall(pattern1, text_norm)
    candidates.extend(matches)
    
    # Pattern 2: "phim (điện ảnh) X"
    pattern2 = r"(?:phim|bộ phim)\s+(?:điện ảnh\s+)?([^,.\n]+?)(?=\s+của|\s+năm|\s+do|\.|\n|$)"
    matches = re.findall(pattern2, text_norm)
    candidates.extend(matches)
    
    # Pattern 3: "là phim X" hoặc "là một bộ phim X"
    pattern3 = r"là\s+(?:một\s+bộ\s+)?phim\s+([^,.\n]+?)(?=\s+của|\s+năm|\s+do|\.|\n|$)"
    matches = re.findall(pattern3, text_norm)
    candidates.extend(matches)
    
    return candidates


def split_into_tokens(text: str) -> List[str]:
    """
    Tách text thành các token (từ/cụm)
    'chính kịch lãng mạn' -> ['chính kịch', 'lãng mạn']
    'cổ trang lịch sử' -> ['cổ trang', 'lịch sử']
    """
    # Tách theo dấu gạch ngang, phẩy, gạch chéo
    parts = re.split(r"\s*[-,/]\s*", text)
    
    tokens = []
    for part in parts:
        part = part.strip()
        if part:
            # Tách theo từ đơn và cụm 2 từ
            words = part.split()
            
            # Thêm các cụm 2 từ liên tiếp
            for i in range(len(words)):
                # Từ đơn
                tokens.append(words[i])
                # Cụm 2 từ
                if i < len(words) - 1:
                    tokens.append(f"{words[i]} {words[i+1]}")
            
    return list(set(tokens))  # Loại trùng


def match_genre(token: str) -> Set[str]:
    """
    Match một token với các genre keywords
    """
    token_norm = normalize_text(token)
    matched_genres = set()
    
    for genre_id, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            keyword_norm = normalize_text(keyword)
            if token_norm == keyword_norm:
                matched_genres.add(genre_id)
                break
    
    return matched_genres


def classify_movie_genre(text: str) -> List[str]:
    """
    Phân loại thể loại phim - tách hết tất cả
    """
    # Bước 1: Tìm các cụm có thể là thể loại
    candidates = extract_genre_candidates(text)
    
    if not candidates:
        return []
    
    # Bước 2: Tách từng candidate thành tokens
    all_tokens = []
    for candidate in candidates:
        tokens = split_into_tokens(candidate)
        all_tokens.extend(tokens)
    
    # Bước 3: Match từng token với genre keywords
    found_genres = set()
    for token in all_tokens:
        matched = match_genre(token)
        found_genres.update(matched)
    
    return sorted(found_genres)


def format_genre_list(
    genre_keys: list[str],
    label_map: dict[str, str] = GENRE_LABEL_MAP
) -> str:
    """
    Chuyển key -> viết dạng có dấu
    """
    labels = []

    for key in genre_keys:
        if key not in label_map:
            raise ValueError(f"Genre key không hợp lệ: {key}")
        labels.append(label_map[key])

    return ", ".join(labels)

from networkx.readwrite import json_graph
def update_film_genre_and_save(B, film, res, output_path="data/updated/B.json"):
    """
    B     : networkx graph
    film  : id của node film
    res   : genre (str | list[str])
    """

    # 1. Kiểm tra node tồn tại
    if film not in B.nodes:
        raise ValueError(f"Film node '{film}' không tồn tại trong graph")

    attrs = B.nodes[film]

    # 2. Chỉ cho phép film
    if attrs.get("type") != "film":
        raise ValueError(f"Node '{film}' không phải type=film")

    # 3. Đảm bảo info tồn tại
    if "infobox" not in attrs or attrs["infobox"] is None:
        attrs["infobox"] = {}

    # 4. Gán genre
    attrs["infobox"]["genre"] = res

    # 5. Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 6. Lưu graph ra JSON
    data = json_graph.node_link_data(B,  edges="edges")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return B
def run_sample():
    samples = {
        "Nha ba Nu": """
        Nhà bà Nữ (tiếng Anh: The House of No Man) là một bộ phim điện ảnh Việt Nam thuộc thể loại hài – chính kịch ra mắt vào năm 2023 do Trấn Thành làm đạo diễn và đồng sản xuất, với sự tham gia diễn xuất của các diễn viên gồm Lê Giang, Uyển Ân, Song Luân, Trấn Thành, Khả Như, Quỳnh Lý, Phương Lan, Dương Lâm, Ngọc Giàu và Việt Anh. Lấy cảm hứng từ câu chuyện về người phụ nữ bán bánh canh cua giá 300.000 VND gây tranh cãi trên mạng xã hội và câu chuyện gia đình của một người bạn trong giới giải trí của Trấn Thành, tác phẩm xoay quanh những mâu thuẫn trong gia đình của bà Ngọc Nữ, một chủ tiệm bánh canh cua ở khu chung cư cũ.
        """,

        "Mat Biec": """
        Mắt biếc là phim điện ảnh chính kịch lãng mạn
        của Việt Nam năm 2019 do Victor Vũ đạo diễn.
        """,

        "Vo ba": """
        Vợ ba là một bộ phim cổ trang lịch sử tâm lý xã hội
        năm 2018 của đạo diễn Ash Mayfair.
        """,

        "Mua do": """
        Mưa đỏ (tên đầy đủ: Mưa đỏ: Máu xương đổ xuống – Đất trời lưu danh; tiếng Anh: Red Rain) 
        là một bộ phim điện ảnh Việt Nam thuộc thể loại lịch sử – chiến tranh – chính kịch ra mắt năm 2025 do Đặng Thái Huyền làm đạo diễn, được chuyển thể từ tiểu thuyết cùng tên của nhà văn Chu Lai và đồng thời lấy cảm hứng từ sự kiện 81 ngày đêm chiến đấu để bảo vệ Thành cổ Quảng Trị năm 1972.[2] 
        """,
        
        "Test": """
        Đây là phim hành động võ thuật Hồng Kông.
        """
    }

    print("=" * 60)
    for name, text in samples.items():
        print(f"\n{name}:")
        result = classify_movie_genre(text)
        print(f"  Genres: {result}")
        print(f"  Count: {len(result)}")
    print("\n" + "=" * 60)



run_sample()

# ============================================= 

# print(type(wiki_enrich))
# print(wiki_enrich.keys())
# print(wiki_enrich['Trấn Thành'].keys())

# update_graph()
import csv
import os
def write_result_to_csv(film, summary, res, file_name="test1/csv/movie_result.csv"):
    # Kiểm tra file đã tồn tại chưa
    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Nếu file chưa tồn tại thì ghi header
        if not file_exists:
            writer.writerow(["film", "summary", "result"])

        # Ghi dữ liệu
        writer.writerow([film, summary, res])


def run_update():
    for film in film_list:
        summary_of_film = wiki_enrich[film]['summary']
        key_list = classify_movie_genre(summary_of_film)
        res = format_genre_list(key_list)
        write_result_to_csv(film, summary_of_film, res)
        update_film_genre_and_save(B, film, res)
       
    

run_update()