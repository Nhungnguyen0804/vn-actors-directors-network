from src.data_prep.load_graph import B, G_collab
from collections import defaultdict
from src.nlp.text_utils import normalize_entity, normalize_entity_name,remove_text_in_parentheses,split_text_into_sentences
import json
def detect_spouse_relations(G_collab):
    seen_pairs = set()
    out = []
    
    # Tạo mapping từ tên rút gọn sang tên đầy đủ trong graph
    name_to_node = {}
    for node_id in G_collab.nodes():
        if G_collab.nodes[node_id].get("type") == "person":
            # Lưu cả tên đầy đủ
            name_to_node[node_id] = node_id
            
            # Lưu các biến thể tên (không có prefix, suffix)
            base_name = node_id.split("(")[0].strip()
            if base_name not in name_to_node:
                name_to_node[base_name] = node_id
    
    for n, data in G_collab.nodes(data=True):
        if data.get("type") != "person":
            continue
            
        info = data.get("info", {})
        spouse_raw = info.get("spouse")
        
        if not spouse_raw:
            continue
            
        # Chuẩn hóa tên spouse
        spouse_name_raw = spouse_raw.split("(")[0].strip()
        
        # Tìm node ID chính xác trong graph
        spouse_node_id = name_to_node.get(spouse_name_raw)
        
        if not spouse_node_id:
            # Thử tìm với tên đầy đủ
            spouse_node_id = spouse_name_raw if G_collab.has_node(spouse_name_raw) else None
        
        if spouse_node_id:
            # Tạo cặp đã sắp xếp để tránh trùng lặp ngược chiều
            pair = tuple(sorted([n, spouse_node_id]))
            
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                # Lưu theo thứ tự alphabet
                out.append((pair[0], "SPOUSE_OF", pair[1] , {}))
    
    return out



def find_co_spouse_groups(pairs):
    # tìm các cặp bị lặp
    person_to_spouses = defaultdict(set)
    for a, _, b in pairs:
        person_to_spouses[a].add(b)
        person_to_spouses[b].add(a)
    
   
    co_spouse_map = defaultdict(set)
    
    for person, spouses in person_to_spouses.items():
        for spouse in spouses:
          
            other_spouses = person_to_spouses[spouse] - {person}
            for other in other_spouses:
             
                key = tuple(sorted([person, other]))
                co_spouse_map[key].add(spouse)
    
    # Chuyển thành: person -> set of co-spouses
    name_to_co_spouses = defaultdict(set)
    for (p1, p2), spouses in co_spouse_map.items():
        name_to_co_spouses[p1].add(p2)
        name_to_co_spouses[p2].add(p1)
    
    return name_to_co_spouses



def normalize_spouse_pairs(pairs):
   
    # Step 1: gom spouse theo anchor
    spouse_map = defaultdict(list)

    for a, _, b, _ in pairs:
        spouse_map[a].append(b)
        spouse_map[b].append(a)

    # Step 2: tìm nhóm cần chuẩn hóa
    name_fix_map = {}

    for anchor, spouses in spouse_map.items():
        if len(spouses) < 2:
            continue

        # group theo clean name
        clean_groups = defaultdict(list)
        for s in spouses:
            clean = remove_text_in_parentheses(s)
            clean_groups[clean].append(s)

        # nếu 1 clean name map tới nhiều variant → chọn bản "đẹp nhất"
        for clean, variants in clean_groups.items():
            if len(variants) < 2:
                continue

            # chọn variant dài nhất (thường là tên đầy đủ hơn)
            best = max(variants, key=len)

            for v in variants:
                if v != best:
                    name_fix_map[v] = best

    # Step 3: Apply normalize
    normalized = []

    for a, rel, b, _ in pairs:
        # Nếu input có dùng "SPOUSE" cũ, chuẩn hoá về "SPOUSE_OF"
        if rel == "SPOUSE":
            rel = "SPOUSE_OF"
        a_new = name_fix_map.get(a, a)
        b_new = name_fix_map.get(b, b)

        # sort lại để giữ canonical order
        pair = tuple(sorted([a_new, b_new]))
        normalized.append((pair[0], rel, pair[1], {}))

    # Step 4: remove duplicate (chỉ dựa trên a, rel, b, không dùng metadata)
    seen = set()
    deduped = []
    for a, rel, b, metadata in normalized:
        key = (a, rel, b)  # CHỈ dùng 3 field đầu để check duplicate
        if key not in seen:
            seen.add(key)
            deduped.append((a, rel, b, metadata))
    

    return deduped


def canonical_name(name, all_names):
    tokens = name.split()
    if len(tokens) <= 1:
        return name  # tên quá ngắn, không canonical hóa được
    
    best = name
    
    for other in all_names:
        if other == name:
            continue
        
        other_tokens = other.split()
        
        # Chỉ xét canonical nếu tên dài hơn hoặc bằng
        if len(other_tokens) < len(tokens):
            continue
        
        # Rule 1: tên dài hơn và kết thúc bằng tên ngắn
        if other.endswith(name) and len(other_tokens) > len(tokens):
            # Kiểm tra thêm: các token phải match theo thứ tự
            # "Ngọc Diệp" phải là 2 từ cuối của "Đinh Ngọc Diệp"
            if other_tokens[-len(tokens):] == tokens:
                best = other
                break  # tìm thấy canonical tốt nhất
    
    return best

def dedup_spouse_pairs(raw_pairs):
   
    # B1: thu thập tất cả tên
    all_names = set()
    for a, rel, b in raw_pairs:
        all_names.add(a)
        all_names.add(b)
    
    co_spouse_groups = find_co_spouse_groups(raw_pairs)
    
    # B3: Xây dựng canonical map CHỈ trong các nhóm co-spouse
    canonical = {}
    for name in all_names:
        canonical[name] = name  # mặc định
        
        if name in co_spouse_groups:
            # Nhóm tất cả tên cần xét (bao gồm name và các co-spouses)
            group_names = list(co_spouse_groups[name]) + [name]
            
            # Tìm tên đầy đủ nhất trong nhóm
            best = name
            for other in group_names:
                other_tokens = other.split()
                best_tokens = best.split()
                
                # Ưu tiên tên dài hơn
                if len(other_tokens) > len(best_tokens):
                    # Kiểm tra nếu other chứa name (ví dụ: "Đinh Ngọc Diệp" chứa "Ngọc Diệp")
                    if other.endswith(name) or name in other:
                        best = other
                # Nếu cùng độ dài, ưu tiên tên có dấu ngoặc 
                elif len(other_tokens) == len(best_tokens):
                    if "(" in other and "(" not in best:
                        best = other
            
            canonical[name] = best
    
    # Debug: in các mapping
    print("\n=== CANONICAL MAPPING (chỉ khi có chung vợ/chồng) ===")
    any_mapping = False
    for orig, canon_name in canonical.items():
        if orig != canon_name:
            print(f"  {orig} -> {canon_name}")
            any_mapping = True
    if not any_mapping:
        print("  (Không có mapping nào)")
    print("==================================================\n")
    
    # B4: dedup
    seen = set()
    result = []
    
    for a, rel, b in raw_pairs:
        ca = canonical[a]
        cb = canonical[b]
        
        # Skip nếu canonical trùng nhau
        if ca == cb:
            continue
        
        # Tạo key duy nhất (không quan tâm thứ tự)
        key = tuple(sorted([ca, cb]))
        
        if key not in seen:
            seen.add(key)
            result.append((ca, rel, cb))
    
    return result




'''
('Hari Won', 'SPOUSE', 'Trấn Thành')
('Hồng Đào', 'SPOUSE', 'Quang Minh')
('Victor Vũ', 'SPOUSE', 'Đinh Ngọc Diệp')
('Lê Văn Anh', 'SPOUSE', 'Tú Vi')
('Hải Yến', 'SPOUSE', 'Khương Ngọc')
'''



# cùng quê
HOMETOWN_PATTERNS = [
    r"quê ở",
    r"sinh ra tại",
    r"xuất thân từ",
    r"người.*đến từ",
    r"người.*quê.*",
]
import re
from collections import defaultdict
def detect_same_hometown(G_collab):
    """
    Phát hiện quan hệ 'cùng quê quán' giữa các person trong G_collab.
    Dựa trên quê quán thật từ infobox của node (ưu tiên 'quê quán', fallback 'birth_place' nếu không có).
    Chuẩn hóa quê quán để so sánh (lower case, strip, lấy phần tỉnh/quốc gia nếu có định dạng '..., Tỉnh, Quốc gia').

    Returns:
        List[Tuple[str, str, str]]:
            [(personA, "SAME_HOMETOWN_AS", personB), ...]
            Đã loại trùng lặp và chuẩn hoá thứ tự A < B.
    """
    # Thu thập tất cả person và quê quán của họ
    hometowns = defaultdict(list)  # hometown_normalized -> list of persons
    original_hometowns = {}  # person -> original hometown string
    possible_keys = ['quê quán', 'quê', 'hometown', 'birth_place']  # Các key có thể trong infobox, ưu tiên theo thứ tự

    for node, data in G_collab.nodes(data=True):
        if data.get("type") != "person":
            continue
        info = data.get("info", {})
        if not info:
            continue
        
        # Tìm quê quán từ các key возможные, ưu tiên đầu tiên có giá trị
        hometown = None
        for key in possible_keys:
            if key in info and info[key]:
                hometown = info[key]
                break
        
        if not hometown:
            continue  # Skip nếu không có quê quán
        original_hometowns[node] = hometown
        # Chuẩn hóa quê quán
        # - Lower case
        # - Remove extra spaces
        # - Nếu có comma, lấy last 2 parts (ví dụ: 'ABC, Quảng Đông, Trung Quốc' -> 'Quảng Đông, Trung Quốc')
        # - Loại bỏ dấu ngoặc, text thừa nếu cần (cải tiến: dùng regex đơn giản)
        hometown = re.sub(r'\(.*?\)', '', hometown)  # Loại phần trong ngoặc (nếu có note)
        hometown = hometown.strip().lower()
        parts = [p.strip() for p in hometown.split(',') if p.strip()]
        if len(parts) > 1:
            normalized = ', '.join(parts[-2:])  # Lấy 2 phần cuối: tỉnh, quốc gia
        else:
            normalized = parts[0] if parts else ''
        
        if normalized:
            hometowns[normalized].append(node)
    
    # Tạo cặp từ các group có >=2 persons
    out = []
    seen = set()
    # FIX: Dùng .items() thay vì .values()
    for normalized, persons in hometowns.items():  # <-- SỬA Ở ĐÂY
        if len(persons) < 2:
            continue
        # Capitalize normalized cho output đẹp
        normalized_cap = ', '.join(word.capitalize() for word in normalized.split(', '))

        # Sort persons để thứ tự nhất quán
        persons = sorted(persons)
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                a, b = persons[i], persons[j]
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    out.append((a, "SAME_HOMETOWN_AS", b,{"hometown": normalized_cap}))
    
    return out
# ĐÓNG TRONG PHIM
def detect_acted_in(G_bipartite):
    out = []
    for u, v, data in G_bipartite.edges(data=True):
        if data.get("role", "").lower() == "actor":
            out.append((u, "ACTED_IN", v, {}))
    return out


def detect_directed(G_bipartite):
    out = []

    for u, v, data in G_bipartite.edges(data=True):
        role = data.get("role", "").lower()

        if role == "director":
            out.append((u, "DIRECTED", v, {}))

    return out

def detect_collaboration_with_weight(G_collab, min_films=1, min_weight=0.0):
    """
    Phát hiện quan hệ hợp tác với ngưỡng tùy chỉnh.
        G_collab: Đồ thị collaboration
        min_films: Số phim hợp tác tối thiểu (default: 1)
        min_weight: Trọng số hợp tác tối thiểu (default: 0.0)
    
        List[Tuple[str, str, str, dict]]:
            [(personA, "COLLABORATED_WITH", personB, metadata), ...]
            metadata chứa film_count, films, weight
    """
    out = []
    seen = set()

    for u, v, data in G_collab.edges(data=True):
        film_count = data.get("film_count", 0)
        weight = data.get("weight", 0.0)
        films = data.get("films", [])
        
        # Điều kiện
        if film_count < min_films or weight < min_weight:
            continue

        # Chỉ lấy person–person
        if G_collab.nodes[u].get("type") != "person":
            continue
        if G_collab.nodes[v].get("type") != "person":
            continue

        # Chuẩn hóa thứ tự
        a, b = sorted([u, v])
        key = (a, b)
        
        if key not in seen:
            seen.add(key)
            metadata = {
                "film_count": film_count,
                "films": films,
                "weight": weight,
                "collaboration_types": data.get("collaboration_types", [])
            }
            out.append((a, "COLLABORATED_WITH", b, metadata))

    return out
def detect_collaboration(G_collab):
    return detect_collaboration_with_weight(G_collab, min_films=1, min_weight=0.0)


def detect_same_school(G_collab):
    """
    Phát hiện quan hệ 'cùng trường học' giữa các person trong graph.
    """
    school_keys = ["alma_mater", "education", "học_vấn", "education_background"]

    school_map = defaultdict(list)
    import re

    for node, data in G_collab.nodes(data=True):
        if data.get("type") != "person":
            continue

        info = data.get("info", {})
        school = None

        for key in school_keys:
            if key in info and info[key]:
                school = info[key]
                break

        if not school:
            continue

        # Chuẩn hóa trường học
        s = school
        s = re.sub(r'\(.*?\)', '', s)
        s = s.strip().lower()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        normalized = parts[-1] if parts else ""

        if normalized:
            school_map[normalized].append(node)

    out = []
    seen = set()

    for normalized, persons in school_map.items():
        if len(persons) < 2:
            continue

        persons = sorted(persons)
        norm_cap = normalized.capitalize()

        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                a, b = persons[i], persons[j]
                if (a, b) not in seen:
                    seen.add((a, b))
                    out.append((a, "SAME_SCHOOL_AS", b, {"school": norm_cap}))

    return out



raw = normalize_spouse_pairs(detect_spouse_relations(G_collab))
same_hometown_pairs = detect_same_hometown(G_collab)


acted_in_pairs = detect_acted_in(B)
directed_pairs = detect_directed(B)
collaboration_pairs = detect_collaboration(G_collab)


def collect_all_relations():
    triples = []

    triples += normalize_spouse_pairs(detect_spouse_relations(G_collab))
    triples += detect_same_hometown(G_collab)
    triples += detect_acted_in(B)
    triples += detect_directed(B)
    triples += detect_same_school(G_collab)


    for a, rel, b, meta in detect_collaboration(G_collab):
        triples.append((a, rel, b, meta ))

    return triples

triples = collect_all_relations()

with open("data/triples_for_RE.json", "w", encoding="utf8") as f:
    json.dump(triples, f, ensure_ascii=False, indent=2)

print('Done DETECT!')


# ==== TEST 1: SPOUSE ====
raw_spouse = detect_spouse_relations(G_collab)
normalized_spouse = normalize_spouse_pairs(raw_spouse)

print("\n=== SPOUSE TEST ===")
print("Raw spouse count:", len(raw_spouse))
print("Normalized spouse count:", len(normalized_spouse))

assert all(rel == "SPOUSE_OF" for _, rel, _ , _ in normalized_spouse)

# ==== TEST 2: SAME_HOMETOWN_AS ====
same_hometown = detect_same_hometown(G_collab)
same_school = detect_same_school(G_collab)
print("\n=== SAME_HOMETOWN TEST ===")
print("Count:", len(same_hometown))
assert all(rel == "SAME_HOMETOWN_AS" for _, rel, _ ,_ in same_hometown)

# ==== TEST 3: ACTED_IN ====
acted_in = detect_acted_in(B)
print("\n=== ACTED_IN TEST ===")
print("Count:", len(acted_in))
assert all(rel == "ACTED_IN" for _, rel, _ , _ in acted_in)

# ==== TEST 4: DIRECTED ====
directed = detect_directed(B)
print("\n=== DIRECTED TEST ===")
print("Count:", len(directed))
assert all(rel == "DIRECTED" for _, rel, _ , _ in directed)

# ==== TEST 5: COLLABORATED_WITH ====
collab = detect_collaboration(G_collab)
print("\n=== COLLABORATION TEST ===")
print("Count:", len(collab))
assert all(rel == "COLLABORATED_WITH" for _, rel, _,_ in collab)

# ==== TEST 6: Collect All Relations ====
triples = collect_all_relations()

print("\n=== ALL TRIPLES TEST ===")
print("Total triples:", len(triples))
assert any(t[1] == "SPOUSE_OF" for t in triples)
assert any(t[1] == "SAME_HOMETOWN_AS" for t in triples)
assert any(t[1] == "ACTED_IN" for t in triples)
assert any(t[1] == "DIRECTED" for t in triples)
assert any(t[1] == "COLLABORATED_WITH" for t in triples)
assert any(t[1] == "SAME_SCHOOL_AS" for t in triples)
print('Done TEST')


for x in normalized_spouse[:10]:
    print(x)
for x in same_hometown[:10]:
    print(x)
for x in acted_in[:10]:
    print(x)
for x in directed[:10]:
    print(x)
for x in collaboration_pairs[:10]:
    print(x)
for x in same_school[:10]:
    print(x)