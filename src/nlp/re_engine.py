# ==============================================================================
# 1. IMPORTS & SETUP 
# ==============================================================================
from src.nlp.ner import person_list, film_list, wiki_enrich, run_combine_ner
from src.data_prep.load_graph import B, G_collab
from src.nlp.text_utils import normalize_entity, normalize_entity_name,remove_footnotes,split_text_into_sentences
import re
import unicodedata
import itertools
# Thêm thư viện SetFit (nhớ pip install setfit trước)
from setfit import SetFitModel

# ==============================================================================
# 2. PRE-PROCESSING & NORMALIZATION 
# ==============================================================================

# Chỉ thêm entity vào graph_nodes nếu thỏa điều kiện
graph_nodes = []
for e in (person_list + film_list):
    if e in wiki_enrich:
        clean_text = wiki_enrich[e].get("clean_wikitext", "")
        if isinstance(clean_text, str) and clean_text.strip():
            graph_nodes.append(e)





# Regex Char cho tiếng Việt

# VN_WORD_CHAR = r"0-9A-Za-zÀ-ỹđĐ"
# VN_WORD_CHAR = r"A-Za-z0-9À-ỹđĐ"
# VN_WORD_CHAR = r"A-Za-z0-9À-ỿĐđ"
VN_WORD_CHAR = r"A-Za-z0-9À-ỹĐđ"


# ==============================================================================
# 3. SETFIT HELPERS (Thay thế logic của SpERT)
# ==============================================================================

def get_char_spans(clean_text, ner_out):
    """
    Thay thế hàm convert_ner_to_spans cũ.
    Mục tiêu: Tìm vị trí ký tự (start_char, end_char) của entity trong text
    để sau này chèn thẻ [TAG].
    """
    spans = []
    used_char_ranges = set()

    for ent in ner_out:
        raw_name = ent["name"]
        label = ent["type"] # Ví dụ: PER, ORG...
        # clean_text = normalize_entity_name(raw_name)
        norm_name = normalize_entity_name(raw_name)
        if not norm_name:
            continue

        # Tìm tất cả các vị trí xuất hiện của entity trong text
        pattern = rf"(?<![{VN_WORD_CHAR}])" + re.escape(norm_name) + rf"(?![{VN_WORD_CHAR}])"
        
        for match in re.finditer(pattern, clean_text, flags=re.IGNORECASE):
            start_char = match.start()
            end_char = match.end()

            # Kiểm tra trùng lặp vị trí (overlap)
            is_overlap = False
            for u_start, u_end in used_char_ranges:
                if not (end_char <= u_start or start_char >= u_end):
                    is_overlap = True
                    break
            
            if is_overlap:
                continue

            used_char_ranges.add((start_char, end_char))
            
            spans.append({
                "text": clean_text[start_char:end_char], # Text gốc trong câu
                "start": start_char,
                "end": end_char,
                "label": label
            })
    
    # Sắp xếp theo vị trí xuất hiện để dễ xử lý sau này
    spans.sort(key=lambda x: x["start"])
    return spans

def create_masked_text(text, span1, span2):
    """
    Hàm quan trọng nhất cho SetFit RE:
    Biến đổi: "Elon Musk mua Twitter" 
    Thành: "[PER] Elon Musk [/PER] mua [ORG] Twitter [/ORG]"
    """
    # Xử lý: Luôn thay thế từ thằng nằm sau trước để không làm lệch index thằng nằm trước
    # Sắp xếp 2 span theo thứ tự ngược (thằng nào start lớn hơn thì xử lý trước)
    pair = sorted([span1, span2], key=lambda x: x['start'], reverse=True)
    
    masked_text = text
    for p in pair:
        # Tạo chuỗi thay thế: [LABEL] text [/LABEL]
        replacement = f"[{p['label']}] {p['text']} [/{p['label']}]"
        
        # Cắt ghép chuỗi dựa trên index gốc
        masked_text = masked_text[:p['start']] + replacement + masked_text[p['end']:]
        
    return masked_text

# ==============================================================================
# 4. RUN SETFIT RELATION EXTRACTION
# ==============================================================================

def run_setfit_rel_extraction(
    clean_text,
    ner_out,
    setfit_model, # Truyền model SetFit vào đây
    person_list,
    film_list,
    wiki_enrich=None
):
    # TÁCH VĂN BẢN THÀNH TỪNG CÂU
    sentences = split_text_into_sentences(clean_text)
    triples = []

    for sent in sentences:
        sent = remove_footnotes(sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        # 1. Lấy vị trí các entity (Char Spans)
        entity_spans = get_char_spans(sent, ner_out)

        
        if len(entity_spans) < 2:
           continue # bỏ qa câu này # Cần ít nhất 2 entity để có quan hệ

        # 2. Tạo các cặp (Pairs) - Permutations (A->B và B->A có thể khác nhau)
        # Nếu quan hệ của là 2 chiều (như married_to), dùng combinations.
        # Nếu quan hệ có hướng (như father_of), dùng permutations.
        pairs = itertools.permutations(entity_spans, 2)
        # pairs = list(itertools.permutations(entity_spans, 2))
    
        # Chuẩn bị batch input để predict 1 lần cho nhanh
        batch_inputs = []
        batch_meta = [] # Lưu thông tin metadata để map lại kết quả
    
        for e1, e2 in pairs:
            
            # --- FILTER 2: cách nhau < 30 từ ---
            gap = abs(e1['start'] - e2['start'])
            if gap > 200:   # tương đương khoảng 30–40 từ
                continue

            # --- FILTER 3: ít nhất 1 trong 2 là person/film ---
            if not (e1["label"] in ["PER", "FILM", 'ORG', 'LOC'] or e2["label"] in ["PER", "FILM",'ORG', 'LOC']):
                continue

            # 3) Mask
            masked_input = create_masked_text(sent, e1, e2)
            masked_input = re.sub(r"\s+", " ", masked_input).strip()

            batch_inputs.append(masked_input)
            batch_meta.append((e1, e2))

        # Không có pair hợp lệ
        if not batch_inputs:
            continue

        # 4) Predict
        predictions = setfit_model.predict(batch_inputs)

        # 5) Map output
        for idx, pred in enumerate(predictions):
            rel = str(pred).upper().strip()
            if rel in ("NO_RELATION", "NONE", "O", ""):
                continue

            e1, e2 = batch_meta[idx]
            subj_raw = normalize_entity_name(e1['text'])
            obj_raw  = normalize_entity_name(e2['text'])

            subj_norm, subj_type = normalize_entity(subj_raw, person_list, film_list, wiki_enrich)
            obj_norm,  obj_type  = normalize_entity(obj_raw,  person_list, film_list, wiki_enrich)

            # --- FILTER UNKNOWN chỉ nếu không phải person/film ---
            if subj_type == "Unknown" and subj_norm not in film_list + person_list:
                continue
            if obj_type == "Unknown" and obj_norm not in film_list + person_list:
                continue

            # tránh self-loop
            if subj_norm == obj_norm:
                continue
            # --- NEW FILTER 4: Check type compatibility ---
            if not is_valid_pair(subj_type, obj_type, rel):
                # print("Invalid type pair:", (subj_type, obj_type), "for relation", rel)
                continue
            triples.append((subj_norm, pred, obj_norm))

    # unique
    triples = list(dict.fromkeys(triples))
    return triples

def run_setfit_rel_extraction_debug(
    clean_text,
    ner_out,
    setfit_model,
    person_list,
    film_list,
    wiki_enrich=None,
    debug=True
):

    if debug:
        print("\n===== DEBUG WITH SENTENCE SPLIT =====\n")

    # ================================================================
    # COMMENT IN HOA: TÁCH VĂN BẢN THÀNH TỪNG CÂU
    sentences = split_text_into_sentences(clean_text)
    # ================================================================

    triples = []

    for si, sent in enumerate(sentences):

        if debug:
            print(f"\n--- SENTENCE {si}: {sent}\n")

        sent = remove_footnotes(sent)
        sent = re.sub(r"\s+", " ", sent).strip()

        spans = get_char_spans(sent, ner_out)

        if debug:
            print("Entity spans:", spans)

        if len(spans) < 2:
            continue

        pairs = list(itertools.permutations(spans, 2))
        batch_inputs = []
        meta = []

        for e1, e2 in pairs:

            if abs(e1['start'] - e2['start']) > 200:
                if debug: print("Skip: gap too large")
                continue

            masked = create_masked_text(sent, e1, e2)
            masked = re.sub(r"\s+", " ", masked)

            if debug:
                print("Masked:", masked)

            batch_inputs.append(masked)
            meta.append((e1, e2))

        if not batch_inputs:
            continue

        preds = setfit_model.predict(batch_inputs)

        if debug:
            print("Preds:", preds)

        for idx, pred in enumerate(preds):
            rel = str(pred).upper().strip()
            if rel in ["NO_RELATION", "NONE", "O", ""]:
                if debug: print("Skip no relation")
                continue

            e1, e2 = meta[idx]

            subj_raw = normalize_entity_name(e1['text'])
            obj_raw = normalize_entity_name(e2['text'])

            subj_norm, subj_type = normalize_entity(subj_raw, person_list, film_list, wiki_enrich)
            obj_norm, obj_type = normalize_entity(obj_raw, person_list, film_list, wiki_enrich)

            if not is_valid_pair(subj_type, obj_type, rel):
                if debug: print("Invalid type pair, skip")
                continue

            if subj_norm == obj_norm:
                continue

            triples.append(((subj_norm, subj_type), rel, (obj_norm, obj_type)))
            if debug:
                print("✓ ADD TRIPLE:", ((subj_norm, subj_type), rel, (obj_norm, obj_type)))

    triples = list(dict.fromkeys(triples))

    if debug:
        print("\n======= FINAL TRIPLES =======")
        print(triples)

    return triples



# Set global relation set để dedup toàn cục


def dedup_triples(triples, seen_set=None):
    """
    Dedup triples, có thể dùng set cục bộ hoặc toàn cục
    """
    if seen_set is None:
        seen_set = set()  # Set cục bộ cho mỗi lần gọi
    
    new = []
    for s, r, o in triples:
        # Đảm bảo s và o là strings, không phải tuples
        if isinstance(s, tuple):
            s_str = s[0] if isinstance(s[0], str) else str(s[0])
        else:
            s_str = str(s)
            
        if isinstance(o, tuple):
            o_str = o[0] if isinstance(o[0], str) else str(o[0])
        else:
            o_str = str(o)
        
        key = (s_str, r, o_str)
        if key not in seen_set:
            seen_set.add(key)
            new.append((s, r, o))
    return new

# ==============================================================================
# 5. MAIN PROCESS (Đã cập nhật để dùng SetFit)
# ==============================================================================

# Load SetFit Model (cần train trước và lưu vào folder hoặc dùng path huggingface)
#  đã train trước và lưu model 


try:
    # Load model RE
    # Nếu chưa train, có thể comment dòng này lại để test logic code trước
    re_model = SetFitModel.from_pretrained("data/re_model") 
    print("SetFit model loaded successfully.")
except Exception as e:
    print("Chưa load được model SetFit (hãy train trước):", e)
    re_model = None

RELATION_TYPES = {
    "ACTED_IN",
    "DIRECTED",
    "SPOUSE_OF",
    "COLLABORATED_WITH",
    "SAME_HOMETOWN_AS",
    "SAME_SCHOOL_AS"
}
VALID_TYPES = {
    "ACTED_IN": [("PER", "FILM")],
    "DIRECTED": [("PER", "FILM")],
    
    "SPOUSE_OF": [("PER", "PER")],
    "COLLABORATED_WITH": [("PER", "PER")],

    "SAME_HOMETOWN_AS" : [("PER", "PER")],
    "SAME_SCHOOL_AS": [("PER", "PER")],
   
}

def is_valid_pair(subj_type, obj_type, relation):
    relation = relation.upper()
    valid = VALID_TYPES.get(relation, [])
    return (subj_type, obj_type) in valid



RE_RES_FILE = f"data/re_results.jsonl"


if re_model is not None:

    for entity in graph_nodes:

        if entity not in wiki_enrich:
            continue

        clean_text = wiki_enrich[entity].get("clean_wikitext", "")
        if not clean_text:
            continue

        # --------------------------
        # 1) NER kết hợp
        # --------------------------
        ner_out = run_combine_ner(
            text=clean_text,
            person_list=person_list,
            film_list=film_list,
            wiki_enrich=wiki_enrich,
            bipartite_graph=B
        )

        # --------------------------
        # 2) RE bằng SetFit
        # --------------------------
        relations = run_setfit_rel_extraction_debug(
            clean_text=clean_text,
            ner_out=ner_out,
            setfit_model=re_model,
            person_list=person_list,
            film_list=film_list,
            wiki_enrich=wiki_enrich
        )

        # --------------------------
        # 3) Dedup
        # --------------------------
        # 3) Lọc quan hệ sai schema (dựa vào VALID_TYPES)
        filtered_relations = []
        for (s, r, o) in relations:
            # s và o là tuple (name, type)
            subj_type = s[1]  # Lấy type từ tuple
            obj_type = o[1]   # Lấy type từ tuple
            
            if is_valid_pair(subj_type, obj_type, r):
                filtered_relations.append((s, r, o))
        # 4) Dedup
        filtered_relations = dedup_triples(filtered_relations)


        # --------------------------
        # 4) Xuất file cho entity
        # --------------------------
        
        # Mỗi lần chạy toàn pipeline → reset file
        # (Chỉ reset 1 lần ở entity đầu tiên)
        if entity == graph_nodes[0]:  
            open(RE_RES_FILE, "w", encoding="utf-8").close()

        # Chuẩn bị danh sách quan hệ cho entity này
        import json
        relations_list = []
        for (s, r, o) in filtered_relations:
            s_name = s[0] if isinstance(s, tuple) else s
            o_name = o[0] if isinstance(o, tuple) else o
            
            relations_list.append({
                "subject": s_name,
                "relation": r,
                "object": o_name
            })

        # Ghi 1 dòng JSON cho entity
        with open(RE_RES_FILE, "a", encoding="utf-8") as f:
            json.dump({
                "entity": entity,
                "relations": relations_list
            }, f, ensure_ascii=False)
            f.write("\n")

        print(f"Wrote RE for {entity} → {RE_RES_FILE}")
        # --------------------------
        # 5) In ra console
        # --------------------------
        # print("\n************************* RE ********************************")
        # if filtered_relations:
        #     print(f"[{entity}] → {filtered_relations}")
        # else:
        #     print(f"[{entity}] → (no extracted relations)")

else:
    print("Chưa train model SetFit RE!")


















# test thử 1 entity cụ thể =================================
# entity = "Trấn Thành"
# print(wiki_enrich[entity].keys())
# # clean_text = "Ngày 25 tháng 12 năm 2016, Trấn Thành kết hôn với nữ ca sĩ mang hai dòng máu Việt-Hàn Hari Won[7] tại Trung tâm Hội nghị Gem Center, Thành phố Hồ Chí Minh.[8] Anh có hai người em gái tên Huỳnh Trinh Mi và Huỳnh Uyển Ân, trong đó Uyển Ân cũng trở thành một diễn viên như anh sau khi cô tham gia đóng chính trong phim Nhà bà Nữ."


# clean_text = wiki_enrich[entity]["clean_wikitext"]

# ner_out = run_combine_ner(clean_text, person_list, film_list, wiki_enrich, B)

# rels = run_setfit_rel_extraction_debug(
#     clean_text,
#     ner_out,
#     re_model,
#     person_list,
#     film_list,
#     wiki_enrich
# )

# import os
# os.makedirs("test/re", exist_ok=True)

# out_path = f"test/re/{entity}.txt"
# with open(out_path, "w", encoding="utf-8") as f:
#     for (s, r, o) in rels:
#         f.write(f"{s}\t{r}\t{o}\n")

# print("Saved:", out_path)


# print('+++++++++++++++ test thử ++++++++++++++++++++++++++')
# # Test với các câu đơn giản
# test_sentences = [
#     "[PER] A [/PER] kết hôn với [PER] B [/PER].",
#     "[PER] A [/PER] đóng vai chính trong [FILM] C [/FILM].",
#     "[PER] A [/PER] đạo diễn phim [FILM] C [/FILM].",
#     "[PER] A [/PER] hợp tác với [PER] B [/PER] trong dự án.",
#     "[PER] A [/PER] và [PER] B [/PER] cùng quê ở [LOC] Hà Nội [/LOC]."
# ]

# predictions = re_model.predict(test_sentences)
# for sent, pred in zip(test_sentences, predictions):
#     print(f"{sent} ====> {pred}")