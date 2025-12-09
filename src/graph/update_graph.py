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



re_res_dict = load_jsonl_to_dict("data/re_results.jsonl", "entity")


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


print(find_existing_node(G_collab, "Trấn Thành"))

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


save_graph_json(B_updated, "data/updated/B.json")
save_graph_json(G_collab_updated, "data/updated/G_collab.json")