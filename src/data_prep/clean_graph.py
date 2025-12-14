import json
import networkx as nx
import mwparserfromhell
from networkx.readwrite import json_graph
# đầu vào là graph cần làm sạch 

B_origin_path = "data/vn_bipartite_graph.json"
G_collab_origin_path = "data/vn_film_collaboration_graph.json"


wiki_dict_path = "data/wiki_dict.json"

# Load graph từ node-link JSON
def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data,link="edges" )

# print dòng đầu tiên của jsonl
import json

def print_first_n_jsonl(path, n=10):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(f"--- record {i+1} ---")
            print(json.dumps(obj, ensure_ascii=False, indent=2))


def find_empty_info_person_nodes(B, wiki_titles):
    """
    B           : networkx graph
    wiki_titles : list[str] – danh sách title wiki đã có
    """

    empty_nodes = []
    fount_list = []
    count_fount = 0
    count_not_fount = 0

    for node, attrs in B.nodes(data=True):
        node_type = attrs.get("type")
        info = attrs.get("info")

        # 1. chỉ lấy person + info rỗng
        if node_type == "person" and isinstance(info, dict) and len(info) == 0:
            person_id = attrs.get("id", node)

            # 2. tạo title dạng wiki
            wiki_title = f"{person_id} (diễn viên)"

            # 3. kiểm tra trong list title
            if wiki_title in wiki_titles:
                # print(f"[FOUND] {wiki_title}")
                count_fount +=1
                fount_list.append(wiki_title)
            else:
                # print(f"[NOT FOUND] {wiki_title}")
                count_not_fount +=1
                empty_nodes.append(person_id)

            
    print('found', count_fount)
    print('not fount', count_not_fount)
    return fount_list, empty_nodes


import json

def load_titles_from_jsonl(path):
    titles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "title" in obj:
                titles.append(obj["title"])
    return titles


import networkx as nx

def relabel_person_nodes_with_wiki_title(B, fount_list):
    
    # "Phương Oanh (diễn viên)" -> "Phương Oanh"

    # relabel
    mapping = {}
    for wiki_title in fount_list:
        person_name = wiki_title.replace(" (diễn viên)", "")

        # chỉ relabel nếu node cũ tồn tại
        if B.has_node(person_name):
            mapping[person_name] = wiki_title

    print(f"[RELABEL] Số node được đổi tên: {len(mapping)}")

    # inplace = False để an toàn 
    B = nx.relabel_nodes(B, mapping, copy=False)

    return B


def save_graph_json(B_input, path):
    data = nx.node_link_data(B_input,edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

import re

def normalize_person_ids(B):
    rename_map = {}
    nodes_to_remove = []

    for node, attrs in list(B.nodes(data=True)):
        if attrs.get("type") != "person":
            continue

        old_id = node
        info = attrs.get("info", {})

        # 1. Nếu có ngoặc chứa nghề nghiệp → giữ nguyên
        if re.search(
            r"\(.*?(diễn viên|ca sĩ|nghệ sĩ).*?\)",
            old_id,
            flags=re.IGNORECASE
        ):
            continue

        # 2. Cắt các hậu tố không mong muốn
        new_id = re.sub(
            r"\s+(diễn viên|nghệ sĩ|ca sĩ|đóng|vai)$",
            "",
            old_id,
            flags=re.IGNORECASE
        ).strip()

        if new_id == old_id or not new_id:
            continue

        # 3. Nếu id mới đã tồn tại
        if B.has_node(new_id):
            # info rỗng → xoá
            if not info:
                nodes_to_remove.append(old_id)
        else:
            rename_map[old_id] = new_id

    # xoá trước (tránh đụng relabel)
    if nodes_to_remove:
        print(f"[REMOVE] Xoá {len(nodes_to_remove)} node person info rỗng")
        B.remove_nodes_from(nodes_to_remove)

    # relabel sau
    if rename_map:
        print(f"[RENAME] Chuẩn hoá {len(rename_map)} node person")
        nx.relabel_nodes(B, rename_map, copy=False)

    if not nodes_to_remove and not rename_map:
        print("[NORMALIZE] Không có node nào cần xử lý")

    return B


import re

def is_garbage_name(name: str) -> bool:
    
    name = name.strip().lower()

    if len(name) <= 2:
        return True

    # đại từ, từ mơ hồ
    garbage_words = {
        "anh ấy", "cô ấy", "ông ấy", "bà ấy",
        "người này", "người kia", "nhân vật này",
        "diễn viên này", "ca sĩ này"
    }
    if name in garbage_words:
        return True

    # toàn ký tự lạ
    if not re.search(r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                     r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
                     r"ùúụủũưừứựửữỳýỵỷỹ]", name):
        return True

    return False

def filter_useless_nodes(B, debug=False):
    """
    Input : networkx graph B
    Output: (B_filtered, removed_nodes)
    """
    nodes_to_remove = []

    for node, attrs in B.nodes(data=True):
        node_type = attrs.get("type")
        info = attrs.get("info", {})
        degree = B.degree(node)
        # CHỈ XÓA PERSON
        if node_type != "person":
            continue
        # 1. node mồ côi + không có info
        if not info:
            if degree == 0:
                nodes_to_remove.append(node)
                continue
            # else:
            #     print(f"Person: {node}, info={bool(info)}, degree={degree}")

        # # 2. tên rác (kể cả có cạnh)
        # if is_garbage_name(node):
        #     nodes_to_remove.append(node)

    if debug:
        print(f"[CLEAN] Remove {len(nodes_to_remove)} nodes")

    B.remove_nodes_from(nodes_to_remove)
    return B, nodes_to_remove

def extract_infobox_from_wikitext(wikitext):
    
    if not wikitext:
        return {}
    try:
        # biến chuỗi wikitext => Wikicode obj
        code = mwparserfromhell.parse(wikitext)

    except Exception:
        return {}
    
    # duyệt qua toàn bộ cấu trúc wikitext, trả về list all các template (các khối {{ ... }})
    templates = code.filter_templates() 
   
    info = {}

    for template in templates:
        # template là obj

        # tên của template 
        name = str(template.name).lower()
        infoboxs = ["nhân vật", "phim", "diễn viên", "nghệ sĩ",  
                    "film", "truyền hình", "điện ảnh",
                    "television" , "person"]

        for infobox in infoboxs:
            if infobox in name.lower():
                # iterate params. in list tham so
                for param in template.params:
                    key = str(param.name).strip()
                    value = str(param.value).strip()
                    if key and value:
                        info[key] = value
                # break nếu k rỗng k false 
                if info:
                    break
    return info
def enrich_person_nodes_from_wiki_jsonl_stream(
    B, wiki_jsonl_path, fount_list, verbose=True
):
    targets = set(fount_list)
    found = 0
    no_infobox = 0

    with open(wiki_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title")

            if title not in targets:
                continue

            wikitext = obj.get("text")
            if not wikitext:
                no_infobox += 1
                targets.remove(title)
                continue
            
            infobox = extract_infobox_from_wikitext(wikitext)
            

            if infobox:
                B.nodes[title]["info"] = infobox
                found += 1
            else:
                no_infobox += 1

            targets.remove(title)

            if not targets:
                break

    if verbose:
        print(f"✔ Update infobox: {found}")
        print(f"✖ Không có infobox: {no_infobox}")

    return B

def run_and_save_updated_graph(G_input, output_path):
    wiki_dict_jsonl_dict = "data/wiki_dict.jsonl"
    title_list = load_titles_from_jsonl(wiki_dict_jsonl_dict)
    # print(title_list)
    G_input = normalize_person_ids(G_input)
    fount_list, empty_nodes = find_empty_info_person_nodes(G_input, title_list)
    B_rename_id_node = relabel_person_nodes_with_wiki_title(G_input,fount_list)
    B_enrich = enrich_person_nodes_from_wiki_jsonl_stream(B_rename_id_node, wiki_dict_jsonl_dict,fount_list )
        # sau khi add infobox rồi mới lọc node 
    B_enrich, removed_node_list = filter_useless_nodes(B_enrich, debug=True)
    
    # save_graph_json(B_rename_id_node, "data/graph/B_rename_id_node.json")
    save_graph_json(B_enrich, output_path)
    
def run():
    B_origin = load_graph(B_origin_path)
    G_collab_origin = load_graph(G_collab_origin_path)
    B_OUTPUT = "data/updated/B.json"
    G_OUTPUT = "data/updated/G_collab.json"
    # print_first_n_jsonl(wiki_dict_jsonl_dict, n=10)
    run_and_save_updated_graph(B_origin,B_OUTPUT)
    run_and_save_updated_graph(G_collab_origin,G_OUTPUT)
    print('Thực thi file clean graph')
run()

