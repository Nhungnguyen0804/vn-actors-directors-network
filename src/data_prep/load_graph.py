# RUN FILE NÀY: python -m src.data_prep.load_graph
import json
import networkx as nx
print(nx.__version__)
from networkx.readwrite import json_graph

from src.constant import G_COLLAB_JSON, BIPARTITE_JSON
'''
load đồ thị 2 phía để lấy list film list actor

'''
import networkx, sys
print("version:", networkx.__version__)
print("networkx path:", networkx.__file__)
print("python:", sys.executable)
print('==================================')

import json
def check_keys(path):
    data = json.load(open(path, encoding="utf-8"))
    print(data.keys())


# Load graph từ node-link JSON
def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data,link="edges" )

def load_bipartite_graph_and_nodes(B):
    # Tách danh sách ACTORS & MOVIES
    person_list = []
    film_list = []
    for node, attrs in B.nodes(data=True):
        ntype = attrs.get("type")
        if ntype == "person":
            person_list.append(node)
        elif ntype == "film":
            film_list.append(node)
    
    return B, person_list, film_list




print('check key trong file json +++++++++++')
check_keys(BIPARTITE_JSON)
check_keys(G_COLLAB_JSON)
print('======================================')

# Load graph
B = load_graph(BIPARTITE_JSON)
print(f"Tổng số nodes: {B.number_of_nodes()}")
print(f"Tổng số edges: {B.number_of_edges()}")

# Kiểm tra thử 1 node xem có ID chuẩn không (Trấn Thành)
if "Trấn Thành" in B.nodes:
    print("\nNode 'Trấn Thành' tồn tại.")
    print("Attributes:", B.nodes["Trấn Thành"])
else:
    print("\nKhông tìm thấy node 'Trấn Thành'. ID có thể đang bị sai.")
    # In thử 5 node đầu tiên để xem ID là gì
    print("5 Node ID đầu tiên:", list(B.nodes())[:5])

# Kiểm tra role của node (để tách danh sách phim/diễn viên)
print("\n--- Kiểm tra phân loại Node ---")




#TEST 
def debug_json_structure(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"--- DEBUG FILE: {path} ---")
    print(f"Keys in JSON: {data.keys()}")
    
    if "nodes" in data:
        print(f"Sample Node 0: {data['nodes'][0]}")
        # Nếu sample node có chứa 'source' và 'target' -> File JSON bị sai nguồn
        if 'source' in data['nodes'][0]:
            print("!!! Key 'nodes' đang chứa dữ liệu của links/edges !!!")
            
    if "edges" in data:
        print(f"Sample Link 0: {data['edges'][0]}")

# Chạy debug
debug_json_structure(BIPARTITE_JSON)

B, person_list, film_list = load_bipartite_graph_and_nodes(B)
G_collab = load_graph(G_COLLAB_JSON)
#
# In kết quả kiểm tra
print("Số person:", len(person_list))
print("Số phim:", len(film_list))
print("Số node trong collab graph:", G_collab.number_of_nodes())
print("Số cạnh collab graph:", G_collab.number_of_edges())

print("Ví dụ 10 actor:", person_list[:10])
print("Ví dụ 10 movie:", film_list[:10])


debug_json_structure(G_COLLAB_JSON)

print('**************************************************')
print("Node Trấn Thành:", G_collab.nodes["Trấn Thành"])
print('-------------------------------------------------')
print("Edge Trấn Thành - Uyển Ân:", G_collab["Trấn Thành"]["Uyển Ân"])


