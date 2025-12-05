# RUN FILE NÀY: python -m src.data_prep.load_graph
import json
import networkx as nx
print(nx.__version__)
from networkx.readwrite import json_graph

from src.constant import ACTOR_COLLABORATION_JSON, BIPARTITE_JSON
'''
load đồ thị 2 phía để lấy list film list actor

'''
import networkx, sys
print("IPYNB version:", networkx.__version__)
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
    return json_graph.node_link_graph(data, link="edges")

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
check_keys(ACTOR_COLLABORATION_JSON)
print('======================================')

# Load graph
B = load_graph(BIPARTITE_JSON)
G_collab = load_graph(ACTOR_COLLABORATION_JSON)

#TEST 
B, person_list, film_list = load_bipartite_graph_and_nodes(B)

#
# In kết quả kiểm tra
print("Số person:", len(person_list))
print("Số phim:", len(film_list))
print("Số node trong collab graph:", G_collab.number_of_nodes())
print("Số cạnh collab graph:", G_collab.number_of_edges())

print("Ví dụ 10 actor:", person_list[:10])
print("Ví dụ 10 movie:", film_list[:10])

