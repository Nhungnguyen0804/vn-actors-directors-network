import networkx as nx
import community.community_louvain as community_louvain
import json

from src.data_prep.load_graph import load_graph   
def compute_page_rank(G):
    """
    Tính PageRank cho đồ thị
    """
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    return pagerank_scores


def compute_degree_centrality(G):
    """Tính độ trung tâm bậc của đồ thị"""
    return nx.degree_centrality(G)


def compute_louvain_communities(G):
    """
    Phát hiện cộng đồng sử dụng thuật toán Louvain
    """
    
    partition = community_louvain.best_partition(G)
    return partition


def get_actor_subgraph(G, actor_name):
    """
    Lấy đồ thị con của một diễn viên cụ thể
    """
    if actor_name not in G:
        print(f"Diễn viên {actor_name} không có trong đồ thị.")
        return None
    
    neighbors = list(G.neighbors(actor_name))
    subgraph_nodes = neighbors + [actor_name]
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph


def shortest_path_between_actors(G, actor1, actor2):
    """
    Tìm đường đi ngắn nhất giữa hai diễn viên
    """
    try:
        path = nx.shortest_path(G, source=actor1, target=actor2)
        return path
    except nx.NetworkXNoPath:
        print(f"Không có đường đi giữa {actor1} và {actor2}.")
        return None
    except nx.NodeNotFound as e:
        print(e)
        return None
    
    
def compute_betweenness(G):
    """
    Tính độ trung gian (betweenness centrality) cho đồ thị
    """
    return nx.betweenness_centrality(G)


def analyze_actor_importance(G):
    """
    Phân tích tầm quan trọng của diễn viên trong đồ thị
    """
    pagerank = compute_page_rank(G)
    degree_centrality = compute_degree_centrality(G)
    betweenness = compute_betweenness(G)
    
    importance_data = {}
    for actor in G.nodes():
        importance_data[actor] = {
            'pagerank': pagerank.get(actor, 0),
            'degree_centrality': degree_centrality.get(actor, 0),
            'betweenness': betweenness.get(actor, 0)
        }
    
    return importance_data

G_actor_collab = load_graph('data/updated/G_collab.json')
page_rank = compute_page_rank(G_actor_collab)
print('Page Rank:', page_rank)


# actor_importance = analyze_actor_importance(G_actor_collab)
# degree_centrality = compute_degree_centrality(G_actor_collab)

# print('Actor Importance:', actor_importance)
# print('Degree Centrality:', degree_centrality)
# group  = compute_louvain_communities(G_actor_collab)
# for comm_id in set(group.values()):
#     members = [actor for actor, comm in group.items() if comm == comm_id]
#     print(f"Community {comm_id}: {members}")
    
# actor_subgraph = get_actor_subgraph(G_actor_collab, "Ngô Thanh Vân")
# print(f"Số nút trong đồ thị con của Ngô Thanh Vân: {actor_subgraph.number_of_nodes()}")
# for neighbor in actor_subgraph.neighbors("Ngô Thanh Vân"):
#     print(f"Ngô Thanh Vân hợp tác với: {neighbor}")
    
# betweenness = compute_betweenness(G_actor_collab)
# print("Betweenness Centrality:")
# for actor, score in betweenness.items():
#     print(f"Actor: {actor}, Betweenness: {score}")
    
# actor_importance = analyze_actor_importance(G_actor_collab)
# print("Actor Importance Analysis:")
# for actor, metrics in actor_importance.items():
#     print(f"Actor: {actor}, Metrics: {metrics}")