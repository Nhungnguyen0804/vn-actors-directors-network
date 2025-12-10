import json
from networkx.readwrite import json_graph
import networkx as nx
def _load_graph_from_file(path):
    """
    Thử load theo vài định dạng JSON khác nhau để tránh KeyError: 'links'
    - ưu tiên node-link format (nodes + links)
    - nếu thất bại thử adjacency format
    - nếu vẫn thất bại, thử chuyển 'edges' -> 'links' rồi dùng node-link
    - (fallback) nếu là danh sách cạnh, xây dựng đồ thị từ danh sách cạnh (yêu cầu nx đã được import ở các cell khác)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Try node-link format (expects 'nodes' and 'links')
    try:
        return json_graph.node_link_graph(data)
    except KeyError:
        # Try adjacency format
        try:
            return json_graph.adjacency_graph(data)
        except Exception:
            # If the file used 'edges' key instead of 'links', convert it
            if isinstance(data, dict) and 'edges' in data:
                data2 = dict(data)
                data2['links'] = data2.pop('edges')
                return json_graph.node_link_graph(data2)
            # If the JSON is a plain list, assume it's an edge list
            if isinstance(data, list):
                try:
                    # Use nx from other cells (do not import networkx here to avoid duplicate imports)
                    G = nx.Graph()
                    G.add_edges_from(data)
                    return G
                except NameError:
                    # nx not defined; re-raise a clear error
                    raise RuntimeError("networkx (nx) is not available in the notebook namespace for fallback edge-list loading.")
            # If none matched, re-raise the original issue
            raise


def load_graphs():
    """Hàm tiện ích để đọc lại đồ thị từ file JSON"""
    # Đọc collaboration graph (về thứ tự để trả về G_actor_collab trước)
    G_collab_loaded = _load_graph_from_file('data/vn_film_collaboration_graph.json')
    
    # Đọc bipartite graph
    G_bipartite_loaded = _load_graph_from_file('data/vn_bipartite_graph.json')
    
    print("Đã load lại đồ thị từ JSON thành công!")
    return G_collab_loaded, G_bipartite_loaded
