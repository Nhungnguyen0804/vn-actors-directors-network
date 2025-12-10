import json
from networkx.readwrite import json_graph
import networkx as nx
from pathlib import Path
import sys

# ==================== IMPORTS ====================
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from load_graph import load_graphs
except ImportError:
    try:
        from ..load_graph import load_graphs
    except ImportError:
        def load_graphs():
            return nx.Graph(), nx.Graph()


# ==================== UTILITY FUNCTIONS ====================

def normalize_text(text):
    """Chuẩn hóa text"""
    import unicodedata
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = text.lower().strip()
    return text


def find_node_by_name(G, name, fuzzy=False):
    """
    Tìm node trong graph bằng tên
    
    Args:
        G: NetworkX graph
        name: Tên node cần tìm
        fuzzy: Sử dụng fuzzy matching hay không
    
    Returns:
        str: Node ID (hoặc None nếu không tìm thấy)
    """
    if not G or not G.nodes():
        return None
    
    # Case 1: Exact match (direct)
    if name in G.nodes():
        return name
    
    # Case 2: Normalize match
    name_norm = normalize_text(name)
    
    for node_id in G.nodes():
        node_name = G.nodes[node_id].get('name', str(node_id))
        if normalize_text(node_name) == name_norm:
            return node_id
        
        # Also check node_id itself
        if normalize_text(str(node_id)) == name_norm:
            return node_id
    
    # Case 3: Substring match
    for node_id in G.nodes():
        node_name = G.nodes[node_id].get('name', str(node_id))
        if name_norm in normalize_text(node_name) or normalize_text(node_name) in name_norm:
            return node_id
    
    # Case 4: Fuzzy matching
    if fuzzy:
        try:
            from fuzzywuzzy import fuzz, process
            
            node_names = {}
            for node_id in G.nodes():
                node_name = G.nodes[node_id].get('name', str(node_id))
                node_names[node_name] = node_id
            
            matches = process.extract(name, list(node_names.keys()), scorer=fuzz.token_sort_ratio, limit=1)
            if matches and matches[0][1] >= 75:
                return node_names[matches[0][0]]
        except:
            pass
    
    return None


# ==================== BASIC GRAPH QUERY ====================

def graph_query_relation(G, start_node, relation_type, debug=False):
    """
    Tìm các node hàng xóm có quan hệ cụ thể với start_node
    
    Args:
        G: NetworkX graph
        start_node: Node ID hoặc tên
        relation_type: Loại quan hệ (ví dụ: 'actor', 'ACTED_IN')
        debug: In debug info
    
    Returns:
        list: Danh sách node IDs khớp
    """
    
    # Tìm node ID từ tên
    if start_node not in G:
        start_node = find_node_by_name(G, start_node, fuzzy=True)
        if not start_node:
            if debug:
                print(f"Node not found: {start_node}")
            return []
    
    result_nodes = []
    
    # Duyệt qua các node hàng xóm (neighbors)
    for neighbor in G.neighbors(start_node):
        if neighbor is None:
            continue
        
        try:
            edge_data = G.edges[start_node, neighbor]
        except:
            continue
        
        # Kiểm tra quan hệ
        rel = edge_data.get('role') or edge_data.get('relation') or edge_data.get('type')
        
        if rel == relation_type:
            result_nodes.append(neighbor)
            if debug:
                print(f" Found: {neighbor} (relation: {rel})")
    
    return result_nodes


def get_node_name(G, node_id):
    """
    Lấy tên hiển thị của node từ node ID
    
    Args:
        G: NetworkX graph
        node_id: Node ID
    
    Returns:
        str: Tên node
    """
    
    if node_id not in G.nodes():
        return str(node_id)
    
    node_data = G.nodes[node_id]
    
    # Thứ tự ưu tiên lấy tên
    name = (
        node_data.get('name') or
        node_data.get('title') or
        node_data.get('full_name') or
        str(node_id)
    )
    
    return name


def get_node_type(G, node_id):
    """
    Lấy loại node (person/film/org/location)
    
    Args:
        G: NetworkX graph
        node_id: Node ID
    
    Returns:
        str: Loại node
    """
    
    if node_id not in G.nodes():
        return None
    
    node_data = G.nodes[node_id]
    return node_data.get('type', None)


# ==================== ACTOR & MOVIE QUERIES ====================

def graph_query_movies_by_actor(G, actor_name, get_names=True, debug=False):
    """
    Tìm danh sách phim mà diễn viên tham gia
    
    Args:
        G: NetworkX graph (bipartite)
        actor_name: Tên diễn viên (node ID hoặc tên)
        get_names: Trả về tên phim hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách tên phim hoặc node IDs
    """
    
    # Tìm node ID của diễn viên
    actor_node = find_node_by_name(G, actor_name, fuzzy=True)
    if not actor_node:
        if debug:
            print(f"Actor not found: {actor_name}")
        return []
    
    if debug:
        print(f"\n[QUERY] Finding movies for actor: {get_node_name(G, actor_node)}")
    
    # Tìm các node phim có quan hệ 'actor' hoặc 'ACTED_IN'
    movie_nodes = []
    
    for relation_type in ['actor', 'ACTED_IN', 'film', 'ACTED_IN']:
        movies = graph_query_relation(G, actor_node, relation_type, debug=debug)
        movie_nodes.extend(movies)
    
    # Loại bỏ duplicate
    movie_nodes = list(set(movie_nodes))
    
    if not movie_nodes:
        if debug:
            print(f"No movies found for {actor_name}")
        return []
    
    if get_names:
        # Lấy tên phim từ node IDs
        movie_names = [get_node_name(G, movie_id) for movie_id in movie_nodes]
        return movie_names
    else:
        return movie_nodes


def graph_query_actors_of_movie(G, movie_name, get_names=True, debug=False):
    """
    Tìm danh sách diễn viên tham gia bộ phim
    
    Args:
        G: NetworkX graph (bipartite)
        movie_name: Tên phim (node ID hoặc tên)
        get_names: Trả về tên diễn viên hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách tên diễn viên hoặc node IDs
    """
    
    # Tìm node ID của phim
    movie_node = find_node_by_name(G, movie_name, fuzzy=True)
    if not movie_node:
        if debug:
            print(f"Movie not found: {movie_name}")
        return []
    
    if debug:
        print(f"\n[QUERY] Finding actors for movie: {get_node_name(G, movie_node)}")
    
    # Tìm các node diễn viên có quan hệ 'actor' hoặc 'ACTED_IN'
    actor_nodes = []
    
    for relation_type in ['actor', 'ACTED_IN', 'person', 'ACTED_IN']:
        actors = graph_query_relation(G, movie_node, relation_type, debug=debug)
        actor_nodes.extend(actors)
    
    # Loại bỏ duplicate
    actor_nodes = list(set(actor_nodes))
    
    if not actor_nodes:
        if debug:
            print(f"No actors found for {movie_name}")
        return []
    
    if get_names:
        # Lấy tên diễn viên từ node IDs
        actor_names = [get_node_name(G, actor_id) for actor_id in actor_nodes]
        return actor_names
    else:
        return actor_nodes


def graph_query_common_movies(G, actor1, actor2, debug=False):
    """
    Tìm phim mà 2 diễn viên cùng tham gia
    
    Args:
        G: NetworkX graph (bipartite)
        actor1: Tên diễn viên 1
        actor2: Tên diễn viên 2
        debug: In debug info
    
    Returns:
        list: Danh sách tên phim chung
    """
    
    if debug:
        print(f"\n[QUERY] Finding common movies for: {actor1} & {actor2}")
    start_node = find_node_by_name(G, actor1, fuzzy=True)
    end_node = find_node_by_name(G, actor2, fuzzy=True)
    
    edge_data = G.edges[start_node, end_node]
        
    # Tìm phim chung (intersection)
    common_movies = edge_data['films']
    if debug:
        print(f"Common movies: {len(common_movies)} found")
    
    # Chuyển node ID sang tên
    return [get_node_name(G, movie_id) for movie_id in common_movies]


def graph_query_collaborations(G, actor_name, get_names=True, debug=False):
    """
    Tìm những diễn viên khác mà actor_name đã hợp tác
    
    Args:
        G: NetworkX graph (collaboration graph)
        actor_name: Tên diễn viên
        get_names: Trả về tên hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách diễn viên hợp tác
    """
    
    # Tìm node ID
    actor_node = find_node_by_name(G, actor_name, fuzzy=True)
    if not actor_node:
        if debug:
            print(f" Actor not found: {actor_name}")
        return []
    
    if debug:
        print(f"\n[QUERY] Finding collaborations for: {get_node_name(G, actor_node)}")
    
    collaborators = []
    
    # Duyệt qua neighbors (collaboration graph)
    for neighbor in G.neighbors(actor_node):
        if neighbor is None:
            continue
        
        try:
            edge_data = G.edges[actor_node, neighbor]
            # Có thể kiểm tra relation nếu cần
        except:
            pass
        
        collaborators.append(neighbor)
    
    if get_names:
        collab_names = [get_node_name(G, c_id) for c_id in collaborators]
        return collab_names
    else:
        return collaborators


def graph_query_spouse(G, person_name, debug=False):
    """
    Tìm vợ/chồng của người
    
    Args:
        G: NetworkX graph
        person_name: Tên người
        debug: In debug info
    
    Returns:
        str hoặc None: Tên vợ/chồng
    """
    
    # Tìm node ID
    person_node = find_node_by_name(G, person_name, fuzzy=True)
    
    if person_node not in G.nodes():
        return None
    
    node_data = G.nodes[person_node]
    
    # Lấy thông tin vợ/chồng từ node attributes
    spouse = node_data['info'].get('spouse')
    
    if debug:
        print(f"\n[QUERY] Spouse of {get_node_name(G, person_node)}: {spouse}")
    
    return spouse


def graph_query_node_info(G, node_name, debug=True):
    """
    Lấy tất cả thông tin về một node
    
    Args:
        G: NetworkX graph
        node_name: Tên node hoặc node ID
        debug: In debug info
    
    Returns:
        dict: Thông tin node
    """
    
    # Tìm node ID
    node_id = find_node_by_name(G, node_name, fuzzy=True)
    
    if node_id not in G.nodes():
        return None
    
    node_data = G.nodes[node_id]
    
    result = {
        'node_id': node_id,
        'name': get_node_name(G, node_id),
        'type': get_node_type(G, node_id),
        'degree': G.degree(node_id),
        'attributes': node_data
    }
    
    return result


def graph_query_degree(G, node_name):
    """
    Lấy số lượng quan hệ (degree) của node
    
    Args:
        G: NetworkX graph
        node_name: Tên node hoặc node ID
    
    Returns:
        int: Số lượng neighbors
    """
    
    node_id = find_node_by_name(G, node_name, fuzzy=True)
    if not node_id or node_id not in G:
        return 0
    
    return G.degree(node_id)


# ==================== MULTI-HOP QUERY FUNCTIONS ====================

def graph_query_actor_via_movie(G, movie_name, exclude_actor=None, get_names=True, debug=False):
    """
    🔗 2-HOP: Tìm các diễn viên khác trong phim (trừ 1 diễn viên)
    
    Reasoning:
    1. Lấy tên phim
    2. Tìm tất cả diễn viên trong phim
    3. Loại bỏ exclude_actor
    4. Return danh sách còn lại
    
    Args:
        G: NetworkX graph (bipartite)
        movie_name: Tên phim
        exclude_actor: Tên diễn viên cần loại bỏ
        get_names: Trả về tên hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách diễn viên khác
    """
    
    if debug:
        print(f"\n[QUERY 2-HOP] Finding other actors in movie: {movie_name}")
        if exclude_actor:
            print(f"   (excluding: {exclude_actor})")
    
    # Step 1: Lấy tất cả diễn viên trong phim
    all_actors = graph_query_actors_of_movie(G, movie_name, get_names=False, debug=debug)
    
    if not all_actors:
        if debug:
            print(f"   No actors found in {movie_name}")
        return []
    
    # Step 2: Loại bỏ diễn viên ngoại lệ
    if exclude_actor:
        exclude_node = find_node_by_name(G, exclude_actor, fuzzy=True)
        other_actors = [a for a in all_actors if a != exclude_node]
    else:
        other_actors = all_actors
    
    if debug:
        print(f"   Found {len(other_actors)} other actors")
    
    if get_names:
        return [get_node_name(G, actor_id) for actor_id in other_actors]
    else:
        return other_actors


def graph_query_movie_via_actor(G, actor_name, exclude_movie=None, get_names=True, debug=False):
    """
     2-HOP: Tìm các phim khác của diễn viên (trừ 1 phim)
    
    Reasoning:
    1. Lấy tên diễn viên
    2. Tìm tất cả phim của diễn viên
    3. Loại bỏ exclude_movie
    4. Return danh sách còn lại
    
    Args:
        G: NetworkX graph (bipartite)
        actor_name: Tên diễn viên
        exclude_movie: Tên phim cần loại bỏ
        get_names: Trả về tên hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách phim khác
    """
    
    if debug:
        print(f"\n[QUERY 2-HOP] Finding other movies for actor: {actor_name}")
        if exclude_movie:
            print(f"   (excluding: {exclude_movie})")
    
    # Step 1: Lấy tất cả phim của diễn viên
    all_movies = graph_query_movies_by_actor(G, actor_name, get_names=False, debug=debug)
    
    if not all_movies:
        if debug:
            print(f"   No movies found for {actor_name}")
        return []
    
    # Step 2: Loại bỏ phim ngoại lệ
    if exclude_movie:
        exclude_node = find_node_by_name(G, exclude_movie, fuzzy=True)
        other_movies = [m for m in all_movies if m != exclude_node]
    else:
        other_movies = all_movies
    
    if debug:
        print(f"   Found {len(other_movies)} other movies")
    
    if get_names:
        return [get_node_name(G, movie_id) for movie_id in other_movies]
    else:
        return other_movies


def graph_query_actor_via_collaboration(G, actor2_name, exclude_actor=None, get_names=True, debug=False):
    """
    2-HOP: Tìm các diễn viên khác hợp tác với 1 diễn viên (trừ 1 diễn viên)
    
    Reasoning:
    1. Lấy tên diễn viên actor2
    2. Tìm tất cả diễn viên hợp tác với actor2
    3. Loại bỏ exclude_actor
    4. Return danh sách còn lại
    
    Args:
        G: NetworkX graph (collaboration)
        actor2_name: Tên diễn viên chính (người ta muốn tìm hợp tác của người này)
        exclude_actor: Tên diễn viên cần loại bỏ
        get_names: Trả về tên hay node ID
        debug: In debug info
    
    Returns:
        list: Danh sách diễn viên hợp tác khác
    """
    
    if debug:
        print(f"\n[QUERY 2-HOP] Finding collaborators of: {actor2_name}")
        if exclude_actor:
            print(f"   (excluding: {exclude_actor})")
    
    # Step 1: Lấy tất cả diễn viên hợp tác với actor2
    all_collaborators = graph_query_collaborations(G, actor2_name, get_names=False, debug=debug)
    
    if not all_collaborators:
        if debug:
            print(f"   No collaborators found for {actor2_name}")
        return []
    
    # Step 2: Loại bỏ diễn viên ngoại lệ
    if exclude_actor:
        exclude_node = find_node_by_name(G, exclude_actor, fuzzy=True)
        other_collaborators = [a for a in all_collaborators if a != exclude_node]
    else:
        other_collaborators = all_collaborators
    
    if debug:
        print(f"   Found {len(other_collaborators)} other collaborators")
    
    if get_names:
        return [get_node_name(G, actor_id) for actor_id in other_collaborators]
    else:
        return other_collaborators


def graph_query_indirect_collaboration(G_collab, actor1_name, actor2_name, debug=False):
    """
     3-HOP: Tìm diễn viên "cầu nối" giữa 2 diễn viên
    
    Reasoning:
    1. Lấy tên diễn viên 1
    2. Lấy tất cả diễn viên hợp tác với actor1
    3. Lấy tất cả diễn viên hợp tác với actor2
    4. Tìm giao tập (intersection) = cầu nối
    5. Return danh sách cầu nối
    
    Args:
        G_collab: NetworkX graph (collaboration)
        actor1_name: Tên diễn viên 1
        actor2_name: Tên diễn viên 2
        debug: In debug info
    
    Returns:
        list: Danh sách diễn viên cầu nối
    """
    
    if debug:
        print(f"\n[QUERY 3-HOP] Finding bridge actors between: {actor1_name} & {actor2_name}")
    
    # Step 1: Tìm collaborators của actor1
    actor1_collaborators = set(graph_query_collaborations(
        G_collab, actor1_name, get_names=False, debug=debug
    ) or [])
    
    if not actor1_collaborators:
        if debug:
            print(f"   No collaborators found for {actor1_name}")
        return []
    
    # Step 2: Tìm collaborators của actor2
    actor2_collaborators = set(graph_query_collaborations(
        G_collab, actor2_name, get_names=False, debug=debug
    ) or [])
    
    if not actor2_collaborators:
        if debug:
            print(f"   No collaborators found for {actor2_name}")
        return []
    
    # Step 3: Tìm giao tập (cầu nối)
    bridge_actors = actor1_collaborators.intersection(actor2_collaborators)
    
    if debug:
        print(f"   Found {len(bridge_actors)} bridge actors")
    
    return [get_node_name(G_collab, actor_id) for actor_id in bridge_actors]


def graph_query_movie_chain(G_bipartite, actor_name, movie_name, debug=False):
    """
     3-HOP: Tìm phim mà actor đóng chung với các diễn viên từ movie cho trước
    
    Reasoning:
    1. Lấy tên diễn viên + tên phim
    2. Tìm tất cả diễn viên trong movie
    3. Tìm tất cả phim của actor
    4. Với mỗi phim của actor, kiểm tra xem có diễn viên từ movie không
    5. Return danh sách phim chung
    
    Args:
        G_bipartite: NetworkX graph (bipartite)
        actor_name: Tên diễn viên
        movie_name: Tên phim tham chiếu
        debug: In debug info
    
    Returns:
        list: Danh sách phim chung
    """
    
    if debug:
        print(f"\n[QUERY 3-HOP] Finding movie chain:")
        print(f"   Actor: {actor_name}")
        print(f"   Reference movie: {movie_name}")
    
    # Step 1: Lấy diễn viên trong movie tham chiếu
    ref_movie_actors = set(graph_query_actors_of_movie(
        G_bipartite, movie_name, get_names=False, debug=debug
    ) or [])
    
    if not ref_movie_actors:
        if debug:
            print(f"   No actors found in {movie_name}")
        return []
    
    if debug:
        print(f"   Found {len(ref_movie_actors)} actors in reference movie")
    
    # Step 2: Lấy tất cả phim của actor
    actor_movies = graph_query_movies_by_actor(
        G_bipartite, actor_name, get_names=False, debug=debug
    ) or []
    
    if not actor_movies:
        if debug:
            print(f"   No movies found for {actor_name}")
        return []
    
    if debug:
        print(f"   Found {len(actor_movies)} movies for {actor_name}")
    
    # Step 3: Cho mỗi phim của actor, kiểm tra có diễn viên từ ref_movie không
    common_films = []
    
    for film_id in actor_movies:
        try:
            film_actors = set(graph_query_actors_of_movie(
                G_bipartite, film_id, get_names=False, debug=False
            ) or [])
            
            # Kiểm tra intersection
            if film_actors.intersection(ref_movie_actors):
                common_films.append(film_id)
                if debug:
                    print(f"   ✓ {get_node_name(G_bipartite, film_id)} has actors from {movie_name}")
        except:
            continue
    
    if debug:
        print(f"   Found {len(common_films)} common films")
    
    return [get_node_name(G_bipartite, film_id) for film_id in common_films]


# ==================== ROUTING FOR MULTI-HOP ====================

def route_multihop_query(intent_type, G_bipartite, G_collab, entities_dict, question, debug=False):
    """
     Điều hướng các query multi-hop
    
    Args:
        intent_type: Loại intent (actor_via_movie, movie_via_actor, ...)
        G_bipartite: Bipartite graph
        G_collab: Collaboration graph
        entities_dict: Dict {entity_name: node_id}
        question: Câu hỏi gốc
        debug: In debug info
    
    Returns:
        dict: Kết quả query
    """
    
    entity_names = list(entities_dict.keys())
    entity_ids = list(entities_dict.values())
    
    if intent_type == 'actor_via_movie':
        # Cần: 1 phim + 1 diễn viên
        if len(entity_ids) >= 2:
            movie_ref = entity_names[0]
            actor_ref = entity_names[1]
            try:
                others = graph_query_actor_via_movie(
                    G_bipartite, movie_ref, exclude_actor=actor_ref, debug=debug
                )
                return {
                    'status': 'success',
                    'data': others,
                    'message': f"Diễn viên khác trong phim {movie_ref} (ngoài {actor_ref})"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'movie_via_actor':
        # Cần: 1 diễn viên + 1 phim
        if len(entity_ids) >= 2:
            actor_ref = entity_names[0]
            movie_ref = entity_names[1]
            try:
                others = graph_query_movie_via_actor(
                    G_bipartite, actor_ref, exclude_movie=movie_ref, debug=debug
                )
                return {
                    'status': 'success',
                    'data': others,
                    'message': f"Phim khác của {actor_ref} (ngoài {movie_ref})"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'actor_via_collaboration':
        # Cần: 2 diễn viên
        if len(entity_ids) >= 2:
            actor1 = entity_names[0]
            actor2 = entity_names[1]
            try:
                others = graph_query_actor_via_collaboration(
                    G_collab, actor2, exclude_actor=actor1, debug=debug
                )
                return {
                    'status': 'success',
                    'data': others,
                    'message': f"Diễn viên khác hợp tác với {actor2} (ngoài {actor1})"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'indirect_collaboration':
        # Cần: 2 diễn viên
        if len(entity_ids) >= 2:
            actor1 = entity_names[0]
            actor2 = entity_names[1]
            try:
                bridges = graph_query_indirect_collaboration(
                    G_collab, actor1, actor2, debug=debug
                )
                return {
                    'status': 'success',
                    'data': bridges,
                    'message': f"Diễn viên cầu nối giữa {actor1} và {actor2}"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    elif intent_type == 'movie_chain':
        # Cần: 1 diễn viên + 1 phim
        if len(entity_ids) >= 2:
            actor_ref = entity_names[0]
            movie_ref = entity_names[1]
            try:
                common = graph_query_movie_chain(
                    G_bipartite, actor_ref, movie_ref, debug=debug
                )
                return {
                    'status': 'success',
                    'data': common,
                    'message': f"Phim chung của {actor_ref} với diễn viên từ {movie_ref}"
                }
            except Exception as e:
                return {'status': 'error', 'data': None, 'message': str(e)}
    
    else:
        return {
            'status': 'error',
            'data': None,
            'message': f'Intent "{intent_type}" không được hỗ trợ'
        }


# ==================== TEST MULTI-HOP QUERIES ====================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("MULTI-HOP GRAPH QUERY TESTS")
    print("="*100)
    
    # Load graphs
    G_actor_collab, G_bipartite = load_graphs()
    
    print(f"\n[INIT] Graphs loaded")
    print(f"   Bipartite: {len(G_bipartite.nodes())} nodes, {len(G_bipartite.edges())} edges")
    print(f"   Collaboration: {len(G_actor_collab.nodes())} nodes, {len(G_actor_collab.edges())} edges")
    
    # Test 1: actor_via_movie
    print("\n" + "-"*100)
    print("Test 1: Actors via Movie (2-HOP)")
    print("-"*100)
    try:
        movie = "Bố Già"
        exclude_actor = "Trấn Thành"
        others = graph_query_actor_via_movie(G_bipartite, movie, exclude_actor, debug=True)
        print(f"\nOther actors in '{movie}' (excluding '{exclude_actor}'):")
        for actor in others[:5]:
            print(f"   - {actor}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: movie_via_actor
    print("\n" + "-"*100)
    print("Test 2: Movies via Actor (2-HOP)")
    print("-"*100)
    try:
        actor = "Trấn Thành"
        exclude_movie = "Bố Già"
        others = graph_query_movie_via_actor(G_bipartite, actor, exclude_movie, debug=True)
        print(f"\nOther movies of '{actor}' (excluding '{exclude_movie}'):")
        for movie in others[:5]:
            print(f"   - {movie}")
    except Exception as e:
        print(f" Error: {e}")
    
    # Test 3: actor_via_collaboration
    print("\n" + "-"*100)
    print("Test 3: Actors via Collaboration (2-HOP)")
    print("-"*100)
    try:
        actor1 = "Trấn Thành"
        actor2 = "Hari Won"
        others = graph_query_actor_via_collaboration(G_actor_collab, actor2, exclude_actor=actor1, debug=True)
        print(f"\nCollaborators of '{actor2}' (excluding '{actor1}'):")
        for actor in others[:5]:
            print(f"   - {actor}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: indirect_collaboration
    print("\n" + "-"*100)
    print("Test 4: Indirect Collaboration (3-HOP)")
    print("-"*100)
    try:
        actor1 = "Trấn Thành"
        actor2 = "Hari Won"
        bridges = graph_query_indirect_collaboration(G_actor_collab, actor1, actor2, debug=True)
        print(f"\nBridge actors between '{actor1}' and '{actor2}':")
        for actor in bridges[:5]:
            print(f"   - {actor}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: movie_chain
    print("\n" + "-"*100)
    print("Test 5: Movie Chain (3-HOP)")
    print("-"*100)
    try:
        actor = "Trấn Thành"
        movie = "Bố Già"
        common = graph_query_movie_chain(G_bipartite, actor, movie, debug=True)
        print(f"\nMovies of '{actor}' with actors from '{movie}':")
        for film in common[:5]:
            print(f"   - {film}")
    except Exception as e:
        print(f" Error: {e}")
    
    print("\n" + "="*100 + "\n")