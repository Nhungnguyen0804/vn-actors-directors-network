from neo4j import GraphDatabase
import networkx as nx

# Cấu hình kết nối Neo4j
URI = "neo4j+s://0538a688.databases.neo4j.io"
AUTH = ("neo4j", "askC5IvfBm2QXlzpKKn6gb9CEGxdouOCdTTKMhI6Si4")  # Thay password của bạn vào đây

def load_graph_from_neo4j():
    """
    Kết nối tới Neo4j, lấy toàn bộ Nodes và Relationships
    để dựng thành một NetworkX Graph.
    """
    driver = GraphDatabase.driver(URI, auth=AUTH)
    G = nx.MultiGraph() # Hoặc nx.DiGraph() nếu cần đồ thị có hướng

    print(" Đang load dữ liệu từ Neo4j...")

    with driver.session() as session:
        # ==================== 1. LOAD NODES ====================
        # Lấy ID, Labels và Properties của tất cả các node
        query_nodes = """
        MATCH (n)
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
        """
        result_nodes = session.run(query_nodes)
        
        count_nodes = 0
        for record in result_nodes:
            node_id = record["id"]
            labels = record["labels"]
            props = record["props"]
            
            # --- BẮT ĐẦU SỬA ĐỔI ---
            
            node_type = 'unknown'
            
            # Ưu tiên 1: Kiểm tra LABEL (Chuẩn Neo4j)
            # Chuyển hết về chữ thường để so sánh cho dễ
            labels_lower = [l.lower() for l in labels]
            
            if 'film' in labels_lower or 'movie' in labels_lower:
                node_type = 'film'
            elif 'person' in labels_lower or 'actor' in labels_lower or 'director' in labels_lower:
                node_type = 'person'
            
            # Ưu tiên 2: Nếu Label không có, kiểm tra PROPERTY (Dữ liệu import từ CSV hay bị dính cái này)
            if node_type == 'unknown':
                # Lấy thuộc tính 'type' hoặc 'Type' trong properties
                raw_type_prop = props.get('type', '').lower()
                
                if raw_type_prop in ['film', 'movie']:
                    node_type = 'film'
                elif raw_type_prop in ['person', 'actor', 'director']:
                    node_type = 'person'

            # --- KẾT THÚC SỬA ĐỔI ---

            # 2. Chuẩn bị thuộc tính cho Node
            node_attrs = props.copy()
            node_attrs['type'] = node_type # Ghi đè type chuẩn hóa vào
            
            # Xử lý tên hiển thị (như code cũ của bạn)
            display_name = props.get('info_name') or props.get('title') or props.get('full_name') or str(node_id)
            node_attrs['name'] = display_name

            G.add_node(node_id, **node_attrs)
            count_nodes += 1
        # ==================== 2. LOAD RELATIONSHIPS (Optional) ====================
        # Nếu việc Linking chỉ cần tên Node thì bước này không bắt buộc, 
        # nhưng nếu chatbot cần suy luận (Graph Traversal) thì CẦN.
        query_rels = """
        MATCH (a)-[r]->(b)
// Thêm 'HAS_SAME_SCHOOL' vào danh sách
WHERE type(r) IN ['ACTED_IN', 'DIRECTED', 'COLLABORATED', 'HAS_SAME_SCHOOL', 'HAS_SAME_LOCATION'] 
RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS rel_type, properties(r) AS props
    
        """
        result_rels = session.run(query_rels)
        
        count_rels = 0
        for record in result_rels:
            source = record["source"]
            target = record["target"]
            rel_type = record["rel_type"]
            props = record["props"]
            
            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, type=rel_type, **props)
                count_rels += 1

    driver.close()
    print(f" Đã load xong: {count_nodes} nodes, {count_rels} edges.")
    return G

if __name__ == "__main__":
    G = load_graph_from_neo4j()
    # Ví dụ: In thông tin một số node
   
    # in ra cac kieu canh quan he trong do thi
    edge_types = set(data['type'] for u, v, data in G.edges(data=True))
    print("Edge types in the graph:", edge_types)