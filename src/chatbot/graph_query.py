from neo4j import GraphDatabase
import unicodedata
import logging

# ==================== CONFIGURATION ====================
URI = "neo4j+s://0538a688.databases.neo4j.io"
AUTH = ("neo4j", "askC5IvfBm2QXlzpKKn6gb9CEGxdouOCdTTKMhI6Si4")

# 🔇 TẮT CẢNH BÁO CỦA NEO4J DRIVER (Để output sạch đẹp)
logging.getLogger("neo4j").setLevel(logging.ERROR)

driver = GraphDatabase.driver(URI, auth=AUTH)

# ==================== UTILITY FUNCTIONS ====================

def close_driver():
    """Đóng kết nối Neo4j"""
    driver.close()

def normalize_input(text):
    """Chuẩn hóa input"""
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFC', text).strip()

# ==================== 1-HOP QUERIES ====================

def graph_query_movies_by_actor(actor_name, debug=False):
    """
    🔍 1-HOP: Tìm phim diễn viên tham gia
    """
    actor_name = normalize_input(actor_name)
    if debug: print(f"\n[QUERY] Movies by actor: {actor_name}")
    
    # Đã bỏ m.title, m.name để tránh warning
    query = """
    MATCH (p:PERSON)-[:ACTED_IN]->(m:FILM)
    WHERE toLower(p.info_name) CONTAINS toLower($name) 
       OR toLower(p.info_birth_name) CONTAINS toLower($name)
       OR toLower(p.id) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(m.info_name, m.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"name": actor_name}, debug)


def graph_query_actors_of_movie(movie_name, debug=False):
    """
    🔍 1-HOP: Tìm diễn viên trong phim
    """
    movie_name = normalize_input(movie_name)
    if debug: print(f"\n[QUERY] Actors in movie: {movie_name}")
    
    # Đã bỏ m.title để tránh warning
    query = """
    MATCH (m:FILM)<-[:ACTED_IN]-(p:PERSON)
    WHERE toLower(m.info_name) CONTAINS toLower($name) 
       OR toLower(m.id) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(p.info_name, p.info_birth_name, p.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"name": movie_name}, debug)


def graph_query_movies_by_director(director_name, debug=False):
    """
    🔍 1-HOP: Tìm phim của đạo diễn
    Relation: DIRECTED_IN
    """
    director_name = normalize_input(director_name)
    if debug: print(f"\n[QUERY] Movies by director: {director_name}")
    
    query = """
    MATCH (p:PERSON)-[:DIRECTED_IN]->(m:FILM)
    WHERE toLower(p.info_name) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(m.info_name, m.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"name": director_name}, debug)


def graph_query_director_of_movie(movie_name, debug=False):
    """
    🔍 1-HOP: Tìm đạo diễn
    """
    movie_name = normalize_input(movie_name)
    if debug: print(f"\n[QUERY] Director of movie: {movie_name}")
    
    query = """
    MATCH (m:FILM)<-[:DIRECTED_IN]-(p:PERSON)
    WHERE toLower(m.info_name) CONTAINS toLower($name)
       OR toLower(m.id) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(p.info_name, p.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"name": movie_name}, debug)


def graph_query_same_schoolmates(person_name, debug=False):
    """
    🔍 1-HOP: Bạn học
    """
    person_name = normalize_input(person_name)
    if debug: print(f"\n[QUERY] Schoolmates of: {person_name}")
    
    query = """
    MATCH (p1:PERSON)-[r:HAS_SAME_SCHOOL]-(p2:PERSON)
    WHERE toLower(p1.info_name) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(p2.info_name, p2.id) AS name, r.school AS school
    ORDER BY name
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, name=person_name):
            mate = record["name"]
            school = record["school"]
            txt = f"{mate} (Trường {school})" if school else mate
            results.append(txt)
            if debug: print(f"  - {txt}")
    
    if debug: print(f"✅ Found {len(results)} items")
    return results


def graph_query_same_location(person_name, debug=False):
    """
    🔍 1-HOP: Đồng hương
    """
    person_name = normalize_input(person_name)
    if debug: print(f"\n[QUERY] Same location as: {person_name}")
    
    query = """
    MATCH (p1:PERSON)-[:HAS_SAME_LOCATION]-(p2:PERSON)
    WHERE toLower(p1.info_name) CONTAINS toLower($name)
    RETURN DISTINCT COALESCE(p2.info_name, p2.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"name": person_name}, debug)


# ==================== 2-HOP QUERIES ====================

def graph_query_common_movies(actor1, actor2, debug=False):
    """
    🔗 2-HOP: Phim chung
    """
    actor1 = normalize_input(actor1)
    actor2 = normalize_input(actor2)
    if debug: print(f"\n[QUERY] Common movies: {actor1} & {actor2}")
    
    query = """
    MATCH (p1:PERSON)-[:ACTED_IN]->(m:FILM)<-[:ACTED_IN]-(p2:PERSON)
    WHERE toLower(p1.info_name) CONTAINS toLower($a1) 
      AND toLower(p2.info_name) CONTAINS toLower($a2)
    RETURN DISTINCT COALESCE(m.info_name, m.id) AS result
    ORDER BY result
    """
    return _execute_list_query(query, {"a1": actor1, "a2": actor2}, debug)


def graph_query_shortest_path(person1, person2, debug=False):
    """
    Path: Đường đi ngắn nhất
    """
    person1 = normalize_input(person1)
    person2 = normalize_input(person2)
    if debug: print(f"\n[QUERY] Shortest path: {person1} -> {person2}")
    
    query = """
    MATCH path = shortestPath((p1:PERSON)-[*]-(p2:PERSON))
    WHERE toLower(p1.info_name) CONTAINS toLower($p1) 
      AND toLower(p2.info_name) CONTAINS toLower($p2)
      AND p1 <> p2
    RETURN nodes(path) AS nodes, relationships(path) AS rels
    LIMIT 1
    """
    
    with driver.session() as session:
        record = session.run(query, p1=person1, p2=person2).single()
        
        if not record:
            if debug: print("   No path found")
            return None
            
        nodes = record["nodes"]
        rels = record["rels"]
        
        names = []
        for n in nodes:
            # Chỉ lấy info_name hoặc id, bỏ qua title/name để tránh warning
            name = n.get("info_name") or n.get("id") or "Unknown"
            names.append(name)

        descriptions = []
        for i, rel in enumerate(rels):
            start = names[i]
            end = names[i+1]
            r_type = rel.type
            
            if r_type == "ACTED_IN": desc = "đóng trong"
            elif r_type == "DIRECTED_IN": desc = "đạo diễn"
            elif r_type == "HAS_SAME_SCHOOL": desc = "học cùng trường"
            elif r_type == "HAS_SAME_LOCATION": desc = "cùng quê"
            else: desc = "liên quan"
            
            descriptions.append(f"{start} --[{desc}]--> {end}")

        result = {
            "path": names,
            "length": len(rels),
            "description": " -> ".join(descriptions)
        }
        if debug: print(f"  Path: {result['description']}")
        return result


def graph_query_node_info(node_name, debug=True):
    """
    🔍 INFO: Thông tin chi tiết node
    """
    node_name = normalize_input(node_name)
    if debug: print(f"\n[QUERY] Info for: {node_name}")

    query = """
    MATCH (n)
    WHERE (n:PERSON OR n:FILM) 
      AND (toLower(n.info_name) CONTAINS toLower($name) 
           OR toLower(n.id) CONTAINS toLower($name))
    RETURN n, labels(n) AS labels
    LIMIT 1
    """
    
    with driver.session() as session:
        record = session.run(query, name=node_name).single()
        
        if not record:
            if debug: print("  ❌ Node not found")
            return None
            
        node = record["n"]
        labels = record["labels"]
        
        node_type = "person" if "PERSON" in labels else "film" if "FILM" in labels else "unknown"
        
        props = dict(node)
        # Chỉ giữ lại info_ và id
        cleaned_props = {k: v for k, v in props.items() if k.startswith("info_") or k == "id"}
        
        # An toàn lấy tên
        display_name = props.get("info_name") or props.get("id")

        return {
            "name": display_name,
            "type": node_type,
            "properties": cleaned_props
        }


# ==================== MULTI-HOP / ADVANCED QUERIES ====================

def graph_query_collaborators_of_collaborators(actor_name, limit=10, debug=False):
    """
    🔀 MULTI-HOP (3-HOP): Bạn diễn của bạn diễn (Friends of friends)
    Logic: Actor A -> Movie 1 -> Actor B -> Movie 2 -> Actor C
    Mục đích: Gợi ý các diễn viên có "vòng tròn quan hệ" gần gũi.
    """
    actor_name = normalize_input(actor_name)
    if debug: print(f"\n[QUERY] Collaborators of collaborators for: {actor_name}")

    # Query này đi 4 bước: (Start)-[1]->(Movie)-[2]->(Middle)-[3]->(Movie)-[4]->(End)
    query = """
    MATCH (start:PERSON)-[:ACTED_IN]->(m1:FILM)<-[:ACTED_IN]-(middle:PERSON)-[:ACTED_IN]->(m2:FILM)<-[:ACTED_IN]-(end:PERSON)
    WHERE toLower(start.info_name) CONTAINS toLower($name)
      AND start <> end 
      AND start <> middle 
      AND middle <> end
    
    // Đếm số lần 'end' xuất hiện để tìm người có liên kết mạnh nhất
    WITH end, count(DISTINCT m2) as common_projects
    RETURN end.info_name AS result
    ORDER BY common_projects DESC
    LIMIT $limit
    """
    return _execute_list_query(query, {"name": actor_name, "limit": limit}, debug)


def graph_query_actors_in_movies_by_director(director_name, limit=15, debug=False):
    """
    🔀 MULTI-HOP (2-HOP Directed): Tìm diễn viên đóng phim của Đạo diễn X
    Logic: Director -> (DIRECTED_IN) -> Film <- (ACTED_IN) <- Actor
    """
    director_name = normalize_input(director_name)
    if debug: print(f"\n[QUERY] Actors in movies directed by: {director_name}")

    query = """
    MATCH (d:PERSON)-[:DIRECTED_IN]->(m:FILM)<-[:ACTED_IN]-(a:PERSON)
    WHERE toLower(d.info_name) CONTAINS toLower($name)
    RETURN DISTINCT a.info_name AS result
    LIMIT $limit
    """
    return _execute_list_query(query, {"name": director_name, "limit": limit}, debug)


def graph_query_related_people_2_hops(entity_name, limit=10, debug=False):
    """
    🔀 MULTI-HOP (Generic 2-Hop): Tìm tất cả người liên quan trong vòng 2 bước
    Bất kể là bạn học, đồng hương, hay bạn diễn.
    Logic: (Start) -[*1..2]- (End)
    """
    entity_name = normalize_input(entity_name)
    if debug: print(f"\n[QUERY] Related people (2-hops) for: {entity_name}")

    query = """
    MATCH (start:PERSON)-[*1..2]-(end:PERSON)
    WHERE toLower(start.info_name) CONTAINS toLower($name)
      AND start <> end
    RETURN DISTINCT end.info_name AS result
    LIMIT $limit
    """
    return _execute_list_query(query, {"name": entity_name, "limit": limit}, debug)
# ==================== INTERNAL HELPER ====================

def _execute_list_query(query, params, debug):
    """Hàm chạy query trả về list string (đỡ lặp code)"""
    results = []
    with driver.session() as session:
        res = session.run(query, **params)
        for record in res:
            val = record["result"]
            if val:
                results.append(val)
                if debug: print(f"  - {val}")
    
    if debug: print(f"✅ Found {len(results)} items")
    return results

# ==================== TEST EXECUTION ====================
if __name__ == "__main__":
    print("--- TESTING GRAPH QUERY (FINAL CLEAN) ---")
    graph_query_movies_by_actor("Trấn Thành", debug=True)
    graph_query_common_movies("Trấn Thành", "Ninh Dương Lan Ngọc", debug=True)
    close_driver()