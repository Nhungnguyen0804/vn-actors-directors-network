from neo4j import GraphDatabase
import unicodedata
import logging
import re
from typing import List, Dict, Any

URI = "neo4j+s://0538a688.databases.neo4j.io"
AUTH = ("neo4j", "askC5IvfBm2QXlzpKKn6gb9CEGxdouOCdTTKMhI6Si4")

logging.getLogger("neo4j").setLevel(logging.ERROR)

driver = GraphDatabase.driver(URI, auth=AUTH)


def close_driver():
    """Đóng kết nối Neo4j"""
    driver.close()




# ========================================================
# RELATIONSHIP REGISTRY
# ========================================================

RELATIONSHIPS = {
    # === PERSON -> FILM (EDGES) ===
    "PERSON_ACTED_IN_FILM": {
        "start": "PERSON",
        "rel": "ACTED_IN",
        "end": "FILM",
        "direction": "->",
        "return_field": "info_name",
        "description": "diễn viên đóng phim"
    },
    
    "PERSON_DIRECTED_FILM": {
        "start": "PERSON",
        "rel": "DIRECTED_IN",
        "end": "FILM",
        "direction": "->",
        "return_field": "info_name",
        "description": "đạo diễn phim"
    },
    
    # === PERSON -> PERSON (EDGES) ===
    "PERSON_SAME_SCHOOL": {
        "start": "PERSON",
        "rel": "HAS_SAME_SCHOOL",
        "end": "PERSON",
        "direction": "-",
        "return_field": "info_name",
        "description": "bạn học",
        "extra_fields": ["school"]
    },
    
    "PERSON_SAME_LOCATION": {
        "start": "PERSON",
        "rel": "HAS_SAME_LOCATION",
        "end": "PERSON",
        "direction": "-",
        "return_field": "info_name",
        "description": "đồng hương"
    },
    
    # === FILM -> PERSON (REVERSE EDGES) ===
    "FILM_HAS_ACTOR": {
        "start": "FILM",
        "rel": "ACTED_IN",
        "end": "PERSON",
        "direction": "<-",
        "return_field": "info_name",
        "description": "diễn viên trong phim"
    },
    
    "FILM_HAS_DIRECTOR": {
        "start": "FILM",
        "rel": "DIRECTED_IN",
        "end": "PERSON",
        "direction": "<-",
        "return_field": "info_name",
        "description": "đạo diễn của phim"
    },
    
    # === PERSON PROPERTIES ===
    "PERSON_SPOUSE": {
        "start": "PERSON",
        "type": "PROPERTY",
        "property": "info_spouse",
        "description": "vợ/chồng",
        "target_label": "PERSON"
    },
    
    "PERSON_BIRTH_DATE": {
        "start": "PERSON",
        "type": "PROPERTY",
        "property": "info_birth_date",
        "description": "năm sinh"
    },
    
    "PERSON_BIRTH_PLACE": {
        "start": "PERSON",
        "type": "PROPERTY",
        "property": "info_birth_place",
        "description": "quê quán"
    },
    
    "PERSON_OCCUPATION": {
        "start": "PERSON",
        "type": "PROPERTY",
        "property": "info_occupation",
        "description": "nghề nghiệp"
    },
    
    "PERSON_EDUCATION": {
        "start": "PERSON",
        "type": "PROPERTY",
        "property": "info_education",
        "description": "học vấn"
    },
    
    # === FILM PROPERTIES ===
    "FILM_RELEASE_DATE": {
        "start": "FILM",
        "type": "PROPERTY",
        "property": "infobox_released",
        "description": "năm công chiếu"
    },
    
    "FILM_REVENUE": {
        "start": "FILM",
        "type": "PROPERTY",
        "property": "infobox_gross",
        "description": "doanh thu"
    },
    
    "FILM_GENRE": {
        "start": "FILM",
        "type": "PROPERTY",
        "property": "infobox_genre",
        "description": "thể loại phim",
        "is_list": True
    }
}


# ========================================================
# CORE QUERY BUILDER
# ========================================================

def build_query_from_relationships(
    start_entity: str,
    start_label: str,
    relationships: List[str],
    limit: int = 20,
    debug: bool = False
) -> List[Any]:
    """
    Xây dựng và thực thi query từ danh sách relationships
    Hỗ trợ EDGES, PROPERTIES, và COLLECT
    """
    
    if debug:
        print(f"\n[DYNAMIC QUERY] Start: {start_entity} ({start_label})")
        print(f"  Chain: {' -> '.join(relationships)}")
    
    current_entity = start_entity
    current_label = start_label
    previous_results = None
    
    for i, rel_key in enumerate(relationships):
        if rel_key not in RELATIONSHIPS:
            if debug:
                print(f"  ❌ Unknown relationship: {rel_key}")
            return []
        
        rel_config = RELATIONSHIPS[rel_key]
        
        if debug:
            print(f"\n  [HOP {i+1}] {rel_key} ({rel_config.get('description', '')})")
        
        # === CASE 1: PROPERTY HOP ===
        if rel_config.get("type") == "PROPERTY":
            prop = rel_config["property"]
            is_list_property = rel_config.get("is_list", False)
            
            # CASE 1A: Collect property từ nhiều nodes
            if previous_results and is_list_property:
                if debug:
                    print(f"    → Collecting '{prop}' from {len(previous_results)} nodes")
                
                query = f"""
                MATCH (n:{current_label})
                WHERE toLower(n.info_name) IN [x IN $names | toLower(x)]
                   OR toLower(n.id) IN [x IN $names | toLower(x)]
                WITH COLLECT(DISTINCT n.{prop}) AS all_props
                UNWIND all_props AS prop_value
                WITH DISTINCT prop_value
                WHERE prop_value IS NOT NULL
                RETURN COLLECT(DISTINCT prop_value) AS result
                """
                
                with driver.session() as session:
                    record = session.run(query, names=previous_results).single()
                    
                    if not record or not record["result"]:
                        if debug:
                            print(f"    ❌ No values found")
                        return []
                    
                    # Flatten and clean
                    result_list = []
                    for item in record["result"]:
                        if isinstance(item, list):
                            result_list.extend(item)
                        elif isinstance(item, str) and ',' in item:
                            result_list.extend([x.strip() for x in item.split(',')])
                        else:
                            result_list.append(item)
                    
                    result_list = sorted(list(set(result_list)))
                    
                    if debug:
                        print(f"    ✓ Collected {len(result_list)} values: {result_list}")
                    
                    return result_list
            
            # CASE 1B: Single property lookup
            else:
                query = f"""
                MATCH (n:{current_label})
                WHERE toLower(n.info_name) CONTAINS toLower($name)
                   OR toLower(n.id) CONTAINS toLower($name)
                   OR toLower(n.info_birth_name) CONTAINS toLower($name)
                RETURN n.{prop} AS result
                ORDER BY 
                    CASE 
                        WHEN toLower(n.info_name) = toLower($name) THEN 1
                        WHEN toLower(n.id) = toLower($name) THEN 2
                        ELSE 3
                    END
                LIMIT 1
                """
                
                with driver.session() as session:
                    record = session.run(query, name=current_entity).single()
                    
                    if not record or not record["result"]:
                        if debug:
                            print(f"    ❌ Property {prop} not found")
                        return []
                    
                    result = record["result"]
                    
                    # Clean property value
                    if prop == "info_spouse" and "(" in result:
                        result = result.split("(")[0].strip()
                    
                    # Handle list properties
                    if is_list_property:
                        if isinstance(result, list):
                            result_list = result
                        elif isinstance(result, str) and ',' in result:
                            result_list = [x.strip() for x in result.split(',')]
                        else:
                            result_list = [result]
                        
                        if debug:
                            print(f"    ✓ Values: {result_list}")
                        return result_list
                    
                    if debug:
                        print(f"    ✓ Value: {result}")
                    
                    if i == len(relationships) - 1:
                        return [result]
                    
                    current_entity = result
                    current_label = rel_config.get("target_label", current_label)
                    previous_results = None
        
        # === CASE 2: EDGE HOP ===
        else:
            start_label_rel = rel_config["start"]
            end_label_rel = rel_config["end"]
            rel_type = rel_config["rel"]
            direction = rel_config["direction"]
            return_field = rel_config.get("return_field", "info_name")
            extra_fields = rel_config.get("extra_fields", [])
            
            # Build pattern
            if direction == "->":
                pattern = f"(n:{start_label_rel})-[r:{rel_type}]->(m:{end_label_rel})"
            elif direction == "<-":
                pattern = f"(n:{start_label_rel})<-[r:{rel_type}]-(m:{end_label_rel})"
            else:
                pattern = f"(n:{start_label_rel})-[r:{rel_type}]-(m:{end_label_rel})"
            
            # Query with fuzzy matching
            query = f"""
            MATCH {pattern}
            WHERE toLower(n.info_name) CONTAINS toLower($name)
               OR toLower(n.id) CONTAINS toLower($name)
               OR toLower(n.info_birth_name) CONTAINS toLower($name)
            WITH m, r,
                CASE 
                    WHEN toLower(n.info_name) = toLower($name) THEN 1
                    WHEN toLower(n.id) = toLower($name) THEN 2
                    WHEN toLower(n.info_birth_name) = toLower($name) THEN 3
                    ELSE 4
                END AS priority
            ORDER BY priority
            RETURN DISTINCT 
                COALESCE(m.{return_field}, m.id) AS result
                {', r.' + extra_fields[0] + ' AS extra' if extra_fields else ''}
            LIMIT $limit
            """
            
            results = []
            with driver.session() as session:
                for record in session.run(query, name=current_entity, limit=limit):
                    result_val = record["result"]
                    
                    if extra_fields and "extra" in record and record["extra"]:
                        result_val = f"{result_val} (Trường {record['extra']})"
                    
                    results.append(result_val)
                    
                    if debug:
                        print(f"    - {result_val}")
            
            if debug:
                print(f"    ✓ Found {len(results)} results")
            
            if i == len(relationships) - 1:
                return results
            
            if not results:
                if debug:
                    print(f"    ❌ No results found")
                return []
            
            # Lưu results cho hop tiếp
            previous_results = results
            
            # Nếu hop tiếp không phải list property, chỉ lấy result đầu
            if i + 1 < len(relationships):
                next_rel_config = RELATIONSHIPS[relationships[i + 1]]
                if not next_rel_config.get("is_list", False):
                    current_entity = results[0]
                    if "(" in current_entity:
                        current_entity = current_entity.split("(")[0].strip()
                    previous_results = None
            
            current_label = end_label_rel
    
    return []


# ========================================================
# FLEXIBLE QUERY
# ========================================================

def query_flexible(
    start_entity: str,
    start_label: str,
    relationship_chain: List[str],
    debug: bool = False
) -> List[Any]:
    """Query linh hoạt với chuỗi relationships bất kỳ"""
    return build_query_from_relationships(
        start_entity,
        start_label,
        relationship_chain,
        limit=50,
        debug=debug
    )


# ========================================================
# CONVENIENCE FUNCTIONS
# ========================================================

def graph_query_movies_by_actor(actor_name: str, debug: bool = False):
    """1-HOP: Phim của diễn viên"""
    return build_query_from_relationships(
        actor_name, "PERSON", ["PERSON_ACTED_IN_FILM"], debug=debug
    )


def graph_query_actors_of_movie(movie_name: str, debug: bool = False):
    """1-HOP: Diễn viên trong phim"""
    return build_query_from_relationships(
        movie_name, "FILM", ["FILM_HAS_ACTOR"], debug=debug
    )


def graph_query_movies_by_director(director_name: str, debug: bool = False):
    """1-HOP: Phim của đạo diễn"""
    return build_query_from_relationships(
        director_name, "PERSON", ["PERSON_DIRECTED_FILM"], debug=debug
    )


def graph_query_director_of_movie(movie_name: str, debug: bool = False):
    """1-HOP: Đạo diễn của phim"""
    return build_query_from_relationships(
        movie_name, "FILM", ["FILM_HAS_DIRECTOR"], debug=debug
    )


def graph_query_same_schoolmates(person_name: str, debug: bool = False):
    """1-HOP: Bạn học cùng trường"""
    return build_query_from_relationships(
        person_name, "PERSON", ["PERSON_SAME_SCHOOL"], debug=debug
    )


def graph_query_same_location(person_name: str, debug: bool = False):
    """1-HOP: Đồng hương"""
    return build_query_from_relationships(
        person_name, "PERSON", ["PERSON_SAME_LOCATION"], debug=debug
    )


def graph_query_spouse_movies(person_name: str, debug: bool = False):
    """2-HOP: Phim của vợ/chồng"""
    return query_flexible(
        person_name, "PERSON",
        ["PERSON_SPOUSE", "PERSON_ACTED_IN_FILM"],
        debug=debug
    )


def graph_query_common_movies(actor1: str, actor2: str, debug: bool = False):
    """2-HOP: Phim chung"""
    movies1 = set(graph_query_movies_by_actor(actor1, debug=False))
    movies2 = set(graph_query_movies_by_actor(actor2, debug=False))
    return sorted(list(movies1.intersection(movies2)))


# === CUSTOM QUERIES ===

def graph_query_director_of_actor_movies(actor_name: str, limit: int = 10, debug: bool = False):
    """2-HOP: Đạo diễn các phim diễn viên đóng"""
    query = """
    MATCH (actor:PERSON)-[:ACTED_IN]->(film:FILM)<-[:DIRECTED_IN]-(director:PERSON)
    WHERE toLower(actor.info_name) CONTAINS toLower($name)
       OR toLower(actor.info_birth_name) CONTAINS toLower($name)
    RETURN DISTINCT 
        COALESCE(director.info_name, director.id) AS director,
        COALESCE(film.info_name, film.id) AS film
    ORDER BY film
    LIMIT $limit
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, name=actor_name, limit=limit):
            results.append({
                "director": record["director"],
                "film": record["film"]
            })
    return results


def graph_query_actors_in_director_movies(director_name: str, limit: int = 20, debug: bool = False):
    """2-HOP: Diễn viên trong phim của đạo diễn"""

    query = """
    MATCH (director:PERSON)-[:DIRECTED_IN]->(film:FILM)<-[:ACTED_IN]-(actor:PERSON)
    WHERE toLower(director.info_name) CONTAINS toLower($name)
    RETURN DISTINCT 
        COALESCE(actor.info_name, actor.id) AS actor,
        COALESCE(film.info_name, film.id) AS film
    ORDER BY film
    LIMIT $limit
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, name=director_name, limit=limit):
            results.append({
                "actor": record["actor"],
                "film": record["film"]
            })
    return results


def graph_query_schoolmate_movies(person_name: str, debug: bool = False):
    """2-HOP: Phim của bạn học"""
    query = """
    MATCH (p1:PERSON)-[:HAS_SAME_SCHOOL]-(p2:PERSON)-[:ACTED_IN]->(film:FILM)
    WHERE toLower(p1.info_name) CONTAINS toLower($name)
    RETURN DISTINCT 
        COALESCE(p2.info_name, p2.id) AS schoolmate,
        COALESCE(film.info_name, film.id) AS film
    ORDER BY schoolmate, film
    LIMIT 20
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, name=person_name):
            results.append({
                "schoolmate": record["schoolmate"],
                "film": record["film"]
            })
    return results


def graph_query_common_directors(actor1: str, actor2: str, debug: bool = False):
    """3-HOP: Đạo diễn chung"""
    
    query = """
    MATCH (a1:PERSON)-[:ACTED_IN]->(f1:FILM)<-[:DIRECTED_IN]-(dir:PERSON)
         -[:DIRECTED_IN]->(f2:FILM)<-[:ACTED_IN]-(a2:PERSON)
    WHERE toLower(a1.info_name) CONTAINS toLower($a1)
      AND toLower(a2.info_name) CONTAINS toLower($a2)
      AND a1 <> a2
    RETURN DISTINCT 
        COALESCE(dir.info_name, dir.id) AS director,
        collect(DISTINCT COALESCE(f1.info_name, f1.id)) AS films_actor1,
        collect(DISTINCT COALESCE(f2.info_name, f2.id)) AS films_actor2
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, a1=actor1, a2=actor2):
            results.append({
                "director": record["director"],
                "films_with_actor1": record["films_actor1"],
                "films_with_actor2": record["films_actor2"]
            })
    return results


def graph_query_coactor_network(actor_name: str, depth: int = 2, limit: int = 15, debug: bool = False):
    """N-HOP: Mạng lưới bạn diễn"""
    query = f"""
    MATCH path = (start:PERSON)-[:ACTED_IN*1..{depth*2}]-(end:PERSON)
    WHERE toLower(start.info_name) CONTAINS toLower($name)
      AND start <> end
    WITH end, length(path) AS distance
    ORDER BY distance, end.info_name
    RETURN DISTINCT 
        COALESCE(end.info_name, end.id) AS coactor,
        MIN(distance) AS hops
    LIMIT $limit
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, name=actor_name, limit=limit):
            results.append({
                "name": record["coactor"],
                "distance": record["hops"] // 2
            })
    return results


def graph_query_actor_collaboration_history(actor1: str, actor2: str, debug: bool = False):
    """MULTI-HOP: Lịch sử hợp tác"""
    
    query = """
    MATCH (a1:PERSON)-[:ACTED_IN]->(film:FILM)<-[:ACTED_IN]-(a2:PERSON),
          (film)<-[:DIRECTED_IN]-(director:PERSON)
    WHERE toLower(a1.info_name) CONTAINS toLower($a1)
      AND toLower(a2.info_name) CONTAINS toLower($a2)
    RETURN DISTINCT
        COALESCE(film.info_name, film.id) AS film,
        COALESCE(director.info_name, director.id) AS director,
        film.info_release_date AS release_year
    ORDER BY release_year DESC
    """
    
    results = []
    with driver.session() as session:
        for record in session.run(query, a1=actor1, a2=actor2):
            results.append({
                "film": record["film"],
                "director": record["director"],
                "year": record["release_year"]
            })
    return results


def graph_query_shortest_path(person1: str, person2: str, debug: bool = False):
    """PATH: Đường đi ngắn nhất"""
   
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
            return None
        
        nodes = record["nodes"]
        rels = record["rels"]
        names = [n.get("info_name") or n.get("id") or "Unknown" for n in nodes]
        
        desc_map = {
            "ACTED_IN": "đóng trong",
            "DIRECTED_IN": "đạo diễn",
            "HAS_SAME_SCHOOL": "học cùng trường",
            "HAS_SAME_LOCATION": "cùng quê"
        }
        
        descriptions = []
        for i, rel in enumerate(rels):
            desc = desc_map.get(rel.type, "liên quan")
            descriptions.append(f"{names[i]} --[{desc}]--> {names[i+1]}")
        
        return {
            "path": names,
            "length": len(rels),
            "description": " -> ".join(descriptions)
        }


def graph_query_node_info(node_name: str, debug: bool = False):
    """INFO: Thông tin chi tiết node"""
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
            return None
        
        node, labels = record["n"], record["labels"]
        node_type = "person" if "PERSON" in labels else "film"
        props = dict(node)
        
        cleaned_props = {
            k: v for k, v in props.items() 
            if k.startswith("info_") or k == "id"
        }
        
        return {
            "name": props.get("info_name") or props.get("id"),
            "type": node_type,
            "properties": cleaned_props
        }


def graph_query_with_planner(question: str, entity_name: str, entity_type: str = "PERSON", debug: bool = False):
    """PLANNER: Tự động phát hiện property queries"""
    q_lower = question.lower()
    
    property_chain = []
    
    if any(word in q_lower for word in ["vợ", "chồng", "bà xã", "ông xã"]):
        property_chain.append("PERSON_SPOUSE")
    
    if any(word in q_lower for word in ["sinh năm", "năm sinh", "tuổi"]):
        property_chain.append("PERSON_BIRTH_DATE")
    
    if any(word in q_lower for word in ["quê", "ở đâu", "sinh ra"]):
        property_chain.append("PERSON_BIRTH_PLACE")
    
    if "năm" in q_lower and any(word in q_lower for word in ["công chiếu", "ra mắt"]):
        property_chain.append("FILM_RELEASE_DATE")
    
    if not property_chain:
        return None
    
    label = "FILM" if entity_type.lower() == "film" else "PERSON"
    results = build_query_from_relationships(entity_name, label, property_chain, debug=debug)
    
    return results[0] if results else None


# ========================================================
# TEST
# ========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING DYNAMIC GRAPH QUERY SYSTEM")
    print("="*60)
    
    entity = "Trấn Thành"
    
    print(f"\n[TEST 1] Movies by actor: {entity}")
    movies = graph_query_movies_by_actor(entity, debug=True)
    
    print(f"\n[TEST 2] Spouse's movies")
    spouse_movies = graph_query_spouse_movies(entity, debug=True)
    
    print(f"\n[TEST 3] Genres of actor's films")
    genres = query_flexible(entity, "PERSON", ["PERSON_ACTED_IN_FILM", "FILM_GENRE"], debug=True)
    print(f"  Result: {genres}")
    
    close_driver()
    print("\n✓ All tests completed!")