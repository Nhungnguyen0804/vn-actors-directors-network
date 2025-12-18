from src.chatbot.graph_query import (
        build_query_from_relationships, query_flexible, graph_query_node_info,
        graph_query_common_movies, graph_query_director_of_actor_movies,
        graph_query_actors_in_director_movies, graph_query_schoolmate_movies,
        graph_query_common_directors, graph_query_shortest_path,
        close_driver
    )

INTENT_PATTERNS = {
    # === INTERSECTION QUERIES (HIGHEST PRIORITY FOR 2 ENTITIES) ===
    "get_common_movies": {
        "patterns": [
            r'\b(dong|tham\s+gia)\s+(chung|cung)\b',
            r'\bphim\s+(chung|cung)\b',
            r'\bvoi\b.+\b(dong|phim)\b',  # "X với Y đóng phim"
            r'\bva\b.+\b(dong|phim)\b',   # "X và Y đóng phim"
        ],
        "func": "graph_query_common_movies", "label": "PERSON", "needs_2_entities": True
    },
    "get_common_directors": {
        "patterns": [r'\b(dao\s+dien)\s+(chung|cung)\b'],
        "func": "graph_query_common_directors", "label": "PERSON", "needs_2_entities": True
    },
    "get_relationship_path": {
        "patterns": [
            r'\b(quan\s+he|lien\s+quan|ket\s+noi)\b',
            r'\b(co\s+lien\s+he|co\s+quan\s+he)\b',
        ],
        "func": "graph_query_shortest_path", "label": "PERSON", "needs_2_entities": True
    },
     "get_collaborated_network": {
    "patterns": [
        r'\b(mang\s+luoi|ket\s+noi).+(hop\s+tac)\b',
        r'\b(ban\s+cua\s+ban|hop\s+tac.*hop\s+tac)\b',
    ],
    "rels": ['PERSON_COLLABORATED', 'PERSON_COLLABORATED'],
    "label": "PERSON"
},
    # === SPOUSE QUERIES (MULTI-HOP, ONLY FOR 1 ENTITY) ===
    "get_spouse_info": {
        "patterns": [
            r'\b(vo|chong)\s+(cua\s+)?[\w\s]+\s+(la\s+ai|thong\s+tin)\b',
            r'\b(vo|chong).+(la\s+ai)\b',
        ],
        "rels": ['PERSON_SPOUSE'], "label": "PERSON", "max_entities": 1
    },
    "get_spouse_birthdate": {
        "patterns": [r'\b(vo|chong).+(sinh|nam\s+sinh)\b'],
        "rels": ['PERSON_SPOUSE', 'PERSON_BIRTH_DATE'], "label": "PERSON", "max_entities": 1
    },
    "get_spouse_birthplace": {
        "patterns": [r'\b(vo|chong).+(que|o\s+dau)\b'],
        "rels": ['PERSON_SPOUSE', 'PERSON_BIRTH_PLACE'], "label": "PERSON", "max_entities": 1
    },
    "get_spouse_movies": {
        "patterns": [r'\b(vo|chong).+(dong|phim|tham\s+gia)\b'],
        "rels": ['PERSON_SPOUSE', 'PERSON_ACTED_IN_FILM'], "label": "PERSON", "max_entities": 1
    },
    
    # === MULTI-HOP ===
    
    "get_actor_film_genres": {
        "patterns": [
            r'\b(the\s+loai).+(phim).+(dong|tham\s+gia)\b',
            r'\b(dong|tham\s+gia).+(the\s+loai)\b',
        ],
        "rels": ['PERSON_ACTED_IN_FILM', 'FILM_GENRE'], "label": "PERSON"
    },
    "get_director_film_genres": {
        "patterns": [r'\b(dao\s+dien).+(the\s+loai)\b'],
        "rels": ['PERSON_DIRECTED_FILM', 'FILM_GENRE'], "label": "PERSON"
    },
    "get_director_of_actor_movies": {
        "patterns": [r'\b(dao\s+dien).+(phim).+(dong|tham\s+gia)\b'],
        "func": "graph_query_director_of_actor_movies", "label": "PERSON"
    },
    "get_actors_in_director_movies": {
        "patterns": [r'\b(dien\s+vien).+(phim).+(dao\s+dien)\b'],
        "func": "graph_query_actors_in_director_movies", "label": "PERSON"
    },
    "get_schoolmate_movies": {
        "patterns": [r'\b(ban\s+hoc).+(dong|phim)\b'],
        "func": "graph_query_schoolmate_movies", "label": "PERSON"
    },
   
    
    # === BASIC 1-HOP ===
    "get_director_of_movie": {
        "patterns": [
            r'\b(ai|nguoi\s+nao)\s+(dao\s+dien)\b',
            r'\b(dao\s+dien)\b(?!.+(phim).+(dong))',
        ],
        "rels": ['FILM_HAS_DIRECTOR'], "label": "FILM"
    },
    "get_movies_by_director": {
        "patterns": [r'\b(dao\s+dien).+(phim)\s+(nao|gi)\b'],
        "rels": ['PERSON_DIRECTED_FILM'], "label": "PERSON"
    },
    "get_movies_by_actor": {
        "patterns": [
            r'\b(dong|tham\s+gia)\s+(phim)\b(?!.+(chung|cung|voi|va))',
            r'\bphim\s+(nao|gi)\b(?!.+(chung|cung))',
        ],
        "rels": ['PERSON_ACTED_IN_FILM'], "label": "PERSON"
    },
    "get_actors_of_movie": {
        "patterns": [
            r'\b(ai|nguoi\s+nao)\s+(dong|tham\s+gia)\b',
            r'\b(dien\s+vien|cast)\b',
        ],
        "rels": ['FILM_HAS_ACTOR'], "label": "FILM"
    },
    "get_film_genre": {
        "patterns": [
            r'\b(the\s+loai).+(phim)\b(?!.+(dong|dao\s+dien))',
            r'\bphim.+(the\s+loai|loai|thuoc)\b',
        ],
        "rels": ['FILM_GENRE'], "label": "FILM"
    },
    "get_same_school": {
        "patterns": [r'\b(cung\s+truong|hoc\s+cung)\b'],
        "rels": ['PERSON_SAME_SCHOOL'], "label": "PERSON"
    },
    "get_same_location":{ 
        "patterns":[r'\b(cung|que)\s+(que|huong|noi)\b'],
        "rels": ['PERSON_SAME_LOCATION'], "label": "PERSON",
    },
    "get_birthdate": {
        "patterns": [r'\b(sinh|nam\s+sinh|tuoi)\b(?!.+(vo|chong))'],
        "rels": ['PERSON_BIRTH_DATE'], "label": "PERSON"
    },
    "get_birthplace": {
        "patterns": [r'\b(que|noi\s+sinh|o\s+dau)\b(?!.+(vo|chong))'],
        "rels": ['PERSON_BIRTH_PLACE'], "label": "PERSON"
    },
   
    "get_general_info": {
        "patterns": [r'\b(la\s+ai|thong\s+tin)\b(?!.+(vo|chong))'],
        "func": "graph_query_node_info", "label": "PERSON"
    },
}

# Function mapping
FUNC_MAP = {
    "graph_query_director_of_actor_movies": graph_query_director_of_actor_movies,
    "graph_query_actors_in_director_movies": graph_query_actors_in_director_movies,
    "graph_query_schoolmate_movies": graph_query_schoolmate_movies,
    "graph_query_common_movies": graph_query_common_movies,
    "graph_query_common_directors": graph_query_common_directors,
    "graph_query_node_info": graph_query_node_info,
    "graph_query_shortest_path": graph_query_shortest_path 
}
