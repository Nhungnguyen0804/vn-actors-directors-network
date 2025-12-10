import json
import pandas as pd
from networkx.readwrite import json_graph

INPUT_JSON = "data/cleaned_vn_bipartite_graph.json"
PERSON_CSV = "data/neo4j/person.csv"
FILM_CSV = "data/neo4j/film.csv"
ACTED_IN_CSV = "data/neo4j/acted_in.csv"

def flatten(d, prefix=""):
    """Flatten nested dicts: info.name → name"""
    out = {}
    for k, v in d.items():
        new_k = f"{prefix}{k}" if prefix == "" else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(flatten(v, new_k))
        else:
            out[new_k] = v
    return out


# --------------------------
# LOAD GRAPH JSON
# --------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

G = json_graph.node_link_graph(data, link="edges")


# --------------------------
# EXPORT NODES SPLIT BY TYPE
# --------------------------
person_rows = []
film_rows = []

for node_id, attrs in G.nodes(data=True):
    row = {"id": node_id}

    node_type = attrs.get("type", None)
    row["type"] = node_type

    # flatten nested attributes
    flat = flatten(attrs)
    row.update(flat)

    # push into correct list
    if node_type == "person":
        person_rows.append(row)
    elif node_type == "film":
        film_rows.append(row)

# save csvs
pd.DataFrame(person_rows).to_csv(PERSON_CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(film_rows).to_csv(FILM_CSV, index=False, encoding="utf-8-sig")

print(f"Exported {len(person_rows)} persons → {PERSON_CSV}")
print(f"Exported {len(film_rows)} films   → {FILM_CSV}")


# --------------------------
# EXPORT EDGES
# --------------------------
edge_rows = []
for u, v, attrs in G.edges(data=True):
    row = {"source": u, "target": v}
    row.update(attrs)
    edge_rows.append(row)

df_edges = pd.DataFrame(edge_rows)
df_edges.to_csv(ACTED_IN_CSV, index=False, encoding="utf-8-sig")

print(f"Exported {len(df_edges)} edges → {ACTED_IN_CSV}")
