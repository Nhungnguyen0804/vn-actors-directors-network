import json
import pandas as pd
from networkx.readwrite import json_graph

BIPARTITE_JSON = "data/cleaned_vn_bipartite_graph.json"
COLLAB_JSON = "data/cleaned_vn_film_collaboration_graph.json"

PERSON_CSV = "data/neo4j/person.csv"
FILM_CSV = "data/neo4j/film.csv"
COLLAB_CSV = "data/neo4j/collab.csv"
SCHOOL_CSV = "data/neo4j/same_school.csv"
LOCATION_CSV = "data/neo4j/same_location.csv"

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        new_k = f"{prefix}{k}" if prefix == "" else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(flatten(v, new_k))
        else:
            out[new_k] = v
    return out

# =======================================================
# 1. LOAD BIPARTITE GRAPH (NODES CHUẨN)
# =======================================================
with open(BIPARTITE_JSON, "r", encoding="utf-8") as f:
    data_bi = json.load(f)

G_bi = json_graph.node_link_graph(data_bi, link="edges")

person_rows = []
film_rows = []
existing_persons = set()

# Tạo bảng person & film ban đầu
for node_id, attrs in G_bi.nodes(data=True):
    row = {"id": node_id}
    node_type = attrs.get("type")
    flat = flatten(attrs)
    row.update(flat)

    if node_type == "person":
        person_rows.append(row)
        existing_persons.add(node_id)

    elif node_type == "film":
        film_rows.append(row)

# =======================================================
# 2. LOAD COLLAB GRAPH — NODE CÓ THỂ KHÁC HOÀN TOÀN
# =======================================================
with open(COLLAB_JSON, "r", encoding="utf-8") as f:
    data_collab = json.load(f)

G_col = json_graph.node_link_graph(data_collab, link="edges")

# =======================================================
# 3. TẠO collab.csv + BỔ SUNG NODE NẾU THIẾU
# =======================================================
collab_rows = []
school_rows = []
location_rows = []

for u, v, attrs in G_col.edges(data=True):

    # ---- check missing persons ----
    for p in [u, v]:
        if p not in existing_persons:
            # If node exists in collab graph, extract info
            attr = G_col.nodes[p]
            flat = flatten(attr)
            flat["id"] = p
            flat["type"] = attr.get("type", "person")  # fallback = person

            person_rows.append(flat)
            existing_persons.add(p)

    # ---- export collab row ----
    row = {
        "source": u,
        "target": v,
        "film_count": attrs.get("film_count"),
        "films": ", ".join(attrs.get("films", [])),
        "collaboration_types": ", ".join(attrs.get("collaboration_types", [])),
        # "same_school": attrs.get("same_school"),
        # "school": attrs.get("school"),
        # "same_location": attrs.get("same_location"),
        # "location": attrs.get("location"),
        "weight": attrs.get("weight"),
    }
    collab_rows.append(row)

    # --- export same_school ---
    if attrs.get("same_school") is True and attrs.get("school"):
        school_rows.append({
            "source": u,
            "target": v,
            "school": attrs["school"]
        })

    # --- export same_location ---
    if attrs.get("same_location") is True and attrs.get("location"):
        location_rows.append({
            "source": u,
            "target": v,
            "location": attrs["location"]
        })


# =======================================================
# 4. SAVE UPDATED CSVs
# =======================================================
pd.DataFrame(person_rows).to_csv(PERSON_CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(film_rows).to_csv(FILM_CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(collab_rows).to_csv(COLLAB_CSV, index=False, encoding="utf-8-sig")

print("Updated person.csv:", len(person_rows))
print("Updated film.csv:", len(film_rows))
print("Exported collab.csv:", len(collab_rows))

pd.DataFrame(school_rows).to_csv(SCHOOL_CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(location_rows).to_csv(LOCATION_CSV, index=False, encoding="utf-8-sig")

print("same_school:", len(school_rows))
print("same_location:", len(location_rows))