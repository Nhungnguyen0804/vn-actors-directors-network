
import json
import pandas as pd

def load_jsonl_to_dict(path, key_field="name"):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print("Lỗi JSON tại dòng:", line[:200])
                print("Chi tiết:", e)
                continue

            key = obj.get(key_field)
            if key is not None:
                result[key] = obj

    return result


setfit_res_dict = load_jsonl_to_dict("data/re_setfit_res.jsonl", "entity")

print(len(setfit_res_dict))

print(setfit_res_dict['Ninh Dương Lan Ngọc'].keys())


PER_FILM_CSV = "data/neo4j/person_film.csv"

def append_relation_if_not_exists(source, target, role, character=""):
    """Append vào CSV nếu chưa có dòng đó."""
    

    df = pd.read_csv(PER_FILM_CSV, encoding="utf-8")

    # kiểm tra trùng lặp
    exists = (
        (df["source"] == source) &
        (df["target"] == target) &
        (df["role"] == role)
    ).any()

    if not exists:
        new_row = pd.DataFrame([{
            "source": source,
            "target": target,
            "role": role,
            "character": character
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(PER_FILM_CSV, index=False, encoding="utf-8")
        print(f"✔ Thêm: {source} -[{role}]-> {target}")
    return True


def process_relations_from_dict(input_dict):
    """Duyệt xử lý key relations."""
    total = 0
    for entity, data in input_dict.items():

        if "relations" not in data:
            continue

        for rel in data["relations"]:
            subject = rel.get("subject")
            relation = rel.get("relation")
            obj = rel.get("object")

        if not subject or not obj or not relation:
            continue

        # nếu DIRECTED → ghi vào CSV
        if relation == "DIRECTED":
            add = append_relation_if_not_exists(
                    source=subject,
                        target=obj,
                        role="DIRECTED",
                        character=""
                    )
            if add == True:  total +=1
        
        if relation == "ACTED_IN":
            add = append_relation_if_not_exists(
                        source=subject,
                        target=obj,
                        role="ACTED_IN",
                        character=""
                    )
            if add ==True: total +=1

    print(f"Tổng số dòng mới được thêm vào CSV: {total}")


process_relations_from_dict(setfit_res_dict)
