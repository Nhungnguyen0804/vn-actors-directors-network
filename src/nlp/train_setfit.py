import re
from src.nlp.text_utils import split_text_into_sentences,normalize_entity_name
from src.nlp.ner import person_list, film_list, wiki_enrich, run_combine_ner
import json
from src.nlp.template_data_generator import generate_text_samples_from_triple
from setfit import Trainer
import random
from collections import defaultdict
def load_triples(path="data/triples_for_RE.json"):
    with open(path, "r", encoding="utf8") as f:
        triples = json.load(f)
    return triples

# 4) BUILD TRAIN SAMPLES (dùng triples đã detect từ graph)
# -------------------------
def build_train_samples_from_triples(triples, n_per_triple=4, mask_entities=False):
    """
    Input:
      triples: list of (s, r, o) tuples
      n_per_triple: số câu sinh cho mỗi triple (recommend 3..5)
      mask_entities: bool
    Output:
      samples: list of {"text":..., "label":...}
    """
    samples = []
    for s, r, o in triples:
        try:
            texts = generate_text_samples_from_triple(s, r, o, n_samples=n_per_triple, mask_entities=mask_entities)
        except KeyError as e:
            # relation không có template -> bỏ qua 
            print("Skip triple (no template):", (s, r, o))
            continue

        for txt in texts:
            samples.append({"text": txt, "label": r})
    return samples

def build_train_samples_from_triples(
        triples, 
        n_per_triple=3, 
        mask_entities=False,
        deduplicate=True,
        return_metadata=False
    ):
    """
    Xây tập huấn luyện từ triples.
    - Hỗ trợ kiểm tra trùng lặp
    - Validate output
    - Log chi tiết lỗi
    """

    samples = []
    seen = set()

    for idx, (s, r, o) in enumerate(triples):
        try:
            texts = generate_text_samples_from_triple(
                s, r, o, 
                n_samples=n_per_triple, 
                mask_entities=mask_entities
            )

        except KeyError:
            print(f"Skip triple (no template): {(s, r, o)}")
            continue
        except Exception as e:
            print(f"Error generating for {s}-{r}-{o}: {e}")
            continue

        if not texts:
            print(f"Empty output from template: {(s, r, o)}")
            continue

        for txt in texts:
            if not txt or not isinstance(txt, str):
                continue

            if deduplicate:
                key = (txt, r)
                if key in seen:
                    continue
                seen.add(key)

            item = {"text": txt, "label": r}

            if return_metadata:
                item.update({"subject": s, "object": o})

            samples.append(item)

    return samples

# 5) TRAIN PIPELINE (SetFit)
# -------------------------
def train_setfit_from_samples(samples, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                              test_size=0.2, random_state=42, num_iterations=0, num_epochs=1, save_dir="data/my_fast_re_model"):
    """
    Huấn luyện SetFit nhanh chóng từ các mẫu được tạo.
    - num_iterations/num_epochs được chuyển tiếp đến huấn luyện viên (điều chỉnh khi cần thiết).
    """
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from setfit import SetFitModel, SetFitTrainer

    if not samples:
        raise ValueError("No samples to train on!")

    # Label mapping
    unique_labels = sorted(list({s["label"] for s in samples}))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Split
    train_data, test_data = train_test_split(samples, test_size=test_size, random_state=random_state, stratify=[s["label"] for s in samples])

    train_ds = Dataset.from_list(train_data)
    test_ds  = Dataset.from_list(test_data)
    print('len train dataset',len(train_ds))
    print('len test dataset', len(test_ds))
    # print('in thử train dataset', train_data[3:])

    model = SetFitModel.from_pretrained(
        model_name,
        labels=unique_labels,
        id2label=id2label,
        label2id=label2id
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        column_mapping={"text": "text", "label": "label"},
    )

    # Train chỉ classifier head, BỎ QUA contrastive learning
    from setfit import TrainingArguments
    args = TrainingArguments(
        num_epochs=5,
        batch_size=16,
        num_iterations=0,  # Tắt hoàn toàn contrastive learning
    )
    trainer.train(args=args)

    
    model.save_pretrained(save_dir)
    return model, trainer

def sample_balanced_triples(triples, n_samples=1000):
    """
    Lấy đúng n_samples triples, mỗi relation có số lượng = nhau.
    Nếu thiếu triple ở relation nào → tạo triple synthetic để bù.
    """

    # Nhóm triple theo relation
    by_relation = defaultdict(list)
    for s, r, o in triples:
        by_relation[r].append((s, r, o))

    relations = list(by_relation.keys())
    n_rel = len(relations)
    n_per_rel = n_samples // n_rel  # VD: 1000/5 = 200

    final_triples = []

    for r in relations:
        group = by_relation[r]
        count = len(group)

        # Trường hợp 1: đủ hoặc dư triple → random chọn 200
        if count >= n_per_rel:
            chosen = random.sample(group, n_per_rel)
            final_triples.extend(chosen)

        # Trường hợp 2: thiếu triple → lấy hết + tạo thêm synthetic triple
        else:
            final_triples.extend(group)

            missing = n_per_rel - count
            # Tạo triple synthetic bằng cách clone và sửa nhẹ ID
            for i in range(missing):
                s, _, o = random.choice(group)
                synthetic = (f"{s}_AUG{i}", r, f"{o}_AUG{i}")
                final_triples.append(synthetic)

    # Đảm bảo đúng n_samples
    return final_triples[:n_samples]

# hàm dưới thiếu balance, hôn nhân 40, còn lại 300 ... 
# def sample_balanced_triples(triples, n_samples=1000):
#     """
#     Lấy ngẫu nhiên n_samples triples, cân bằng theo relation
#     """
    
    
#     # Nhóm theo relation
#     by_relation = defaultdict(list)
#     for s, r, o in triples:
#         by_relation[r].append((s, r, o))
    
#     relations = list(by_relation.keys())
#     n_per_relation = n_samples // len(relations)
    
#     sampled = []
#     for r in relations:
#         available = by_relation[r]
#         # Lấy min(n_per_relation, số triple có sẵn)
#         k = min(n_per_relation, len(available))
#         sampled.extend(random.sample(available, k))
    
#     # Nếu còn thiếu, bù thêm ngẫu nhiên
#     if len(sampled) < n_samples:
#         remaining = [t for t in triples if t not in sampled]
#         sampled.extend(random.sample(remaining, n_samples - len(sampled)))
    
#     return sampled[:n_samples]



# if __name__ == "__main__":
    
#     triples = load_triples()
#     print("Số triples:", len(triples))
    
    
#     # LẤY MẪU CÂN BẰNG
#     triples = sample_balanced_triples(triples, n_samples=1000)
#     print("Số triples sau khi sample:", len(triples))

#     # Kiểm tra phân bố
#     from collections import Counter
#     relation_dist = Counter([r for _, r, _ in triples])
#     print("Phân bố relations:", relation_dist)


#     # Bao nhiêu câu / triple (recommend: 3..5)
#     n_per_triple = 1
#     mask_entities = True  # hoặc False nếu không muốn bọc tag

#     samples = build_train_samples_from_triples(triples, n_per_triple, mask_entities=mask_entities)
#     print("Generated samples:", len(samples))
#     print("Hiện ví dụ 6 sample:", samples[:6])

#     # Train thử
#     model, trainer = train_setfit_from_samples(samples, save_dir="data/re_model")
#     print("Done training & saved to data/re_model")


# TRAIN LÂU NÊN SỬ DỤNG GG COLAB
# notebooks/train_setfit.ipynb





