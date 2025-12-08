import random
import json
from collections import defaultdict
from src.nlp.ner import person_list, film_list

TEMPLATES = {
    "SPOUSE": [
        "{s} và {o} là vợ chồng.",
        "{s} kết hôn với {o}.",
        "{s} và {o} đã kết hôn với nhau.",
        "{s} là chồng/vợ của {o}.",
        "{o} là chồng/vợ của {s}.",
        "Cuộc hôn nhân giữa {s} và {o} được nhiều người biết đến.",
        "{s} và {o} sống cùng nhau như một cặp vợ chồng.",
        "{s} và {o} đã lập gia đình.",
        "{s} có quan hệ hôn nhân với {o}.",
        "{o} là người bạn đời của {s}.",
        "{s} và {o} chính thức trở thành vợ chồng sau lễ cưới.",
        "{s} đã tổ chức lễ cưới cùng {o}.",
        "Tình trạng hôn nhân của {s} hiện nay: kết hôn với {o}.",
        "{o} và {s} đã chia sẻ cuộc sống hôn nhân với nhau.",
        "{s} đã nên duyên vợ chồng với {o}.",
        "Người bạn đời của {s} là {o}.",
        "{s} gọi {o} là vợ/chồng của mình.",
        "{s} và {o} là một cặp đôi nổi tiếng.",
        "{o} gắn bó với {s} trong quan hệ hôn nhân.",
        "{s} và {o} chính thức xác nhận quan hệ vợ chồng."
    ],

    "SAME_HOMETOWN_AS": [
        "{s} cùng quê với {o}.",
        "{s} và {o} đều sinh ra ở cùng một quê quán.",
        "{s} có chung nơi sinh cùng {o}.",
        "{s} và {o} đến từ cùng một tỉnh/thành.",
        "{s} và {o} đều xuất thân từ cùng một làng xã.",
        "{s} và {o} có cùng quê quán.",
        "{s} và {o} đều mang quê quán giống nhau.",
        "{s} quê ở cùng nơi với {o}.",
        "{s} và {o} sinh cùng quê.",
        "Quê quán của {s} trùng với quê quán của {o}.",
        "{s} và {o} đều lớn lên ở cùng một nơi.",
        "{s} được biết đến đến từ cùng thành phố với {o}.",
        "{s} và {o} có xuất xứ từ cùng một vùng.",
        "{s} và {o} sinh sống ban đầu ở cùng một địa phương.",
        "{s} có cùng quê với {o}.",
        "{s} và {o} đến từ cùng một huyện/tỉnh.",
        "{s} và {o} chia sẻ cùng một quê quán.",
        "Cả {s} và {o} đều có quê quán tại cùng một nơi."
    ],

    "ACTED_IN": [
        "{s} tham gia diễn xuất trong phim {o}.",
        "{s} là diễn viên của {o}.",
        "{s} có vai diễn trong {o}.",
        "{s} góp mặt với tư cách diễn viên trong {o}.",
        "{s} xuất hiện trên màn ảnh trong tác phẩm {o}.",
        "{s} có mặt trong dàn diễn viên của phim {o}.",
        "{s} góp mặt trong bộ phim {o} với vai trò diễn viên.",
        "{s} đóng vai trong {o}.",
        "Trong phim {o}, {s} đảm nhận 1 vai diễn.",
        "{s} là một trong những diễn viên tham gia {o}.",
        "{s} góp mặt ở dự án điện ảnh {o}.",
        "{s} thủ vai trong tác phẩm {o}.",
        "{s} xuất hiện trong bộ phim mang tên {o}.",
        "{s} từng đóng trong phim {o}.",
        "{s} là diễn viên đóng chính/đóng vai phụ trong {o}.",
        "{s} và {o} có quan hệ diễn xuất (actor–film).",
        "{s} từng hợp tác trong dự án phim {o}.",
        "{s} tham gia sản xuất/diễn xuất ở {o}.",
        "{s} góp mặt trong danh sách diễn viên của {o}.",
        "{s} đảm nhận vai diễn trong {o}."
    ],

    "DIRECTED": [
        "{s} đạo diễn bộ phim {o}.",
        "{s} là đạo diễn của {o}.",
        
        "{s} giữ vai trò đạo diễn trong {o}.",
        "{s} dẫn dắt dự án điện ảnh {o} với tư cách đạo diễn.",
        "{o} được đạo diễn bởi {s}.",
        "{s} chịu trách nhiệm đạo diễn cho {o}.",
        
        "{s} đảm nhiệm vai trò đạo diễn trong {o}.",
        "{s} dẫn dắt ê-kíp thực hiện {o}.",
        "{s} đã đạo diễn bộ phim mang tên {o}.",
        "{s} làm đạo diễn cho tác phẩm {o}.",
        
        
        "{s} là đạo diễn chính của {o}.",
        "{o} do {s} làm đạo diễn.",
        "{s} nắm vai trò đạo diễn trong dự án {o}.",
        "{s} là đầu tàu đạo diễn cho {o}.",
        "{s} tổ chức và đạo diễn bộ phim {o}.",
        "{s} đã chỉ đạo quá trình sản xuất {o} với tư cách đạo diễn."
    ],

    "COLLABORATED_WITH": [
        "{s} đã hợp tác cùng {o} trong một dự án phim.",
        "{s} và {o} từng cộng tác làm phim với nhau.",
        "{s} hợp tác nghề nghiệp với {o}.",
        "{s} và {o} có lịch sử hợp tác nghệ thuật.",
        "{s} đã làm việc chung cùng {o}.",
        "{s} cộng tác với {o} trong nhiều dự án.",
        "{s} từng thực hiện dự án chung với {o}.",
        "{s} và {o} cộng tác trong một bộ phim.",
        "{s} và {o} từng hợp tác trên màn ảnh/sa bàn.",
        "{s} có quan hệ hợp tác chuyên môn với {o}.",
        "{s} làm việc cùng {o} trong vai trò cộng tác viên.",
        "{s} và {o} từng xuất hiện cùng nhau trong cùng một dự án.",
        "{s} từng hợp tác sản xuất/diễn xuất cùng {o}.",
        "{s} và {o} có quan hệ làm việc chung.",
        "{s} đã cùng {o} thực hiện một tác phẩm.",
        "{s} và {o} nằm trong danh sách cộng tác viên của cùng 1 dự án.",
        "{s} và {o} có mối quan hệ hợp tác lâu dài.",
        "{s} từng tham gia hợp tác với {o} trên một bộ phim.",
        "{s} và {o} làm việc cùng nhau cho một dự án.",
        "{s} và {o} có danh sách phim hợp tác chung."
    ]
}
def sample_templates_for_relation(rel, n):
    """
    Trả về n template ngẫu nhiên cho quan hệ *rel* (chưa format).
    - rel: tên quan hệ, ví dụ "SPOUSE", "ACTED_IN".
    - n: số lượng template muốn lấy.
    - Nếu rel không có trong TEMPLATES → báo lỗi để nhắc bổ sung.
    """
    if rel not in TEMPLATES:
        raise KeyError(f"Relation {rel} chưa có template. Thêm vào TEMPLATES.")

    templates = TEMPLATES[rel]
    n = min(n, len(templates))  # tránh yêu cầu nhiều hơn số template hiện có
    return random.sample(templates, n)  # chọn ngẫu nhiên n template


def guess_tag(entity):
    """
    Đoán loại thực thể (tag) dựa vào 2 danh sách đã có:
      - person_list: danh sách người
      - film_list: danh sách phim
    Trả về:
      - "PER" nếu entity nằm trong person_list
      - "FILM" nếu entity nằm trong film_list
      - "ENT" nếu không xác định được → thực thể chung (generic entity)
    """
    if entity in person_list:
        return "PER"
    if entity in film_list:
        return "FILM"
    return "ENT"


def mask_entity_simple(text, entity, tag="PER"):
    """
    Che (mask) entity trong câu bằng cách bọc nó thành:
        [TAG] entity [/TAG]

    - text: câu gốc
    - entity: tên thực thể cần mask chính xác
    - tag: loại thực thể, mặc định "PER"

    Lưu ý:
      - Dùng regex tìm *chính xác* entity theo dạng chuỗi (có phân biệt word-boundary cơ bản).
      - re.IGNORECASE → không phân biệt hoa thường.
      - Không enforce mạnh word-boundary, đủ dùng cho template generation.
    """
    import re
    # (?<!\w)entity(?!\w): tránh match khi entity bị dính vào từ khác
    pattern = r'(?<!\w){}(?!\w)'.format(re.escape(entity))

    # Thay thế bằng markup: [TAG] entity [/TAG]
    return re.sub(
        pattern,
        f"[{tag}] {entity} [/{tag}]",
        text,
        flags=re.IGNORECASE
    )
# 3) SAMPLE GENERATOR


def generate_text_samples_from_triple(s, r, o, n_samples=4, mask_entities=False):
    """
    Sinh ra n_samples câu văn mô tả triple (s, r, o).
    Tham số:
      - s: subject (thực thể bên trái)
      - r: relation (quan hệ)
      - o: object (thực thể bên phải)
      - n_samples: số lượng câu muốn tạo (3–5 nên dùng)
      - mask_entities: nếu True → bọc entity bằng tag dạng [TAG] ... [/TAG]
                       nếu False → giữ nguyên câu không mask.

    Cơ chế:
      1. Lấy n_samples template phù hợp với quan hệ r (random).
      2. Format từng template bằng s và o.
      3. Nếu bật mask_entities:
         - Đoán tag của s và o (PER / FILM / ENT).
         - Dùng hàm mask_entity_simple để bọc entity trong câu.
      4. Trả về danh sách các câu.
    """
    # Bước 1: lấy n template theo quan hệ r
    templates = sample_templates_for_relation(r, n_samples)

    # Bước 2: format các template → tạo câu hoàn chỉnh
    texts = [t.format(s=s, o=o) for t in templates]

    # Bước 3: nếu có bật mask entity
    if mask_entities:
        s_tag = guess_tag(s)  # đoán loại entity của s
        o_tag = guess_tag(o)  # đoán loại entity của o
        masked_texts = []

        for t in texts:
            # mask subject trước
            t2 = mask_entity_simple(t, s, s_tag)
            # mask object sau
            t2 = mask_entity_simple(t2, o, o_tag)
            masked_texts.append(t2)

        return masked_texts

    # nếu không mask → trả lại texts thuần
    return texts


print(generate_text_samples_from_triple('Hari Won', 'SPOUSE', 'Trấn Thành'))