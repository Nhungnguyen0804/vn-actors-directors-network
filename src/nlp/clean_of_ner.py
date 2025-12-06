import unicodedata
import re
# loại bỏ dấu câu trước khi xử lý văn bản
RE_PUNCT = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\'\"“”‘’!#$%&()*+,\-./:;<=>?@\[\]^_`{|}~…—·]+")
# chuẩn hóa khoảng trắng
RE_SPACE = re.compile(r"\s+")
# vào str -> trả str

# ************************************ Dùng cho văn bản dài 

def normalize_text_for_nlp(text):
    """
    Chuẩn hóa toàn diện cho text:
    - Unicode NFKC
    - lowercase
    - remove punctuation
    - remove extra spaces
    - strip
    """
    if not text:
        return ""

    # Normalize Unicode + lowercase
    text = unicodedata.normalize("NFKC", str(text)).lower()

    # Remove punctuation
    text = RE_PUNCT.sub(" ", text)

    # Remove extra spaces
    text = RE_SPACE.sub(" ", text)

    return text.strip()

print(normalize_text_for_nlp('Hoa Mặt Trời')) # ==> hoa mặt trời


import re
import unicodedata

def normalize_entity_name(x):
    # Ninh Dương Lan Ngọc . ==> Ninh Dương Lan Ngọc 
    """Chuẩn hoá tên entity: unicode, xoá khoảng trắng, gom space, bỏ dấu câu đầu/cuối."""
    if not isinstance(x, str):
        x = str(x)

    # Chuẩn hoá unicode + bỏ khoảng trắng đầu/cuối
    x = unicodedata.normalize("NFKC", x).strip()
    if not x:
        return ""

    # Nếu chuỗi chỉ toàn ký tự đặc biệt / số → loại bỏ
    if re.fullmatch(r"[\W\d_]+", x):
        return ""

    # Gom nhiều khoảng trắng thành 1
    x = re.sub(r"\s+", " ", x).strip()

    # Hàm kiểm tra ký tự có phải dấu câu (Unicode) không
    # Ví dụ: ., , : ; … “ ” !
    def _is_punct(ch):
        return unicodedata.category(ch).startswith("P")

    # Bỏ dấu câu ở đầu chuỗi
    start = 0
    while start < len(x) and _is_punct(x[start]):
        start += 1

    # Bỏ dấu câu ở cuối chuỗi
    end = len(x) - 1
    while end >= start and _is_punct(x[end]):
        end -= 1

    # Lấy phần còn lại
    x = x[start:end+1].strip()

    # Gom space lại lần cuối 
    x = re.sub(r"\s+", " ", x).strip()
    return x


print(normalize_entity_name('Ninh Dương Lan Ngọc .'))
def normalize_type(t, default):
    """Chuẩn hóa type: viết hoa chữ đầu."""
    if not t: return default
    t = str(t).strip()
    if not t: return default
    return t[0].upper() + t[1:]