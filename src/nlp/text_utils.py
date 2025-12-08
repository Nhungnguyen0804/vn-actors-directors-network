import unicodedata
import underthesea
import re
# loại bỏ dấu câu trước khi xử lý văn bản
RE_PUNCT = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F\'\"“”‘’!#$%&()*+,\-./:;<=>?@\[\]^_`{|}~…—·]+")
# chuẩn hóa khoảng trắng
RE_SPACE = re.compile(r"\s+")

def remove_footnotes(text):
    return re.sub(r'\s*\[\d+\]\s*', ' ', text)
# vào str -> trả str
def split_text_into_sentences(text):
    return underthesea.sent_tokenize(text)


def remove_text_in_parentheses(text):   # xóa text trong ngoặc
    cleaned_text = re.sub(r"\s*\(.*?\)", "", text)  # kết quả sau khi xóa
    return cleaned_text
print(remove_text_in_parentheses('Hoa Mặt Trời (phim truyền hình)')) # ==> Hoa Mặt Trời
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

def norm(s):
    return unicodedata.normalize("NFC", s.strip()).lower()


# Normalize entity ⇒ trả về PER, FILM để khớp với NER combine 

def normalize_entity(entity_text, person_list, film_list, wiki_enrich=None):
    if not entity_text:
        return "", "UNK"

    t = entity_text.strip()
    tl = norm(t)

    for p in person_list:
        if tl == norm(p):
            return p, "PER"

    for f in film_list:
        if tl == norm(f):
            return f, "FILM"

    # khớp fuzzy vào wiki
    if wiki_enrich:
        for w in wiki_enrich:
            if tl == norm(w):
                return w, "UNK"

    return t, "UNK"

