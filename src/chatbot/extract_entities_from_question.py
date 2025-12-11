import unicodedata
import re
from underthesea import pos_tag, word_tokenize
# ==================== CONSTANTS ====================

VIETNAMESE_STOPWORDS = {
    'và', 'hoặc', 'hay', 'nhưng', 'mà', 'vì', 'khi', 'nếu', 'để', 'từ', 'đến', 'với', 'trong', 'ngoài',
    'trên', 'dưới', 'trước', 'sau', 'giữa', 'bên', 'cạnh', 'quanh', 'xung', 
    'là', 'có', 'được', 'làm', 'cho', 'bị', 'về', 'đi', 'lên', 'xuống', 'vào', 'ra', 'nên',
    'như', 'thì', 'càng', 'lại', 'chỉ', 'cũng', 'còn', 'cơ', 'do', 'tuy', 'tại', 'bằng',
    'của', 'không', 'này', 'kia', 'nọ', 'gì', 'ai', 'nào', 'đâu', 'sao', 'bao',
    'đã', 'đang', 'sẽ', 'đáng', 'vừa', 'sắp', 'chuẩn', 'chưa', 'rồi',
    'tôi', 'tao', 'ta', 'tớ', 'anh', 'chị', 'em', 'bạn', 'ông', 'cha', 'mẹ', 'con', 'cô', 'chú', 'bác',
    'nhanh', 'chậm', 'lâu', 'sớm', 'muộn', 'lắm', 'nhiều', 'ít', 'cả', 'luôn',
    'a', 'à', 'ơi', 'hi', 'hôm', 'nay', 'ngày', 'tháng', 'năm', 'giờ', 'phút', 'giây',
    'muốn', 'tìm', 'kiếm', 'hỏi', 'biết', 'xem', 'tui', 'mik', 'the', 'chung', 'sự', "trường"
}

# Từ khóa dễ nhầm (khi bỏ dấu) - KHÔNG đưa vào stopwords
# Ví dụ: "bà" (bà già) vs "ba" (ba lô) - khác nghĩa nhưng khi bỏ dấu giống nhau
AMBIGUOUS_WORDS = {
    'ba', 'nha', 'gia', 'nu', 'hai'  # Có thể là stopword hoặc entity tùy ngữ cảnh
}

# Họ phổ biến Việt Nam (để nhận diện tên người)
VIETNAMESE_SURNAMES = {
    'nguyen', 'tran', 'le', 'pham', 'hoang', 'phan', 'vu', 'vo', 'dang', 'bui',
    'do', 'ho', 'ngo', 'duong', 'ly', 'truong', 'dinh', 'mai', 'luong', 'cao',
    'thai', 'ta', 'trinh', 'lam', 'ha', 'son', 'huynh', 'nghiem', 'kieu', 'quach'
}

# Tên đệm phổ biến
VIETNAMESE_MIDDLE_NAMES = {
    'van', 'thi', 'duc', 'anh', 'minh', 'hong', 'thu', 'thuy', 'thanh', 'huu',
    'ngoc', 'kim', 'bao', 'hai', 'quoc', 'duy', 'bich', 'xuan', 'gia', 'mai',
    'khanh', 'huyen', 'tuan', 'quang', 'phuong', 'thien', 'trung', 'hoai', 'thu'
}

# Tên (given name) phổ biến
VIETNAMESE_GIVEN_NAMES = {
    'anh', 'linh', 'huong', 'lan', 'my', 'ha', 'trang', 'hoa', 'nga', 'thao',
    'phuong', 'dung', 'yen', 'quyen', 'tam', 'tuan', 'hung', 'huy', 'thanh',
    'long', 'khoa', 'phong', 'trong', 'son', 'hai', 'ngoc', 'bao', 'khanh'
}

ACTION_KEYWORDS = {
    'đóng', 'hợp', 'tác', 'cùng', 'diễn', 'viên', 'phim', 'đạo', 
    'vs', 'tham', 'gia', 'sản', 'xuất', 'chiếu', 'rạp', 'vai', 'thủ',
}

FILLER_WORDS = {
    'co', 'có', 'hay', 'khong', 'không', 'à', 'ơi', 'nào', 'gì', 
    'sao', 'thế', 'vậy', 'hả', 'chưa', 'rồi', 'đâu', 'ko', 'hem', 'hok',
    've', 'về', 'la', 'là', 'cua', 'của', 'ma', 'mà', 'dong', 'dien', 'cho'
}

QUESTION_WORDS = {
    'ai', 'gì', 'đâu', 'nào', 'bao giờ', 'khi nào', 'thế nào', 'sao',
    'như thế nào', 'ra sao', 'mấy', 'bao nhiêu', 'lúc nào'
}

# Từ liên kết MẠNH (ngắt entity ngay lập tức)
STRONG_SEPARATORS = {'và', 'hoặc', 'hay', 'với', 'cùng', 'vs', 'va', 'voi', 'cung'}

# Map sửa lỗi teencode
TEENCODE_MAP = {
    'k': 'khong', 'ko': 'khong', 'kh': 'khong', 'hok': 'khong', 'hem': 'khong',
    'dc': 'duoc', 'dk': 'duoc', 'đc': 'duoc',
    'vs': 'voi', 'j': 'gi', 'z': 'gi', 'dz': 'gi',
    'ng': 'nguoi', 'ntn': 'nhu the nao',
    'bh': 'bao gio', 'bn': 'bao nhieu',
    'cx': 'cung', 'tl': 'tra loi', 'ck': 'chu ky',
     'tui': 'toi'
}

# ==================== UTILITY FUNCTIONS ====================

def remove_vietnamese_accents(text):
    """Loại bỏ dấu tiếng Việt"""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return unicodedata.normalize('NFC', text).replace('đ', 'd').replace('Đ', 'D')


def normalize_text(text):
    """
    Chuẩn hóa: không dấu, chữ thường, bỏ dấu câu
    
    Args:
        text (str): Text cần chuẩn hóa
    
    Returns:
        str: Text đã chuẩn hóa
    """
    if not isinstance(text, str):
        return ''
    
    
   
    text = re.sub(r'[^\w\s]', ' ', text)  # Thay dấu câu bằng khoảng trắng
    text = text.strip()
    
    # Loại bỏ dấu tiếng Việt
    text = remove_vietnamese_accents(text)
    
    # Chữ thường
    text = text.lower()
    
    # Bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def fix_teencode(text):
    """Sửa teencode"""
    words = text.lower().split()
    fixed = []
    for w in words:
        # Sửa teencode đơn lẻ
        if w in TEENCODE_MAP:
            fixed.append(TEENCODE_MAP[w])
        else:
            fixed.append(w)
    return " ".join(fixed)


def expand_stopwords(original_set):
    """Mở rộng stopwords với phiên bản không dấu"""
    expanded = set(original_set)
    for word in original_set:
        expanded.add(remove_vietnamese_accents(word))
    return expanded


# Tạo bộ từ chặn đầy đủ
FULL_BLOCK_WORDS = expand_stopwords(
    VIETNAMESE_STOPWORDS | ACTION_KEYWORDS | FILLER_WORDS | QUESTION_WORDS
)

FULL_SEPARATORS = expand_stopwords(STRONG_SEPARATORS)


def is_vietnamese_name(words):
    """
    Kiểm tra xem chuỗi từ có phải tên người Việt không
    Pattern: [Họ] + [Tên đệm]? + [Tên]
    Ví dụ: Kiều Minh Tuấn, Trấn Thành, Lan Ngọc
    
    Args:
        words: list các từ (đã normalize)
    
    Returns:
        bool: True nếu có khả năng là tên người
    """
    if not words or len(words) < 2 or len(words) > 4:
        return False
    
    words_norm = [normalize_text(w) for w in words]
    
    # Pattern 1: [Họ] + [Tên] (2 từ)
    # Ví dụ: Trấn Thành, Lan Ngọc
    if len(words_norm) == 2:
        first, last = words_norm
        if first in VIETNAMESE_SURNAMES:
            return True
        # Họ không phổ biến nhưng tên phổ biến
        if last in VIETNAMESE_GIVEN_NAMES:
            return True
    
    # Pattern 2: [Họ] + [Tên đệm] + [Tên] (3 từ)
    # Ví dụ: Kiều Minh Tuấn, Mai Tài Phến
    if len(words_norm) == 3:
        first, middle, last = words_norm
        
        # Họ + tên đệm + tên
        if first in VIETNAMESE_SURNAMES and middle in VIETNAMESE_MIDDLE_NAMES:
            return True
        
        # Họ + bất kỳ + tên phổ biến
        if first in VIETNAMESE_SURNAMES and last in VIETNAMESE_GIVEN_NAMES:
            return True
        
        # Bất kỳ + tên đệm + tên
        if middle in VIETNAMESE_MIDDLE_NAMES and last in VIETNAMESE_GIVEN_NAMES:
            return True
    
    # Pattern 3: [Họ] + [Tên đệm] + [Tên đệm] + [Tên] (4 từ - hiếm)
    if len(words_norm) == 4:
        first = words_norm[0]
        if first in VIETNAMESE_SURNAMES:
            return True
    
    return False


def is_stopword(word, check_context=False, surrounding_words=None):
    """
    Kiểm tra từ có phải stopword không (có xét ngữ cảnh)
    
    Args:
        word: Từ cần kiểm tra
        check_context: Có kiểm tra ngữ cảnh không
        surrounding_words: List các từ xung quanh (để xét ngữ cảnh)
    """
    word_norm = normalize_text(word)
    
    # **QUAN TRỌNG: Kiểm tra xem có phải trong tên người không**
    if check_context and surrounding_words:
        # Lấy cửa sổ 3-4 từ xung quanh
        window = [word] + surrounding_words
        if is_vietnamese_name(window):
            return False  # KHÔNG phải stopword nếu nằm trong tên người
    
    # Từ dễ nhầm lẫn (ambiguous) - cần xét ngữ cảnh
    if word_norm in AMBIGUOUS_WORDS:
        if not check_context or not surrounding_words:
            # Nếu không có ngữ cảnh, cho phép giữ lại (không loại)
            return False
        
        # Xét ngữ cảnh: nếu xung quanh có từ có nghĩa → không phải stopword
        surrounding_meaningful = [
            w for w in surrounding_words 
            if normalize_text(w) not in FULL_BLOCK_WORDS 
            and normalize_text(w) not in AMBIGUOUS_WORDS
            and len(w) > 1
        ]
        
        if len(surrounding_meaningful) > 0:
            return False  # Giữ lại (vì có từ có nghĩa xung quanh)
    
    # Kiểm tra stopword thông thường
    return word_norm in FULL_BLOCK_WORDS


def is_valid_entity(text, allow_short=False):
    """Kiểm tra entity hợp lệ"""
    if not text or len(text.strip()) < 2:
        return False
    
    # Loại số thuần
    if text.replace(' ', '').isdigit():
        return False
    
    # Kiểm tra có ít nhất 1 từ có nghĩa
    words = text.split()
    
    # Đếm từ có nghĩa (không phải stopword chắc chắn)
    meaningful = []
    for i, w in enumerate(words):
        word_norm = normalize_text(w)
        
        # Bỏ qua từ quá ngắn
        if len(w) <= 1:
            continue
        
        # Nếu là từ dễ nhầm - xét ngữ cảnh
        if word_norm in AMBIGUOUS_WORDS:
            # Lấy từ xung quanh
            surrounding = []
            if i > 0:
                surrounding.append(words[i-1])
            if i < len(words) - 1:
                surrounding.append(words[i+1])
            
            if not is_stopword(w, check_context=True, surrounding_words=surrounding):
                meaningful.append(w)
        elif word_norm not in FULL_BLOCK_WORDS:
            meaningful.append(w)
    
    # Nếu có ít nhất 1 từ có nghĩa → hợp lệ
    if len(meaningful) > 0:
        return True
    
    # Trường hợp đặc biệt: cho phép entity ngắn nếu tất cả từ là ambiguous
    # Ví dụ: "bộ già", "nhà bà nữ"
    if allow_short and len(words) >= 2:
        ambiguous_count = sum(1 for w in words if normalize_text(w) in AMBIGUOUS_WORDS)
        if ambiguous_count == len(words):
            return True
    
    return False


def reconstruct_with_accents(normalized_phrase, original_text):
    """
    Tái tạo cụm từ có dấu từ văn bản gốc
    """
    # Chuẩn hóa text gốc
    original_norm = normalize_text(original_text)
    phrase_norm = normalize_text(normalized_phrase)
    
    # Tìm vị trí xuất hiện
    idx = original_norm.find(phrase_norm)
    if idx == -1:
        return normalized_phrase
    
    # Đếm số khoảng trắng trước vị trí đó để tính word offset
    prefix = original_norm[:idx]
    word_offset = len(prefix.split()) if prefix else 0
    
    # Lấy đúng số từ từ text gốc
    original_words = original_text.split()
    phrase_words = normalized_phrase.split()
    
    if word_offset + len(phrase_words) <= len(original_words):
        result = " ".join(original_words[word_offset:word_offset + len(phrase_words)])
        return result
    
    return normalized_phrase


# ==================== MAIN EXTRACTION FUNCTION ====================

def extract_entities_with_pos(question):
    """
    Trích xuất thực thể bằng POS tagging (cho câu có dấu)
    Chỉ lấy Danh từ (N, Np, Nu) và loại bỏ Động từ (V)
    """
    try:
        # POS tagging
        pos_tags = pos_tag(question)
        # Format: [('Trấn Thành', 'Np'), ('đóng', 'V'), ('phim', 'N'), ...]
        
        entities_list = []
        current_entity = []
        
        for word, tag in pos_tags:
            word_norm = normalize_text(word)
            
            # Lấy các từ loại là Danh từ (Noun)
            # N: danh từ thường, Np: tên riêng, Nu: danh từ đơn vị, Ny: danh từ viết tắt
            if tag in ['N', 'Np', 'Nu', 'Ny']:
                # Bỏ qua stopword
                if word_norm not in FULL_BLOCK_WORDS and len(word) > 1:
                    current_entity.append(word)
            else:
                # Gặp từ loại khác (động từ, tính từ...) → ngắt cụm
                if current_entity:
                    entity = " ".join(current_entity)
                    if is_valid_entity(entity, allow_short=True):
                        entities_list.append(entity)
                    current_entity = []
        
        # Lưu cụm cuối
        if current_entity:
            entity = " ".join(current_entity)
            if is_valid_entity(entity, allow_short=True):
                entities_list.append(entity)
        
        return entities_list
    
    except Exception as e:
        return []


def extract_entities(question):
    """
    Trích xuất thực thể từ câu hỏi - HYBRID APPROACH
    
    Strategy:
    1. Nếu câu có dấu (≥30% ký tự có dấu) → Dùng POS tagging
    2. Nếu câu không dấu → Dùng heuristic + name pattern
    3. Kết hợp cả 2 để tăng độ chính xác
    
    Args:
        question (str): Câu hỏi đầu vào
        
    Returns:
        list: Danh sách các thực thể
    """
    if not question or len(question.strip()) < 2:
        return []
    
    original_q = question.strip()
    q_fixed = fix_teencode(original_q)
    
    entities_list = []
    
    # ===== KIỂM TRA: Câu có dấu không? =====
    accent_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    accent_count = sum(1 for c in original_q.lower() if c in accent_chars)
    total_alpha = sum(1 for c in original_q if c.isalpha())
    accent_ratio = accent_count / max(total_alpha, 1)
    
   
    
    # ===== PHƯƠNG PHÁP 1: POS TAGGING (nếu câu có dấu) =====
    if accent_ratio >= 0.3:
        pos_entities = extract_entities_with_pos(original_q)
        entities_list.extend(pos_entities)
    
    # ===== PHƯƠNG PHÁP 2: HEURISTIC (luôn chạy để bổ sung) =====
    # Thay các từ nối MẠNH bằng dấu phân cách đặc biệt
    text_to_process = q_fixed.lower()
    for sep in STRONG_SEPARATORS:
        text_to_process = re.sub(rf'\b{sep}\b', ' | ', text_to_process)
    
    # Tách theo dấu câu và |
    segments = re.split(r'[,;|]', text_to_process)
    
    entities_list = []
    
    # ===== BƯỚC 2: Xử lý từng segment =====
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Tokenize
        tokens = segment.split()
        
        current_phrase = []
        i = 0
        
        while i < len(tokens):
            word = tokens[i]
            word_norm = normalize_text(word)
            
            # **CHECK 1: Có phải bắt đầu tên người không?**
            # Nhìn trước 2-3 từ tiếp theo
            lookahead = tokens[i:min(i+4, len(tokens))]
            if is_vietnamese_name(lookahead):
                # Đây là tên người → gom hết
                name_length = min(len(lookahead), 3)  # Tối đa 3 từ
                full_name = " ".join(tokens[i:i+name_length])
                entities_list.append(full_name)
                i += name_length
                current_phrase = []  # Reset cụm hiện tại
                continue
            
            # **CHECK 2: Kiểm tra stopword có xét ngữ cảnh**
            surrounding = []
            if i > 0:
                surrounding.append(tokens[i-1])
            if i < len(tokens) - 1:
                surrounding.append(tokens[i+1])
            if i < len(tokens) - 2:
                surrounding.append(tokens[i+2])
            
            if is_stopword(word, check_context=True, surrounding_words=surrounding) or len(word) <= 1:
                # Gặp stopword → lưu cụm hiện tại
                if current_phrase:
                    entity = " ".join(current_phrase)
                    if is_valid_entity(entity, allow_short=True):
                        entities_list.append(entity)
                    current_phrase = []
            else:
                # Thêm từ vào cụm
                current_phrase.append(word)
                
                # Giới hạn độ dài entity tối đa 4 từ
                if len(current_phrase) >= 4:
                    entity = " ".join(current_phrase)
                    if is_valid_entity(entity, allow_short=True):
                        entities_list.append(entity)
                    current_phrase = []
            
            i += 1
        
        # Lưu cụm cuối
        if current_phrase:
            entity = " ".join(current_phrase)
            if is_valid_entity(entity, allow_short=True):
                entities_list.append(entity)
    
    # ===== BƯỚC 3: Tìm tên riêng (chữ hoa) =====
    # Pattern: Từ bắt đầu bằng chữ hoa
    proper_nouns = re.findall(
        r'\b[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]'
        r'[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]*'
        r'(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]'
        r'[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]*){0,3}',
        original_q
    )
    
    for proper_noun in proper_nouns:
        # Kiểm tra không phải stopword
        if normalize_text(proper_noun) not in FULL_BLOCK_WORDS:
            entities_list.append(proper_noun)
    
    # ===== BƯỚC 4: Làm sạch và khôi phục dấu =====
    cleaned_entities = []
    seen = set()
    
    for entity in entities_list:
        # Khôi phục dấu
        entity_with_accent = reconstruct_with_accents(entity, original_q)
        
        # Loại bỏ stopword ở đầu và cuối
        words = entity_with_accent.split()
        while words and is_stopword(words[0], check_context=False):
            words.pop(0)
        while words and is_stopword(words[-1], check_context=False):
            words.pop()
        
        if not words:
            continue
        
        final_entity = " ".join(words)
        norm = normalize_text(final_entity)
        
        # Kiểm tra trùng lặp
        if norm not in seen and is_valid_entity(final_entity, allow_short=True):
            seen.add(norm)
            cleaned_entities.append(final_entity)
    
    # ===== BƯỚC 5: Loại bỏ entity con (subset) =====
    filtered = []
    for i, ent1 in enumerate(cleaned_entities):
        norm1 = normalize_text(ent1)
        is_subset = False
        
        for j, ent2 in enumerate(cleaned_entities):
            if i == j:
                continue
            norm2 = normalize_text(ent2)
            
            # ent1 là con của ent2 nếu:
            # - norm1 xuất hiện trong norm2
            # - norm1 không bằng norm2
            # - norm1 không dài hơn norm2
            if norm1 != norm2 and norm1 in norm2 and len(norm1) < len(norm2):
                is_subset = True
                break
        
        if not is_subset:
            filtered.append(ent1)
    
    # ===== BƯỚC 6: Sắp xếp theo thứ tự xuất hiện =====
    def position_in_original(entity):
        try:
            return original_q.lower().index(normalize_text(entity))
        except:
            return 9999
    
    filtered.sort(key=position_in_original)
    
    return filtered



# def extract_entities(question: str) -> dict:
#     entities = {
#         'actors': [],
#         'movies': [],
#         'directors': []
#     }
    
#     # Extract movie names in quotes
#     movie_pattern = r"'([^']+)'"
#     movies = re.findall(movie_pattern, question)
#     entities['movies'].extend(movies)
    
#     # Extract names after "sự tham gia của", "diễn viên"
#     actor_patterns = [
#         r'sự tham gia của\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ\s]+?)(?:\s+và|\s+là|$)',
#         r'của\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ\s]+?)(?:\s+là|$)',
#         r'cả\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ\s]+?)\s+và\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+)',
#     ]
    
#     for pattern in actor_patterns:
#         matches = re.findall(pattern, question)
#         if isinstance(matches[0], tuple) if matches else False:
#             entities['actors'].extend([m for match in matches for m in match if m])
#         else:
#             entities['actors'].extend(matches)
    
#     # Clean duplicates
#     entities['actors'] = list(set([a.strip() for a in entities['actors'] if a.strip()]))
#     entities['movies'] = list(set([m.strip() for m in entities['movies'] if m.strip()]))
    
#     return entities



def  entity_linking_question(question):
    entities = extract_entities(question)
    return entities


# ==================== TEST CASES ====================
if __name__ == "__main__":
    test_cases = [
        # Không dấu + teencode
        "phim bo gia cua tran thanh co hay k",
        "tran thanh va hari won dong phim gi",
        "ai la dao dien phim nha ba nu",
        "phim cua lan ngoc dong vs kieu minh tuan",
        
        # Có dấu đầy đủ
        "Trấn Thành đóng phim gì với Tuấn Trần",
        "Phim Lật Mặt 6 của Lý Hải chiếu khi nào",
        "Mai Tài Phến và Thuận Nguyễn đóng phim nào",
        
        # Câu ngắn
        "cho minh hoi ve phim mai",
        "phim lat mat",
        
        # Nhiều entity
        "Trấn Thành hợp tác với Hari Won và Lan Ngọc trong phim nào",
        
        # Edge cases - Tên có stopword
        "ai dong vai chinh phim bo gia",
        "Kaity Nguyen la ai",
        "phim hai cua Tran Thanh",
        "Kieu Minh Tuan dong phim gi",
        "phim cua Thuy Ngan va Kieu Minh Tuan"
    ]

    print("=" * 100)
    print(f"{'CÂU HỎI':<60} | {'THỰC THỂ NHẬN DIỆN'}")
    print("=" * 100)
    
    for q in test_cases:
        entities = entity_linking_question(q)
        print(entities)
        
    
    print("=" * 100)