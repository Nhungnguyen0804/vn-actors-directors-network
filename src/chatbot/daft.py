import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import re

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
# Thêm đường dẫn cha để import các module khác
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import các module graph đã viết trước đó
    from load_graph import load_graphs
    from chatbot.entity_linking_node import extract_entities, entity_linking_question, get_best_match
    from chatbot.extract_entities_from_question import VIETNAMESE_STOPWORDS
    from chatbot.graph_query import (
        graph_query_movies_by_actor, graph_query_actors_of_movie,
        graph_query_common_movies, graph_query_collaborations,
        graph_query_node_info, get_node_name, get_node_type,
        graph_query_actor_via_movie, graph_query_movie_via_actor,
        graph_query_actor_via_collaboration, graph_query_indirect_collaboration,
        graph_query_movie_chain
    )
except ImportError as e:
    print(f"⚠️  Lỗi Import: Không tìm thấy các module phụ thuộc. {e}")
    # (Để code chạy được demo, ta cần các module trên. Nếu thiếu, code sẽ báo lỗi ở đoạn này)


# ==================== 1. QUẢN LÝ BỘ NHỚ (CONTEXT MEMORY) ====================

class ConversationContext:
    """Lớp lưu trữ ngữ cảnh hội thoại để chatbot 'nhớ' câu trước"""
    def __init__(self):
        self.last_entity_name = None
        self.last_entity_id = None
        self.last_entity_type = None
        self.last_intent = None

    def update(self, name, node_id, node_type, intent):
        self.last_entity_name = name
        self.last_entity_id = node_id
        self.last_entity_type = node_type
        self.last_intent = intent
        
    def get_context(self):
        return self.last_entity_name, self.last_entity_id, self.last_entity_type

# Khởi tạo bộ nhớ toàn cục
global_context = ConversationContext()


# ==================== 2. XỬ LÝ INTENT & LOGIC THÔNG MINH ====================

def detect_intent(question):
    """Phát hiện ý định người dùng (Có thêm các intent multi-hop)"""
    q = question.lower()
    
    # Định nghĩa Intent và từ khóa
    rules = [
        ('actor_movies', ['dong phim', 'phim nao', 'tham gia']),
        ('movie_actors', ['dien vien', 'ai dong', 'cast', 'nhan vat']),
        ('common_movies', ['phim chung', 'dong chung', 'cung dong']),
        ('info', ['la ai', 'thong tin', 'tieu su', 'sinh nam']),
        ('collaboration', ['hop tac', 'lam viec cung']),
        ('actor_via_movie', ['ngoai', 'khac', 'con ai']), # Multi-hop
        ('indirect_collaboration', ['cau noi', 'trung gian', 'lien ket']) # Multi-hop
    ]
    
    # Logic chấm điểm đơn giản
    best_intent = 'unknown'
    max_score = 0
    
    for intent, keywords in rules:
        score = 0
        for k in keywords:
            if k in q: score += 1
        if score > max_score:
            max_score = score
            best_intent = intent
            
    return {'intent': best_intent, 'confidence': max_score}

def refine_intent_smart(intent, entity_type):
    """
    🧠 SMART FIX: Tự sửa Intent dựa trên loại Entity.
    VD: Hỏi "Diễn viên của Trấn Thành" (sai logic) -> Sửa thành "Phim của Trấn Thành"
    """
    if not entity_type: return intent

    # Nếu hỏi tìm diễn viên (movie_actors) mà entity lại là Người (person)
    # -> Ý người dùng là tìm phim người đó đóng.
    if intent == 'movie_actors' and entity_type == 'person':
        return 'actor_movies'

    # Nếu hỏi tìm phim (actor_movies) mà entity lại là Phim (movie)
    # -> Ý người dùng là tìm diễn viên trong phim đó.
    if intent == 'actor_movies' and entity_type == 'movie':
        return 'movie_actors'

    return intent


# ==================== 3. ROUTER TRUY VẤN GRAPH ====================

def route_graph_query(entities_dict, G_bi, G_collab, intent):
    """Điều hướng truy vấn đến hàm xử lý tương ứng"""
    intent_type = intent
    ids = list(entities_dict.values())
    names = list(entities_dict.keys())
    
    data = None
    msg = ""

    try:
        if intent_type == 'actor_movies' and ids:
            data = graph_query_movies_by_actor(G_bi, ids[0], get_names=True)
            msg = f"Các bộ phim có sự tham gia của {names[0]}"

        elif intent_type == 'movie_actors' and ids:
            data = graph_query_actors_of_movie(G_bi, ids[0], get_names=True)
            msg = f"Danh sách diễn viên trong phim {names[0]}"

        elif intent_type == 'info' and ids:
            data = graph_query_node_info(G_bi, ids[0])
            msg = f"Thông tin về {names[0]}"
            
        elif intent_type == 'common_movies' and len(ids) >= 2:
            data = graph_query_common_movies(G_collab, ids[0], ids[1])
            msg = f"Phim chung giữa {names[0]} và {names[1]}"

        elif intent_type == 'collaboration' and ids:
            data = graph_query_collaborations(G_collab, ids[0], get_names=True)
            msg = f"Các diễn viên đã hợp tác với {names[0]}"
            
        # ... (Thêm các case multi-hop khác nếu cần)
        else:
            return {'status': 'error', 'message': "Không đủ thông tin hoặc Intent chưa hỗ trợ."}

        return {'status': 'success', 'data': data, 'message': msg, 'entity_name': names[0]}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ==================== 4. XỬ LÝ LLM & CÁ TÍNH (PERSONA) ====================

def load_llm():
    print("📥 Đang tải model Qwen2.5-0.5B (Tiny but Smart)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available(): model.to("cpu")
        print("✅ Tải model thành công!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return None, None

def format_data_for_llm(data, intent):
    """Chuẩn hóa dữ liệu đầu vào cho LLM"""
    if not data: return "Không có dữ liệu."
    
    if intent == 'info' and isinstance(data, dict):
        # Data-to-Text chuẩn bị cho Info
        info = data.get('attributes', {}).get('info', data)
        facts = []
        mapping = {'name': 'Tên', 'birth_date': 'Sinh nhật', 'spouse': 'Vợ/Chồng', 'occupation': 'Nghề nghiệp'}
        for k, v in info.items():
            if k in mapping and v:
                facts.append(f"- {mapping[k]}: {str(v).replace('*','')}")
        return "\n".join(facts)
        
    if isinstance(data, list):
        return ", ".join(data)
        
    return str(data)

def llm_paraphrase_with_personality(model_pack, raw_data, question, intent):
    """
    🎨 Sinh câu trả lời có CÁ TÍNH và EMOJI
    """
    model, tokenizer = model_pack
    if not model: return f"🤖 {raw_data}" # Fallback nếu không có model

    # Chọn giọng điệu (Tone)
    tone = "thân thiện, hữu ích"
    emoji_instr = "Sử dụng các emoji phù hợp (🎬, 🌟, 🎭, 📺) để sinh động."
    
    if intent == 'info':
        task = "Viết đoạn giới thiệu ngắn gọn, hấp dẫn về nhân vật."
    elif intent == 'common_movies':
        task = "Liệt kê các phim chung một cách hào hứng."
    else:
        task = "Trả lời ngắn gọn, trực tiếp vào vấn đề."

    system_prompt = f"""Bạn là Trợ lý Điện ảnh Việt Nam thông minh 🤖.
    Nhiệm vụ: {task}
    
    QUY TẮC CỐT TỬ:
    1. Dựa CHÍNH XÁC vào dữ liệu cung cấp. KHÔNG bịa đặt thông tin ngoài.
    2. {emoji_instr}
    3. Giọng văn {tone}.
    """
    
    user_prompt = f"""DỮ LIỆU CUNG CẤP:
    {raw_data}
    
    CÂU HỎI NGƯỜI DÙNG:
    "{question}"
    
    HÃY TRẢ LỜI:"""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    # Generate
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        temperature=0.3, # Thấp để an toàn
        repetition_penalty=1.1,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Lọc lấy phần trả lời của assistant
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
        
    return response


# ==================== 5. PIPELINE CHÍNH (THE BRAIN) ====================

def get_answer(question, G_bi, G_collab, model_pack):
    print(f"\n💬 USER: {question}")
    
    # B1: Extract Entities
    entities = extract_entities(question)
    
    # --- XỬ LÝ BỘ NHỚ (CONTEXT) ---
    is_using_context = False
    if not entities:
        last_name, last_id, last_type = global_context.get_context()
        if last_name:
            print(f"💡 [MEMORY] Không thấy tên riêng, đang hiểu là hỏi về: {last_name}")
            entities = [last_name]
            is_using_context = True
        else:
            return "🤖 Xin lỗi, tôi chưa biết bạn muốn hỏi về ai. Hãy nhập tên cụ thể nhé!"

    # B2: Linking Entity
    # (Ở đây giả lập hàm link_entities trả về dict {name: id})
    # Trong code thực tế của bạn, hãy dùng hàm link_entities đã viết
    linked = entity_linking_question(question, graph=None, threshold=70, debug=False)
    
    if not linked:
        return f"🤖 Xin lỗi, tôi không tìm thấy thông tin về '{entities[0]}' trong dữ liệu."

    # Lấy thông tin entity chính để xử lý logic
    primary_name = list(linked.keys())[0]
    primary_id = list(linked.values())[0]
    primary_type = get_node_type(G_bi, primary_id) # 'person' hoặc 'movie'

    # B3: Detect Intent & Smart Fix
    intent_res = detect_intent(question)
    raw_intent = intent_res['intent']
    
    # Sửa lỗi logic (VD: hỏi diễn viên của Trấn Thành -> phim của Trấn Thành)
    final_intent = refine_intent_smart(raw_intent, primary_type)
    
    if final_intent != raw_intent:
        print(f"🔧 [SMART FIX] Đổi ý định '{raw_intent}' -> '{final_intent}' cho hợp lý.")

    # Cập nhật bộ nhớ
    global_context.update(primary_name, primary_id, primary_type, final_intent)

    # B4: Query Graph
    graph_res = route_graph_query(linked, G_bi, G_collab, final_intent)
    
    if graph_res['status'] == 'error':
        return "🤖 Có lỗi kỹ thuật khi truy xuất dữ liệu."

    # B5: Format & LLM Generate
    raw_data_str = format_data_for_llm(graph_res['data'], final_intent)
    
    # Nếu không có dữ liệu
    if not graph_res['data'] or raw_data_str == "Không có dữ liệu.":
        return f"🤖 Tiếc quá, tôi không tìm thấy thông tin nào cho câu hỏi này trong hệ thống."

    final_answer = llm_paraphrase_with_personality(
        model_pack, raw_data_str, question, final_intent
    )
    
    return final_answer


# ==================== MAIN RUN ====================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎬 CHATBOT ĐIỆN ẢNH THÔNG MINH (EMOJI VERSION)")
    print("="*50)
    
    # 1. Load Resources
    try:
        G_collab, G_bi = load_graphs()
        print("✅ Graph Loaded.")
        llm_pack = load_llm()
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        exit()

    print("\n🚀 Chatbot đã sẵn sàng! (Gõ 'exit' để thoát)")
    
    # 2. Loop hội thoại
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Tạm biệt! Hẹn gặp lại.")
            break
            
        # Gọi Pipeline
        response = get_answer(user_input, G_bi, G_collab, llm_pack)
        print(f"🤖 Bot: {response}")