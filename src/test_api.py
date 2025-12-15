"""
Test script để kiểm tra JSON output của API
"""

import json
import sys
from pathlib import Path

# Import chatbot function
sys.path.insert(0, str(Path(__file__).parent))

try:
    from api import chatbot_answer  # Hoặc tên file API của bạn
    
    # Import để debug
    from src.chatbot.entity_linking_node import entity_linking_graph
    from src.nlp.ner import extract_entity_from_sentences
    
except ImportError as e:
    print(f"Error: Không thể import - {e}")
    sys.exit(1)

# ==================== TEST CASES ====================

test_cases = [
    "Trấn Thành đóng phim gì?",
    "Ai đạo diễn phim Bố Già?",
    "Vợ của Trấn Thành là ai?",
    "Trấn Thành và Hari Won đóng chung phim nào?",
]

def print_json_pretty(data, title=""):
    """In JSON với màu sắc (nếu có colorama)"""
    print("\n" + "="*80)
    if title:
        print(f"📋 {title}")
        print("="*80)
    
    # In JSON với indent
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    print(json_str)
    print("="*80 + "\n")

def test_single_question(question, debug=False):
    """Test 1 câu hỏi"""
    print(f"\n🔍 Testing: {question}")
    
    result = chatbot_answer(question, debug=debug)
    
    print_json_pretty(result, f"Result for: {question}")
    
    # In summary
    print("📊 Summary:")
    print(f"  ├─ Answer: {result['answer'][:100]}...")
    print(f"  ├─ Status: {result['progress']['status']}")
    print(f"  ├─ Time: {result['progress']['elapsed_ms']}ms")
    print(f"  └─ Steps: {len(result['steps'])}")
    
    return result

def test_all_cases():
    """Test tất cả cases"""
    print("\n" + "🚀 "*20)
    print("TESTING ALL CASES")
    print("🚀 "*20 + "\n")
    
    results = []
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] ", end="")
        result = test_single_question(question, debug=False)
        results.append({
            "question": question,
            "answer": result['answer'],
            "time_ms": result['progress']['elapsed_ms'],
            "steps_count": len(result['steps'])
        })
    
    # In bảng tổng kết
    print("\n" + "="*80)
    print("📊 SUMMARY TABLE")
    print("="*80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['question']}")
        print(f"   Answer: {r['answer'][:80]}...")
        print(f"   Time: {r['time_ms']}ms | Steps: {r['steps_count']}")
    
    print("\n" + "="*80 + "\n")

def save_json_to_file(question, filename="test_output.json"):
    """Lưu kết quả ra file JSON"""
    result = chatbot_answer(question, debug=False)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to {filename}")
    print(f"   File size: {Path(filename).stat().st_size} bytes")

# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GraphRAG API JSON output')
    parser.add_argument('--question', '-q', type=str, help='Test single question')
    parser.add_argument('--all', '-a', action='store_true', help='Test all cases')
    parser.add_argument('--save', '-s', type=str, help='Save output to file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.question:
        # Test 1 câu
        result = test_single_question(args.question, debug=args.debug)
        
        if args.save:
            save_json_to_file(args.question, args.save)
    
    elif args.all:
        # Test tất cả
        test_all_cases()
    
    else:
        # Mặc định: test 1 câu đơn giản
        print("No arguments provided. Testing default question...")
        print("Use --help to see all options\n")
        
        default_question = "Trấn Thành đóng phim gì?"
        test_single_question(default_question, debug=True)