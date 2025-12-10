import json
from pathlib import Path
import shutil
import json
from pathlib import Path
import shutil

def strip_empty_info_from_nodes(inpath, outpath=None, make_backup=True, remove_all_empty_values=False):
    """
    Xoá field 'info' trong các node có type person nếu field 'info' rỗng.
    - inpath: path tới file json gốc
    - outpath: nếu None sẽ tạo file cùng thư mục với tiền tố "cleaned_"
    - make_backup: nếu True sẽ tạo bản sao lưu (inpath -> inpath + ".bak")
    - remove_all_empty_values: nếu True, xoá 'info' khi nó là None, empty dict, empty list, empty string.
      Nếu False, chỉ xoá khi info là dict rỗng {} (hành vi như bản gốc của bạn).
    Trả về Path tới file output.
    """
    p = Path(inpath)
    if outpath is None:
        outpath = p.with_name("cleaned_" + p.name)
    outpath = Path(outpath)

    # Tạo backup nếu cần
    if make_backup:
        bak = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, bak)

    # Đọc JSON
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy nodes (bảo đảm là list)
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Expected 'nodes' to be a list in the input JSON")

    removed_count = 0
    # Xu ly trên bản copy để tránh side-effects bất ngờ
    new_nodes = []
    for node in nodes:
        # Nếu node không phải dict thì giữ nguyên
        if not isinstance(node, dict):
            new_nodes.append(node)
            continue

        import copy
        node_copy = copy.deepcopy(node)  # Tốt hơn dict(node)
        node_type = str(node_copy.get("type", "")).lower()
        if node_type == "person":
            info = node_copy.get("info", None)

            should_remove = False
            if remove_all_empty_values:
                # Xoá nếu None, {}, [], '' hoặc string chỉ chứa whitespace
                if info is None:
                    should_remove = True
                elif isinstance(info, dict) and len(info) == 0:
                    should_remove = True
                elif isinstance(info, (list, tuple)) and len(info) == 0:
                    should_remove = True
                elif isinstance(info, str) and info.strip() == "":
                    should_remove = True
            else:
                # Hành vi gốc: chỉ xoá khi đúng là dict rỗng
                if isinstance(info, dict) and len(info) == 0:
                    should_remove = True

            if should_remove and "info" in node_copy:
                del node_copy["info"]
                removed_count += 1

        new_nodes.append(node_copy)

    # Giữ nguyên mọi top-level keys, chỉ thay nodes
    output_data = dict(data)  # shallow copy
    output_data["nodes"] = new_nodes

    # Ghi file mới (kèm tạo thư mục đích nếu cần)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Đã xoá {removed_count} field 'info' rỗng (person nodes).")
    print(f"✓ File sạch đã lưu: {outpath}")
    if make_backup:
        print(f"✓ Backup file: {bak}")

    return outpath

# CHẠY NGAY cho 2 file của bạn
if __name__ == "__main__":
    # File 1: Collaboration graph
    print("=" * 50)
    print("Xử lý file: vn_film_collaboration_graph.json")
    strip_empty_info_from_nodes("data/vn_film_collaboration_graph.json","data/cleaned_vn_film_collaboration_graph.json")
    print("Xử lý file: vn_bipartite_graph.json")
    strip_empty_info_from_nodes("data/vn_bipartite_graph.json","data/cleaned_vn_bipartite_graph.json")
  
    print("\n" + "=" * 50)
    print("HOÀN THÀNH! Cả 2 file đã được làm sạch.")



import json
from pathlib import Path

def test_cleaned_file(original_path, cleaned_path):
    """
    Kiểm tra xem file đã làm sạch đúng chưa
    """
    print(f"\nTESTING: {Path(cleaned_path).name}")
    print("=" * 60)
    
    # Đọc 2 file
    with open(original_path, 'r', encoding='utf-8') as f:
        original = json.load(f)
    
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        cleaned = json.load(f)
    
    # TEST 1: Số lượng nodes giữ nguyên
    orig_nodes = original.get('nodes', [])
    clean_nodes = cleaned.get('nodes', [])
    
    print(f"Số lượng nodes: {len(orig_nodes)} → {len(clean_nodes)}")
    assert len(orig_nodes) == len(clean_nodes), "ERROR: Số lượng nodes thay đổi!"
    
    # TEST 2: Số lượng edges giữ nguyên
    orig_edges = original.get('edges', [])
    clean_edges = cleaned.get('edges', [])
    
    print(f"Số lượng edges: {len(orig_edges)} → {len(clean_edges)}")
    assert len(orig_edges) == len(clean_edges), "ERROR: Số lượng edges thay đổi!"
    
    # TEST 3: Kiểm tra nodes person có info rỗng
    empty_info_nodes = []
    for node in clean_nodes:
        if node.get('type') == 'person':
            # Có 2 trường hợp đúng:
            # 1. Không có key 'info' (đã bị xoá)
            # 2. Có 'info' nhưng không rỗng
            if 'info' in node:
                info = node.get('info')
                if isinstance(info, dict) and len(info) == 0:
                    empty_info_nodes.append(node['id'])
    
    if empty_info_nodes:
        print(f"VẪN CÒN {len(empty_info_nodes)} nodes có info rỗng:")
        for node_id in empty_info_nodes[:5]:  # Hiển thị 5 cái đầu
            print(f"   - {node_id}")
        if len(empty_info_nodes) > 5:
            print(f"   ... và {len(empty_info_nodes) - 5} nodes khác")
    else:
        print("KHÔNG CÒN nodes person nào có info rỗng")
    
    # TEST 4: Kiểm tra nodes khác (phim) vẫn có info
    film_nodes_with_info = []
    film_nodes_without_info = []
    
    for node in clean_nodes:
        if node.get('type') == 'film' or 'film' in str(node.get('id', '')).lower():
            if 'info' in node:
                film_nodes_with_info.append(node['id'])
            else:
                film_nodes_without_info.append(node['id'])
    
    print(f"✓ Film nodes có info: {len(film_nodes_with_info)}")
    if film_nodes_without_info:
        print(f"Film nodes KHÔNG có info: {len(film_nodes_without_info)}")
        for fid in film_nodes_without_info[:3]:
            print(f"   - {fid}")
    
    # TEST 5: Kiểm tra 10 nodes đầu tiên
    print("\nSample check - 10 nodes đầu tiên:")
    for i, (orig_node, clean_node) in enumerate(zip(orig_nodes[:10], clean_nodes[:10])):
        orig_has_info = 'info' in orig_node
        clean_has_info = 'info' in clean_node
        
        if orig_node.get('type') == 'person' and orig_node.get('info') == {}:
            # Node này đáng lẽ phải bị xoá info
            if not clean_has_info:
                status = "ĐÚNG: Đã xoá info rỗng"
            else:
                status = "SAI: Vẫn còn info rỗng"
        else:
            # Node này nên giữ nguyên info
            if orig_has_info == clean_has_info:
                status = "ĐÚNG: Giữ nguyên info"
            else:
                status = "SAI: Thay đổi info không đúng"
        
        print(f"  {i+1:2d}. {orig_node.get('id', 'N/A')[:30]:30} | {status}")
    
    # TEST 6: Kiểm tra integrity của edges
    print("\nEdge integrity check:")
    
    # Lấy danh sách ID nodes trong file cleaned
    clean_node_ids = {node.get('id') for node in clean_nodes}
    
    broken_edges = []
    for edge in clean_edges:
        source = edge.get('source')
        target = edge.get('target')
        
        if source not in clean_node_ids or target not in clean_node_ids:
            broken_edges.append((source, target))
    
    if broken_edges:
        print(f"CÓ {len(broken_edges)} edges bị broken (tham chiếu node không tồn tại)")
        for s, t in broken_edges[:3]:
            print(f"   - {s} → {t}")
    else:
        print("✓ TẤT CẢ edges đều tham chiếu đúng đến nodes tồn tại")
    
    # TEST 7: So sánh dung lượng file
    orig_size = Path(original_path).stat().st_size / 1024  # KB
    clean_size = Path(cleaned_path).stat().st_size / 1024
    
    reduction = ((orig_size - clean_size) / orig_size) * 100
    
    print(f"\nDung lượng file: {orig_size:.1f}KB → {clean_size:.1f}KB")
    print(f"   Giảm: {reduction:.1f}%")
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ TEST: ", end="")
    
    if empty_info_nodes or broken_edges:
        print("CÓ VẤN ĐỀ - Cần kiểm tra lại!")
        return False
    else:
        print("THÀNH CÔNG - File đã làm sạch đúng!")
        return True

# =============================================
# CHẠY TEST CHO CẢ 2 FILE
# =============================================

if __name__ == "__main__":
    print("🧪 BỘ TEST KIỂM TRA FILE LÀM SẠCH")
    print("=" * 60)
    
    # Danh sách file cần test
    files_to_test = [
        ("data/vn_film_collaboration_graph.json", "data/cleaned_vn_film_collaboration_graph.json"),
      ("data/vn_bipartite_graph.json", "data/cleaned_vn_bipartite_graph.json"),
    ]
    
    all_passed = True
    
    for orig_file, clean_file in files_to_test:
        # Kiểm tra file tồn tại
        if not Path(clean_file).exists():
            print(f"File {clean_file} không tồn tại! Chạy script làm sạch trước.")
            continue
            
        passed = test_cleaned_file(orig_file, clean_file)
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("TẤT CẢ TEST ĐÃ PASSED! Files đã được làm sạch đúng.")
    else:
        print("CÓ TEST FAILED! Cần kiểm tra lại.")

    

    # Mở Python shell và chạy lệnh này:
import json

# Đọc file đã làm sạch
with open("data/cleaned_vn_film_collaboration_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Tìm tất cả nodes person
person_nodes = [n for n in data["nodes"] if n.get("type") == "person"]

# 2. Đếm số có info rỗng
empty_info = [n for n in person_nodes if "info" in n and n["info"] == {}]
print(f"Tổng person nodes: {len(person_nodes)}")
print(f"Số có info rỗng: {len(empty_info)}")
print(f"Ví dụ nodes có info rỗng: {[n['id'] for n in empty_info[:3]] if empty_info else 'Không có'}")

# 3. Kiểm tra node bình thường
normal_nodes = [n for n in person_nodes if "info" in n and n["info"] != {}]
print(f"\nVí dụ node có info đầy đủ:")
for node in normal_nodes[:2]:
    print(f"  {node['id']}: {len(node['info'])} keys")

# 4. Kiểm tra node đã xoá info
deleted_info_nodes = [n for n in person_nodes if "info" not in n]
print(f"\nSố node đã xoá info: {len(deleted_info_nodes)}")
print(f"Ví dụ: {[n['id'] for n in deleted_info_nodes[:3]] if deleted_info_nodes else 'Không có'}")


from pathlib import Path
import json

def check_edge_integrity(cleaned_path):
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    node_ids = {str(n.get('id')) for n in nodes}
    edges = data.get('edges', [])

    broken = []
    for e in edges:
        s = str(e.get('source'))
        t = str(e.get('target'))
        if s not in node_ids or t not in node_ids:
            broken.append((s, t))
    return broken

# Sử dụng:
broken_edges = check_edge_integrity("data/cleaned_vn_film_collaboration_graph.json")
if broken_edges:
    print("Broken edges:", broken_edges[:5])
else:
    print("All edges ok")
