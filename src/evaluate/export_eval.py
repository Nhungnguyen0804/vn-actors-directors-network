import json
import pandas as pd
def export_metrics_to_csv(json1, json2, prefix="result"):
    """
    json1, json2: dict dữ liệu JSON (cùng cấu trúc)
    prefix: tiền tố để đặt tên file CSV
    """

    # -------------------------------
    # 1. BẢNG TỔNG QUAN (overview.csv)
    # -------------------------------
    overview_rows = []

    def extract_overview(js):
        return {
            "Mô hình": js.get("model", ""),
            "Độ chính xác (Accuracy)": js.get("accuracy", 0),
            "Tỉ lệ phân tích đúng (Parse Rate)": js.get("parse_rate", 0),
            "Độ chính xác MCQ": js.get("mcq_accuracy", 0),
            "Số câu hỏi": js.get("total_questions", 0),
            "Số câu đúng": js.get("correct_answers", 0),
        }

    overview_rows.append(extract_overview(json1))
    overview_rows.append(extract_overview(json2))

    df_overview = pd.DataFrame(overview_rows)
    df_overview = df_overview.set_index("Mô hình").T  # Set index rồi mới transpose
    df_overview.to_csv(f"{prefix}_1_overview.csv", encoding="utf-8-sig")


    # ----------------------------------------------------------
    # 2. BẢNG TP/FP/TN/FN (confusion.csv)
    # ----------------------------------------------------------
    confusion_rows = []

    def extract_confusion(js):
        matrix = js.get("confusion_matrix", {})
        return {
            "Mô hình": js.get("model", ""),
            "TP – Dự đoán đúng (dương)": matrix.get("TP", 0),
            "FP – Dự đoán sai (dương giả)": matrix.get("FP", 0),
            "TN – Dự đoán đúng (âm)": matrix.get("TN", 0),
            "FN – Dự đoán sai (âm giả)": matrix.get("FN", 0)
        }

    confusion_rows.append(extract_confusion(json1))
    confusion_rows.append(extract_confusion(json2))

    df_confusion = pd.DataFrame(confusion_rows)
    df_confusion = df_confusion.set_index("Mô hình").T  # Set index rồi mới transpose
    df_confusion.to_csv(f"{prefix}_2_confusion.csv", encoding="utf-8-sig")


    # -----------------------------------------------------------------------
    # 3. BẢNG PRECISION / RECALL / F1 THEO NHÃN (giữ nguyên)
    # -----------------------------------------------------------------------
    rows = []

    # def extract_prf(js):
    #     per = js.get("prf", {}).get("per_label", {})
    #     output = []
    #     for lab, item in per.items():
    #         output.append({
    #             "Mô hình": js.get("model", ""),
    #             "Nhãn": lab,
    #             "Độ chính xác (Precision)": item.get("precision", 0),
    #             "Độ bao phủ (Recall)": item.get("recall", 0),
    #             "F1": item.get("f1", 0),
    #             "TP": item.get("tp", 0),
    #             "FP": item.get("fp", 0),
    #             "FN": item.get("fn", 0)
    #         })
    #     return output

    # rows.extend(extract_prf(json1))
    # rows.extend(extract_prf(json2))

    # df_prf = pd.DataFrame(rows)
    # df_prf.to_csv(f"{prefix}_3_prf_labels.csv", index=False, encoding="utf-8-sig")


    # ----------------------------------------------------
    # 4. BẢNG THỐNG KÊ ĐỘ TRỄ (giữ nguyên)
    # ----------------------------------------------------
    def extract_latency_stats(js):
        lat = js.get("latency_stats", {})
        return {
            
            "Số lượng": lat.get("count", 0),
            "Trung bình": lat.get("mean", 0),
            "Nhỏ nhất": lat.get("min", 0),
            "Lớn nhất": lat.get("max", 0),
            "p50": lat.get("p50", 0),
            "p75": lat.get("p75", 0),
            "p90": lat.get("p90", 0),
            "p95": lat.get("p95", 0)
        }

    json1_lat = extract_latency_stats(json1)
    json2_lat = extract_latency_stats(json2)

    df_latency = pd.DataFrame({
        "Chỉ số": json1_lat.keys(),
        "gemini": json1_lat.values(),
        "chatbot": json2_lat.values()
    })

    df_latency.to_csv(f"{prefix}_3_latency.csv", index=False, encoding="utf-8-sig")

    print("Đã tạo xong các file CSV!")

import json

def load_json_to_dict(file_path: str) -> dict:
    """Đọc file JSON từ path và trả về dict."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("File không tồn tại:", file_path)
    except json.JSONDecodeError:
        print("Lỗi định dạng JSON trong file:", file_path)
    return {}



eval_gem_dict = load_json_to_dict("data/eval/gemini_detailed_metrics.json")
eval_chatbot_dict = load_json_to_dict("data/eval/graphrag_detailed_metrics.json")
export_metrics_to_csv(eval_gem_dict, eval_chatbot_dict, prefix="data/eval/compare")
