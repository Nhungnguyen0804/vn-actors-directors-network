#RUN:python.exe -m src.data_prep.wiki_enrichment
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import os
# Load XML dump → wiki_dict
# đọc file XML wiki dump 

# mỗi bài viết trong thẻ <page> </page> 

    # Trả về danh sách [(title, text), ...]

# dùng iterparse (đọc từng phần nhỏ)


def log_time(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_pages_from_xml(xml_path):
    """
    Đọc từng trang trong file XML (Wikipedia dump).
    Trả về danh sách [(title, text), ...]
    """
    # danh sách lưu kết quả
    pages = []  
    # iterparse => đọc
    context = ET.iterparse(xml_path, events=("start", "end"))

    # Lấy namespace (ở thẻ <mediawiki>)
    event, root = next(context)
    
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0].strip("{")
        namespace_prefix = f"{{{namespace}}}"
    else:
        namespace_prefix = ""

    
    # Duyệt từng p.tử trg XML
    for event, element in context:
        if event == "end" and element.tag == namespace_prefix + "page":
            title_element = element.find(f"./{namespace_prefix}title")
            text_element = element.find(f".//{namespace_prefix}text")
            
            # Lấy nd trg các thẻ, nếu k có thì None 
            if title_element is not None:
                title = title_element.text
            else:
                title = None

            if text_element is not None:
                text = text_element.text
            else:
                text = ''

            pages.append((title, text))  # thêm vào danh sách

            element.clear()  # xóa khỏi bộ nhớ

    return pages


def load_wiki_dump(xml_path):
    """
    Input: file XML dump
    Output: dict {title: wikitext}
    """
    all_pages = read_pages_from_xml(xml_path)
    wiki_dict = {title: text for title, text in all_pages if title}
    return wiki_dict

# Lấy wikitext theo title (có xử lý redirect)

def get_wikitext(title, wiki_dict, depth=0):
    """
    Input: title, wiki_dict
    Output: raw wikitext (string) or ""
    """

    if title not in wiki_dict:
        return ""

    wikitext = wiki_dict[title]

    # Chống loop redirect
    if depth > 5:
        return ""

    # Redirect kiểu "#ĐỔI" hoặc "#redirect"
    if wikitext.lower().startswith("#đổi") or wikitext.lower().startswith("#redirect"):
        start = wikitext.find("[[")
        end = wikitext.find("]]")
        if start != -1 and end != -1:
            new_title = wikitext[start+2:end].strip()
            return get_wikitext(new_title, wiki_dict, depth + 1)

    return wikitext


# Tách SUMMARY từ wikitext

def extract_summary(wikitext):
    """
    Input: raw wikitext
    Output: first summary paragraph
    """

    if not wikitext:
        return ""

    # Xóa template {{...}}
    cleaned = re.sub(r"\{\{[^{}]+\}\}", "", wikitext)

    # Lấy block trước section "=="
    first_block = cleaned.split("==")[0]

    # Tách dòng + lấy dòng đầu tiên có nội dung
    lines = [l.strip() for l in first_block.split("\n") if l.strip()]

    return lines[0] if lines else ""

# Convert wikitext → plain text dùng regex đơn giản

def extract_clean_text(wikitext):
    """
    Input: wikitext
    Output: plain text (bỏ markup cơ bản)
    """

    if not wikitext:
        return ""

    text = wikitext

    # Bỏ template {{ }}
    text = re.sub(r"\{\{[^{}]+\}\}", "", text)

    # Bỏ link nội [[A|B]] → B
    text = re.sub(r"\[\[[^\|\]]+\|([^\]]+)\]\]", r"\1", text)

    # Bỏ link dạng [[A]] → A
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Bỏ headings == ==
    text = re.sub(r"==+[^=]+==+", "", text)

    # Bỏ định dạng '''bold'''
    text = text.replace("'''", "")

    # Bỏ định dạng ''italic''
    text = text.replace("''", "")

    return text.strip()

# Crawl 1 entity

def crawl_single_entity(name, wiki_dict):
    """
    Input: entity name, wiki_dict
    Output: dict {summary, text}
    """
    # ------time---------------------
    start = time.time()
    log_time(f"Bắt đầu crawl: {name}")
    # ------time---------------------

    wikitext = get_wikitext(name, wiki_dict)

    if not wikitext:
        return None

    summary = extract_summary(wikitext)
    clean_text = extract_clean_text(wikitext)
    # ------time---------------------
    runtime = time.time() - start
    log_time(f"Hoàn tất: {name} | {runtime:.2f}s")
    # ------time---------------------
    return {
        "summary": summary,
        "text": clean_text,
        "raw_wikitext": wikitext
    }


# Ghi enrichment_data ra file JSON
def save_entity_jsonl(name, data, output_path="data/wiki_enrichment.jsonl"):
    data_with_name = {"name": name, **data}
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_with_name, ensure_ascii=False) + "\n")

def save_enrichment_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Crawl nhiều entity

def crawl_multiple_entities(entity_list, limit, wiki_dict,output_path="data/wiki_enrichment.jsonl"):
    """
    Input: list tên entity, số lượng tối đa, wiki_dict
    Output: dict enrichment_data
    """

    enrichment_data = {}
    # ------time---------------------
    log_time(f"Bắt đầu crawl {limit} entities...")
    # ------time---------------------

    # Clear file đầu tiên
    if os.path.exists(output_path):
        os.remove(output_path)
    for idx, name in enumerate(entity_list[:limit], 1):
        log_time(f"[{idx}/{limit}] Đang crawl: {name}")
        result = crawl_single_entity(name, wiki_dict)
        if result:
            save_entity_jsonl(name, result, output_path)
            del result  # giảm RAM
        

    
    
    # ------time---------------------
    log_time(f"Hoàn tất crawl {limit} entities. Dữ liệu lưu tại: {output_path}")
    # ------time---------------------


#test

from src.constant import XML_FILE, BIPARTITE_JSON
from src.data_prep.load_graph import load_graph,load_bipartite_graph_and_nodes
wiki_dict = load_wiki_dump(XML_FILE)

print('==================================')
B = load_graph(BIPARTITE_JSON)
B, person_list, film_list = load_bipartite_graph_and_nodes(B)

# crawl all person
ALL_PERSON = len(person_list)
crawl_multiple_entities(person_list, limit=ALL_PERSON, wiki_dict=wiki_dict)



