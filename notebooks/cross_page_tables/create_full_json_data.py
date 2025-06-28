import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# sys.path.append("/home/thangquang/code/WDM-AI-TEMIS")
sys.path.append("/WDM-AI-TEMIS")

import os
from typing import TypedDict, List
from src.WDMParser import WDMPDFParser, WDMMergedTable, WDMTable, WDMText, WDMImage

# pdf_path = "/home/thangquang/code/WDM-AI-TEMIS/data/pdfs"
pdf_path = "/WDM-AI-TEMIS/data/QA_tables/pdf_rerank"

pdf_sources = [
    os.path.join(pdf_path, file)
    for file in os.listdir(pdf_path)
    if file.endswith(".pdf")
]

pdf_file_names = [file for file in os.listdir(pdf_path) if file.endswith(".pdf")]

def convert_to_serializable(obj):
    # Trường hợp object có thuộc tính __dict__ (object đơn giản)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    # Nếu là danh sách các object
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    # Nếu là dict lồng
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return str(obj)  # fallback nếu là kiểu lạ

class JsonInstance(TypedDict):
    source: str
    tables: List[WDMMergedTable]
    texts: List[WDMText]
    images: List[WDMImage]

def main():
    full_json_data = []
    for pdf_source, pdf_file_name in zip(pdf_sources, pdf_file_names):
        parser = WDMPDFParser(
            file_path=pdf_source,
            debug=True,
            debug_level=1,
            # credential_path="/home/thangquang/code/WDM-AI-TEMIS/key_vertex.json"
            credential_path = "/WDM-AI-TEMIS/notebooks/dataset_loading/key.json"
        )
        texts = parser.extract_text()
        tables = parser.extract_tables(merge_span_tables=True, enrich=True)
        images = parser.extract_images()
        
        full_json_data.append(JsonInstance(
            source=pdf_file_name,
            tables=tables,
            texts=texts,
            images=images,
        ))
        # break
    # with open("/home/thangquang/code/WDM-AI-TEMIS/data/full_content.json", "w", encoding="utf-8") as f:
    #     json.dump(full_json_data, f, indent=4)
    with open("/WDM-AI-TEMIS/data/QA_tables/data_rerank.json", "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(full_json_data), f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    main()
