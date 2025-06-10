import json
import sys

sys.path.append("/home/thangquang/code/WDM-AI-TEMIS")

import os
from typing import TypedDict, List
from src.WDMParser import WDMPDFParser, WDMMergedTable, WDMTable, WDMText, WDMImage

pdf_path = "/home/thangquang/code/WDM-AI-TEMIS/data/pdfs"

pdf_sources = [
    os.path.join(pdf_path, file)
    for file in os.listdir(pdf_path)
    if file.endswith(".pdf")
]

pdf_file_names = [file for file in os.listdir(pdf_path) if file.endswith(".pdf")]


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
            credential_path="/home/thangquang/code/WDM-AI-TEMIS/key_vertex.json"
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
    with open("/home/thangquang/code/WDM-AI-TEMIS/data/full_content.json", "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, indent=4)



if __name__ == "__main__":
    main()
