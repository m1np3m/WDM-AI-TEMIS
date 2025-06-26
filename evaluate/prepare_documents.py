import json
import pymupdf
import os
import asyncio
from langchain.docstore.document import Document as LangchainDocument
from time import time


def extract_unique_tables_from_qa(qa_path: str):
    """
    Trích xuất các table unique từ file fixed_.json và tạo LangChainDocument
    """
    # Đọc file QA
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Dictionary để lưu trữ table unique
    # Key: (source, table_idx)
    # Value: table information
    unique_tables = {}

    # Duyệt qua tất cả các câu hỏi để tìm table unique
    for qa_item in qa_data:
        source = qa_item["source"]
        table_idx = qa_item["table_idx"]
        page_numbers = qa_item["page_numbers"]
        context = qa_item["context"]

        # Tạo key unique cho table
        table_key = (source, table_idx)

        # Nếu chưa có table này trong dictionary, thêm vào
        if table_key not in unique_tables:
            unique_tables[table_key] = {
                "source": source,
                "table_idx": table_idx,
                "page_numbers": page_numbers,
                "table_content": context,
            }
        else:
            # Cập nhật page_numbers nếu có thêm trang mới
            existing_pages = set(unique_tables[table_key]["page_numbers"])
            new_pages = set(page_numbers)
            all_pages = sorted(list(existing_pages.union(new_pages)))
            unique_tables[table_key]["page_numbers"] = all_pages

    # Tạo danh sách LangChainDocument
    table_documents = []

    for table_key, table_info in unique_tables.items():
        table_document = LangchainDocument(
            page_content=table_info["table_content"],
            metadata={
                "source": table_info["source"],
                "page_numbers": table_info["page_numbers"],
                "is_table": True,
                "source_table_idx": table_info["table_idx"],
            },
        )
        table_documents.append(table_document)

    return table_documents

async def get_detail_chunks(pdf_path):
    """
    Trích xuất các table detail từ file pdf
    """
    doc = pymupdf.open(pdf_path)
    source = os.path.splitext(os.path.basename(pdf_path))[0]

    page_documents = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Tìm và xử lý tables
        tables = page.find_tables()
        for tab in tables:
            # process the content of table 'tab'
            page.add_redact_annot(tab.bbox)  # wrap table in a redaction annotation

        page.apply_redactions()  # erase all table text

        # do text searches and text extractions here
        
        text = page.get_text().strip()
        if not text:
            continue

        page_document = LangchainDocument(
            page_content=text,
            metadata={
                "source": source,
                "page_numbers": [page_num + 1], 
                "is_table": False
            },
        )
        page_documents.append(page_document)

    doc.close()
    return page_documents

async def process_all_pdfs_in_folder(folder_path, qa_path):
    """
    Xử lý tất cả PDF trong folder một cách bất đồng bộ
    Trả về tuple (page_documents, table_documents)
    """
    all_table_documents = []

    # Xử lý table documents đồng bộ (vì đây là từ file JSON)
    table_docs = extract_unique_tables_from_qa(qa_path)
    all_table_documents.extend(table_docs)

    # Tạo danh sách các task bất đồng bộ cho việc xử lý PDF
    pdf_tasks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            # print(f"\n>>> Đang tạo task cho: {pdf_path}")
            task = get_detail_chunks(pdf_path)
            pdf_tasks.append(task)

    # Chạy tất cả các task bất đồng bộ cùng lúc
    print(f"\n>>> Bắt đầu xử lý {len(pdf_tasks)} file PDF bất đồng bộ...")
    page_docs_results = await asyncio.gather(*pdf_tasks)
    
    # Gộp tất cả kết quả page documents
    all_page_documents = []
    for page_docs in page_docs_results:
        all_page_documents.extend(page_docs)

    print(f"\n>>> Hoàn thành! Đã xử lý {len(all_page_documents)} page documents và {len(all_table_documents)} table documents")
    
    return all_page_documents, all_table_documents


async def main():
    """
    Hàm main để test
    """
    folder_path = "data/QA_tables/pdf"
    qa_path = "data/QA_tables/fixed_label_QA.json"
    
    page_docs, table_docs = await process_all_pdfs_in_folder(folder_path, qa_path)
    print(f"Kết quả: {len(page_docs)} page documents, {len(table_docs)} table documents")


if __name__ == "__main__":
    asyncio.run(main())