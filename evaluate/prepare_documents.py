import json
import pymupdf
import os
import asyncio
from typing import Optional, List, Tuple
from langchain.docstore.document import Document as LangchainDocument
from time import time
from loguru import logger
from src.WDMParser import WDMPDFParser

async def extract_documents_with_wdmparser(
    pdf_folder_path: str,
    credential_path: Optional[str] = None,
    merge_span_tables: bool = True,
    enrich: bool = False,
    extract_text: bool = True,
    extract_images: bool = False,
    pages: Optional[List[int]] = None
):
    """
    Sử dụng WDMParser để trích xuất documents từ folder PDF
    """
    settings = {
        'credential_path': credential_path,
        'debug': True,
        'max_concurrent_files': 3,
        'max_memory_mb': 8192,
        'batch_size': 5
    }
    parser = WDMPDFParser(settings=settings)
    pdf_files = []
    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, filename)
            pdf_files.append(pdf_path)
    if not pdf_files:
        logger.warning(f"Không tìm thấy file PDF nào trong folder: {pdf_folder_path}")
        return [], [], []
    logger.info(f"Tìm thấy {len(pdf_files)} file PDF để xử lý")
    results = await parser.process_documents(
        pdf_documents=pdf_files,
        pages=pages,
        merge_span_tables=merge_span_tables,
        enrich=enrich,
        extract_text=extract_text,
        extract_images=extract_images,
        image_mode="summary"
    )
    if isinstance(results, tuple):
        results = results[0]
    all_page_documents = []
    all_table_documents = []
    all_image_documents = []
    for pdf_path, documents in results.items():
        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "text":
                all_page_documents.append(doc)
            elif doc_type == "table":
                all_table_documents.append(doc)
            elif doc_type == "image":
                all_image_documents.append(doc)
    logger.info(f"Extraction hoàn thành: {len(all_page_documents)} text docs, "
               f"{len(all_table_documents)} table docs, {len(all_image_documents)} image docs")
    return all_page_documents, all_table_documents, all_image_documents

async def process_pdfs_with_wdmparser(pdf_folder_path: str, credential_path: Optional[str] = None):
    """
    Main function để process PDFs với WDMParser và trả về combined documents
    """
    page_docs, table_docs, image_docs = await extract_documents_with_wdmparser(
        pdf_folder_path=pdf_folder_path,
        credential_path=credential_path,
        merge_span_tables=True,
        enrich=False,
        extract_text=True,
        extract_images=False,
        pages=None
    )
    logger.info(f"Final result: {len(page_docs)} page documents, {len(table_docs)} table documents")
    return page_docs, table_docs

def extract_unique_tables_from_qa_legacy(qa_path: str):
    """
    [LEGACY] Trích xuất các table unique từ file fixed_.json và tạo LangChainDocument
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    unique_tables = {}
    for qa_item in qa_data:
        source = qa_item["source"]
        table_idx = qa_item["table_idx"]
        page_numbers = qa_item["page_numbers"]
        context = qa_item["context"]
        table_key = (source, table_idx)
        if table_key not in unique_tables:
            unique_tables[table_key] = {
                "source": source,
                "table_idx": table_idx,
                "page_numbers": page_numbers,
                "table_content": context,
            }
        else:
            existing_pages = set(unique_tables[table_key]["page_numbers"])
            new_pages = set(page_numbers)
            all_pages = sorted(list(existing_pages.union(new_pages)))
            unique_tables[table_key]["page_numbers"] = all_pages
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

async def get_detail_chunks_legacy(pdf_path):
    """
    [LEGACY] Trích xuất các table detail từ file pdf - có lỗi linter đã fix
    """
    doc = pymupdf.open(pdf_path)
    source = os.path.splitext(os.path.basename(pdf_path))[0]
    page_documents = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        try:
            tables = page.find_tables(strategy="lines_strict").tables  # type: ignore
            if tables:
                for tab in tables:
                    page.add_redact_annot(tab.bbox)
                page.apply_redactions()  # type: ignore
        except Exception as e:
            logger.warning(f"Error processing tables on page {page_num + 1}: {e}")
        try:
            text = page.get_text().strip()  # type: ignore
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            text = ""
        if not text:
            logger.warning(f"Source: {source}, Page {page_num + 1} is empty")
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

async def process_all_pdfs_in_folder_legacy(folder_path, qa_path):
    """
    [LEGACY] Xử lý tất cả PDF trong folder một cách bất đồng bộ
    Trả về tuple (page_documents, table_documents)
    """
    all_table_documents = []
    table_docs = extract_unique_tables_from_qa_legacy(qa_path)
    all_table_documents.extend(table_docs)
    pdf_tasks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            task = get_detail_chunks_legacy(pdf_path)
            pdf_tasks.append(task)
    print(f"\n>>> Bắt đầu xử lý {len(pdf_tasks)} file PDF ...")
    page_docs_results = await asyncio.gather(*pdf_tasks)
    all_page_documents = []
    for page_docs in page_docs_results:
        all_page_documents.extend(page_docs)
    print(f"\n>>> Hoàn thành! Đã xử lý {len(all_page_documents)} page documents và {len(all_table_documents)} table documents")
    return all_page_documents, all_table_documents

async def main():
    """
    Hàm main để test - sử dụng WDMParser
    """
    folder_path = "data/QA_tables/pdf"
    credential_path = None
    page_docs, table_docs = await process_pdfs_with_wdmparser(
        pdf_folder_path=folder_path,
        credential_path=credential_path
    )
    print(f"Kết quả WDMParser: {len(page_docs)} page documents, {len(table_docs)} table documents")

async def main_legacy():
    """
    Hàm main legacy để test - sử dụng cách cũ
    """
    folder_path = "data/QA_tables/pdf"
    qa_path = "data/QA_tables/fixed_label_QA.json"
    page_docs, table_docs = await process_all_pdfs_in_folder_legacy(folder_path, qa_path)
    print(f"Kết quả Legacy: {len(page_docs)} page documents, {len(table_docs)} table documents")

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main_legacy())