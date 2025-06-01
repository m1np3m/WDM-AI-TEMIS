from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.docstore.document import Document as LangchainDocument
import os
import json
import fitz  
data_dir = os.getenv("DATA_DIR", "data")
def summarize_table():
    summary_documents = []
    tables_sources = json.load(open(f"{data_dir}/pdf/final_tables.json"))
    sources = list(tables_sources.keys())
    for source in sources:
        for table in tables_sources[source]:
            summary_document = LangchainDocument(
                # summary =summary_fn(table['table_content']),
                page_content=table['table_content'],
                metadata={
                    "source": source,
                    "page_numbers": table['page_numbers'],
                    "is_table": True,
                    "source_table_idx": table['table_idx']
                }
            )
            summary_documents.append(summary_document)
    return summary_documents

def get_detail_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    source = os.path.splitext(os.path.basename(pdf_path))[0]

    page_documents = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if not text:
            continue

        page_document = LangchainDocument(
            page_content=text,
            metadata={
                "source": source,
                "page_numbers": [page_num + 1], 
                "is_table": False
            }
        )
        page_documents.append(page_document)

    doc.close()
    return page_documents

def process_all_pdfs_in_folder(folder_path):
    all_page_documents = []
    all_table_documents = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"\n>>> Đang xử lý: {pdf_path}")
            page_docs = get_detail_chunks(pdf_path)
            all_page_documents.extend(page_docs)
            table_docs = summarize_table()
            all_table_documents.extend(table_docs)

    return all_page_documents, all_table_documents

def add_documents(client, collection_name, documents, chunk_size, chunk_overlap, embedding_model_name):
    """
    This function adds documents to the desired Qdrant collection given the specified RAG parameters.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in documents:
        docs_processed += text_splitter.split_documents([doc])

    docs_contents = []
    docs_metadatas = []

    for doc in docs_processed:
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            docs_contents.append(doc.page_content)
            docs_metadatas.append(doc.metadata)
        else:
            # Handle the case where attributes are missing
            print("Warning: Some documents do not have 'page_content' or 'metadata' attributes.")

    client.set_model(embedding_model_name=embedding_model_name)
    client.add(collection_name=collection_name, metadata=docs_metadatas, documents=docs_contents)


def get_documents(client, collection_name, query, embedding_model, num_documents=5):
    """
    This function retrieves the desired number of documents from the Qdrant collection given a query.
    It returns a list of the retrieved documents.
    """
    client.set_model(embedding_model_name=embedding_model)
    search_results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=num_documents,
    )
    results = [r.page_content for r in search_results]
    return results