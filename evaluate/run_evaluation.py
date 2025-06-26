import argparse
import asyncio
import os
import sys

from prepare_documents import process_all_pdfs_in_folder

# append project dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from langchain.docstore.document import Document as LangchainDocument

from configs.config import *
from libs.common import *
from utils.format_utils import *

# from utils.extract_tables import full_pipeline
from utils.rag_evaluation import *
from utils.rag_qdrant_utils import *

load_dotenv(find_dotenv())

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--pdf_folder", type=str, default=f"{data_dir}/QA_tables/pdf", help="Path to the folder containing the pdf files")
    parser.add_argument("--qa_path", type=str, default=f"{data_dir}/QA_tables/fixed_label_QA.json", help="Path to the qa file")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=128, help="Chunk overlap")
    parser.add_argument("--num_docs", type=int, default=7, help="Number of docs")
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-base-en", help="Embedding model name")
    parser.add_argument("--chunk_type", type=str, default="character", help="Chunk type")
    parser.add_argument("--hybrid_search", type=bool, default=False, help="Hybrid search")
    parser.add_argument("--reranker_model_name", type=str, default=None, help="Reranker model name")
    parser.add_argument("--path_to_save", type=str, default=f"{exps_dir}/", help="Path to save the evaluation results")
    return parser.parse_args()

class Args:
    def __init__(self, pdf_folder, qa_path, chunk_size, chunk_overlap, num_docs, embedding_model_name, chunk_type, hybrid_search, reranker_model_name, path_to_save):
        self.pdf_folder = pdf_folder
        self.qa_path = qa_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_docs = num_docs
        self.embedding_model_name = embedding_model_name
        self.chunk_type = chunk_type
        self.hybrid_search = hybrid_search
        self.reranker_model_name = reranker_model_name
        self.path_to_save = path_to_save


async def main(args):
    pdf_folder = args.pdf_folder
    qa_path = args.qa_path
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    num_docs = args.num_docs
    embedding_model_name = args.embedding_model_name
    chunk_type = args.chunk_type
    hybrid_search = args.hybrid_search
    path_to_save = args.path_to_save
    reranker_model_name = args.reranker_model_name
    
    # print config
    print(f"pdf_folder: {pdf_folder}")
    print(f"qa_path: {qa_path}")
    print(f"chunk_size: {chunk_size}")
    print(f"chunk_overlap: {chunk_overlap}")
    print(f"num_docs: {num_docs}")
    print(f"embedding_model_name: {embedding_model_name}")
    print(f"chunk_type: {chunk_type}")
    print(f"hybrid_search: {hybrid_search}")
    print(f"reranker_model_name: {reranker_model_name}")
    print(f"path_to_save: {path_to_save}")

    hybrid_flag = "hybrid" if hybrid_search else "base"
    reranker_flag = reranker_model_name if reranker_model_name else "NoneReranker"
    
    
    # create qdrant client
    client = qdrant_client.QdrantClient(
        path=f"{path_to_save}/qdrant_client_memory",
    )
    
    COLLECTION_NAME = f"experiment_{hybrid_flag}_{chunk_size}_{chunk_overlap}_{embedding_model_name.split('/')[1]}_{chunk_type}_{reranker_flag}"
    
    # check if collection exists
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists, pass the add documents step")
    else:
        # Prepare evaluation data
        all_page_documents, all_table_documents = asyncio.run(process_all_pdfs_in_folder(pdf_folder, qa_path))
        exp = QdrantRAG(client=client)
        # add documents to qdrant
        if hybrid_search:
            pass
            #TODO: add hybrid search
        else:
            exp.add_documents(
                collection_name=COLLECTION_NAME,
                documents=all_page_documents,
                tables=all_table_documents,
                chunk_size=chunk_size,
                chunk_type=chunk_type,
                chunk_overlap=chunk_overlap,
                embedding_model_name=embedding_model_name) 

    # run evaluation
    if reranker_model_name:
        pass
    else:
        pass


if __name__ == "__main__":
    # args = parse_args() 
    args = Args(
        pdf_folder=f"{data_dir}/QA_tables/pdf",
        qa_path=f"{data_dir}/QA_tables/fixed_label_QA.json",
        chunk_size=512,
        chunk_overlap=128,
        num_docs=7,
        embedding_model_name="BAAI/bge-base-en",
        chunk_type="character",
        hybrid_search=False,
        reranker_model_name=None,
        path_to_save=f"{exps_dir}/") # for testing
    
    asyncio.run(main(args))