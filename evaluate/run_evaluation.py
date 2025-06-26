import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd
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
from utils.reranker_utils import *

load_dotenv(find_dotenv())

global_documents = {"page_documents": [], "table_documents": []}


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--pdf_folder",
        type=str,
        default=f"{data_dir}/QA_tables/pdf",
        help="Path to the folder containing the pdf files",
    )
    parser.add_argument(
        "--qa_path",
        type=str,
        default=f"{data_dir}/QA_tables/fixed_label_QA.json",
        help="Path to the qa file",
    )
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=128, help="Chunk overlap")
    parser.add_argument("--num_docs", type=int, default=7, help="Number of docs")
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="BAAI/bge-base-en",
        help="Embedding model name",
    )
    parser.add_argument(
        "--chunk_type", type=str, default="character", help="Chunk type"
    )
    parser.add_argument(
        "--hybrid_search", type=bool, default=False, help="Hybrid search"
    )
    parser.add_argument(
        "--reranker_model_name", type=str, default=None, help="Reranker model name"
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default=f"{exps_dir}/",
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--force_create",
        action="store_true",
        help="Force recreate collection even if it exists",
    )
    return parser.parse_args()


class Args:
    def __init__(
        self,
        pdf_folder,
        qa_path,
        chunk_size,
        chunk_overlap,
        num_docs,
        embedding_model_name,
        chunk_type,
        hybrid_search,
        reranker_model_name,
        path_to_save,
        force_create=False,
    ):
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
        self.force_create = force_create


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
    force_create = args.force_create

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
    print(f"force_create: {force_create}")

    hybrid_flag = "hybrid" if hybrid_search else "base"
    reranker_flag = reranker_model_name if reranker_model_name else "NoneReranker"

    # Sanitize embedding model name for collection name
    embedding_name = (
        embedding_model_name.split("/")[-1]
        if "/" in embedding_model_name
        else embedding_model_name
    )

    # create qdrant client
    client = qdrant_client.QdrantClient(
        path=f"{path_to_save}/qdrant_client_memory",
    )

    COLLECTION_NAME = f"exp_{hybrid_flag}_cs{chunk_size}_co{chunk_overlap}_nd{num_docs}_{embedding_name}_{chunk_type}_{reranker_flag}"
    print(f"Collection name: {COLLECTION_NAME}")

    # check if collection exists and handle force_create
    if client.collection_exists(COLLECTION_NAME):
        if force_create:
            print(f"Force creating: Deleting existing collection {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(
                f"Collection {COLLECTION_NAME} already exists, pass the add documents step"
            )
            return

    # Prepare evaluation data
    if (
        global_documents["page_documents"] == []
        or global_documents["table_documents"] == []
    ):
        all_page_documents, all_table_documents = await process_all_pdfs_in_folder(
            pdf_folder, qa_path
        )
        global_documents["page_documents"] = all_page_documents
        global_documents["table_documents"] = all_table_documents
    else:
        all_page_documents = global_documents["page_documents"]
        all_table_documents = global_documents["table_documents"]

    exp = QdrantRAG(client=client)

    # add documents to qdrant with error handling
    print("Adding documents to qdrant...")
    try:
        if hybrid_search:
            exp.add_documents_hybrid(
                collection_name=COLLECTION_NAME,
                documents=all_page_documents,
                tables=all_table_documents,
                chunk_type=chunk_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model_name=embedding_model_name,
            )
        else:
            exp.add_documents(
                collection_name=COLLECTION_NAME,
                documents=all_page_documents,
                tables=all_table_documents,
                chunk_size=chunk_size,
                chunk_type=chunk_type,
                chunk_overlap=chunk_overlap,
                embedding_model_name=embedding_model_name,
            )

        print(f"✅ Successfully added documents to collection {COLLECTION_NAME}")

    except KeyboardInterrupt:
        print(
            f"\n⚠️  KeyboardInterrupt detected! Cleaning up incomplete collection {COLLECTION_NAME}"
        )
        try:
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
                print(
                    f"✅ Successfully deleted incomplete collection {COLLECTION_NAME}"
                )
        except Exception as cleanup_error:
            print(f"❌ Error during cleanup: {cleanup_error}")
        raise  # Re-raise the KeyboardInterrupt

    except Exception as e:
        print(f"\n❌ Error occurred while adding documents: {e}")
        print(f"⚠️  Cleaning up incomplete collection {COLLECTION_NAME}")
        try:
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
                print(
                    f"✅ Successfully deleted incomplete collection {COLLECTION_NAME}"
                )
        except Exception as cleanup_error:
            print(f"❌ Error during cleanup: {cleanup_error}")
        raise  # Re-raise the original exception

    # run evaluation
    # load eval df
    with open(qa_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_df = pd.DataFrame(eval_data)
    print(f"Number of questions: {len(eval_df)}")

    # run evaluation
    doc_retrieval_function = (
        exp.get_documents if not hybrid_search else exp.get_documents_hybrid
    )
    reranker_function = (
        None if not reranker_model_name else Reranker(method=reranker_model_name)
    )

    ex: pd.DataFrame = run_ragas_eval(
        eval_df,
        collection_name=COLLECTION_NAME,
        doc_retrieval_function=doc_retrieval_function,
        embedding_model_name=embedding_model_name,
        num_docs=num_docs,
        reranker_function=reranker_function,
        path=os.path.join(path_to_save, COLLECTION_NAME),
        use_optimized_metrics=True,
    )
    print(f"✅ Successfully ran evaluation for {COLLECTION_NAME}")
    print(ex.head())

    return ex


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
        path_to_save=f"{exps_dir}/",
        force_create=False,
    )  # for testing
    start_time = time.time()
    ex = asyncio.run(main(args))
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")

    # plot comparison
    plot_experiment_comparison(
        experiment_results_list=[ex],
        experiment_names=["test_exp"],
        metrics_to_plot=["context_precision", "context_recall", "hit_rate", "mrr"],
    )
