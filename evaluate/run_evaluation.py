import argparse
import asyncio
import json
import os
import sys
import time

import mlflow
import pandas as pd
import qdrant_client
from dotenv import find_dotenv, load_dotenv

# append project dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from libs.common import *
from utils.format_utils import *

# from utils.extract_tables import full_pipeline
from utils.rag_evaluation_optimized import plot_experiment_comparison, run_ragas_eval
from utils.rag_qdrant_utils import QdrantRAG
from utils.reranker_utils_optimized import Reranker

# Import prepare_documents from current directory (evaluate/)
try:
    from .prepare_documents import process_all_pdfs_in_folder
except ImportError:
    # Fallback to local import if relative import fails
    try:
        from prepare_documents import process_all_pdfs_in_folder
    except ImportError:
        # Last fallback - add current directory to path
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from prepare_documents import process_all_pdfs_in_folder

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

def get_collection_name(args):
    hybrid_flag = "hybrid" if args.hybrid_search else "base"
    embedding_name = args.embedding_model_name.split("/")[-1] if "/" in args.embedding_model_name else args.embedding_model_name
    return f"exp_{hybrid_flag}_cs{args.chunk_size}_co{args.chunk_overlap}_{embedding_name}_{args.chunk_type}"

def get_experiment_name(args):
    hybrid_flag = "hybrid" if args.hybrid_search else "base"
    reranker_flag = args.reranker_model_name if args.reranker_model_name else "NoneReranker"
    embedding_name = args.embedding_model_name.split("/")[-1] if "/" in args.embedding_model_name else args.embedding_model_name
    return f"exp_{hybrid_flag}_cs{args.chunk_size}_co{args.chunk_overlap}_nd{args.num_docs}_{embedding_name}_{args.chunk_type}_{reranker_flag}"

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
    COLLECTION_NAME = get_collection_name(args)
    print(f"Collection name: {COLLECTION_NAME}")

    # check if collection exists and handle force_create
    collection_exists = client.collection_exists(COLLECTION_NAME)
    if collection_exists:
        if force_create:
            print(f"Force creating: Deleting existing collection {COLLECTION_NAME}")
            client.delete_collection(COLLECTION_NAME)
            collection_exists = False
        else:
            print(
                f"Collection {COLLECTION_NAME} already exists, skipping document addition"
            )
    start_time = time.time()
    # Only add documents if collection doesn't exist or was recreated
    if not collection_exists:
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
    else:
        # Collection exists, just initialize QdrantRAG
        exp = QdrantRAG(client=client)
    end_time = time.time()
    add_document_time = end_time - start_time
    # run evaluation (always run this part)
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
    
    # FIX: Adjust concurrency based on the reranker to prevent CUDA OOM errors.
    # Heavy rerankers like 'pretrained_bge' should run sequentially.
    max_requests = 5
    if reranker_model_name == "pretrained_bge":
        print("Using 'pretrained_bge' reranker, reducing concurrent requests to 1 to prevent OOM error.")
        max_requests = 1

    start_time = time.time()
    ex: pd.DataFrame = run_ragas_eval(
        eval_df,
        collection_name=COLLECTION_NAME,
        doc_retrieval_function=doc_retrieval_function,
        embedding_model_name=embedding_model_name,
        num_docs=num_docs,
        reranker_function=reranker_function,
        path=os.path.join(path_to_save, get_experiment_name(args)) + ".csv",
        use_optimized_metrics=True,
        max_concurrent_requests=max_requests,  # Use the adjusted value
    )
    end_time = time.time()
    eval_time = end_time - start_time
    print(f"✅ Successfully ran evaluation for {COLLECTION_NAME}")
    
    context_precision = ex['context_precision'].mean()
    context_recall = ex['context_recall'].mean()
    hit_rate = ex['hit_rate'].mean()
    mrr = ex['mrr'].mean()

    return {
        "exp": ex,
        "add_document_time": add_document_time,
        "eval_time": eval_time,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "hit_rate": hit_rate,
        "mrr": mrr,
    }


if __name__ == "__main__":
    # Set MLFlow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("RAG_Evaluation_Experiments")
    
    pdf_folder = f"{data_dir}/QA_tables/pdf"
    qa_path = f"{data_dir}/QA_tables/fixed_label_QA.json"
    
    # Define different configurations to test
    configs = [
        Args(
            pdf_folder=pdf_folder,
            qa_path=qa_path,
            chunk_size=512,
            chunk_overlap=128,
            num_docs=7,
            embedding_model_name="BAAI/bge-base-en",
            chunk_type="character",
            hybrid_search=True,
            reranker_model_name="bce",
            path_to_save=f"{exps_dir}/",
            force_create=False,
        ),
        Args(
            pdf_folder=pdf_folder,
            qa_path=qa_path,
            chunk_size=512,
            chunk_overlap=128,
            num_docs=7,
            embedding_model_name="BAAI/bge-base-en",
            chunk_type="character",
            hybrid_search=True,
            reranker_model_name="pretrained_bge",
            path_to_save=f"{exps_dir}/",
            force_create=False,
        ),
        Args(
            pdf_folder=pdf_folder,
            qa_path=qa_path,
            chunk_size=512,
            chunk_overlap=128,
            num_docs=7,
            embedding_model_name="BAAI/bge-base-en",
            chunk_type="character",
            hybrid_search=True,
            reranker_model_name="flashrank",
            path_to_save=f"{exps_dir}/",
            force_create=False,
        ),
        Args(
            pdf_folder=pdf_folder,
            qa_path=qa_path,
            chunk_size=512,
            chunk_overlap=128,
            num_docs=7,
            embedding_model_name="BAAI/bge-base-en",
            chunk_type="character",
            hybrid_search=True,
            reranker_model_name="st-crossencoder",
            path_to_save=f"{exps_dir}/",
            force_create=False,
        ),
    ]
    
    # You can also parse args instead of using predefined configs
    # args = parse_args()
    # configs = [args]
    
    for i, args in enumerate(configs):
        collection_name = get_collection_name(args)
        experiment_name = get_experiment_name(args)
        with mlflow.start_run(run_name=experiment_name):
            print(f"\n🚀 Starting MLFlow Run {i+1}/{len(configs)}")
            print("=" * 60)
            
            # Log parameters (config)
            mlflow.log_param("pdf_folder", args.pdf_folder)
            mlflow.log_param("qa_path", args.qa_path)
            mlflow.log_param("chunk_size", args.chunk_size)
            mlflow.log_param("chunk_overlap", args.chunk_overlap)
            mlflow.log_param("num_docs", args.num_docs)
            mlflow.log_param("embedding_model_name", args.embedding_model_name)
            mlflow.log_param("chunk_type", args.chunk_type)
            mlflow.log_param("hybrid_search", args.hybrid_search)
            mlflow.log_param("reranker_model_name", args.reranker_model_name or "None")
            mlflow.log_param("force_create", args.force_create)
            
            # Log derived parameters
            hybrid_flag = "hybrid" if args.hybrid_search else "base"
            reranker_flag = args.reranker_model_name if args.reranker_model_name else "NoneReranker"
            embedding_name = args.embedding_model_name.split("/")[-1] if "/" in args.embedding_model_name else args.embedding_model_name
            
            mlflow.log_param("hybrid_flag", hybrid_flag)
            mlflow.log_param("reranker_flag", reranker_flag)
            mlflow.log_param("embedding_name", embedding_name)
            
            # Create collection name and experiment name, and log them
            mlflow.log_param("collection_name", collection_name)
            mlflow.log_param("experiment_name", experiment_name)
            
            try:
                # Run the main evaluation
                result = asyncio.run(main(args))
                
                # Làm tròn các metrics đến 4 chữ số thập phân
                add_document_time = round(result['add_document_time'], 4)
                eval_time = round(result['eval_time'], 4)
                total_time = round(result['add_document_time'] + result['eval_time'], 4)
                context_precision = round(result['context_precision'], 4)
                context_recall = round(result['context_recall'], 4)
                hit_rate = round(result['hit_rate'], 4)
                mrr = round(result['mrr'], 4)
                f1_score = (
                    round(
                        2 * (result['context_precision'] * result['context_recall']) / (result['context_precision'] + result['context_recall']),
                        4,
                    )
                    if (result['context_precision'] + result['context_recall']) > 0
                    else 0
                )

                # Log metrics đã làm tròn
                mlflow.log_metric("add_document_time", add_document_time)
                mlflow.log_metric("eval_time", eval_time)
                mlflow.log_metric("total_time", total_time)
                mlflow.log_metric("context_precision", context_precision)
                mlflow.log_metric("context_recall", context_recall)
                mlflow.log_metric("hit_rate", hit_rate)
                mlflow.log_metric("mrr", mrr)
                mlflow.log_metric("f1_score", f1_score)
                
                # Log artifacts (save CSV results)
                csv_path = os.path.join(args.path_to_save, experiment_name + ".csv")
                if os.path.exists(csv_path):
                    mlflow.log_artifact(csv_path)
                
                # Print results
                print(f"📊 Results for Run {i+1}:")
                print(f"   Add Document Time: {result['add_document_time']:.2f}s")
                print(f"   Eval Time: {result['eval_time']:.2f}s")
                print(f"   Total Time: {result['add_document_time'] + result['eval_time']:.2f}s")
                print(f"   Context Precision: {result['context_precision']:.4f}")
                print(f"   Context Recall: {result['context_recall']:.4f}")
                print(f"   F1 Score: {f1_score:.4f}")
                print(f"   Hit Rate: {result['hit_rate']:.4f}")
                print(f"   MRR: {result['mrr']:.4f}")
                
                mlflow.log_param("status", "success")
                print(f"✅ Successfully completed Run {i+1}")
                
            except Exception as e:
                # Log error information
                mlflow.log_param("status", "failed")
                mlflow.log_param("error_message", str(e))
                print(f"❌ Run {i+1} failed with error: {e}")
                raise
    
    print(f"\n🎉 Completed all {len(configs)} runs!")
    print("📈 Check MLFlow UI with: mlflow ui")
    print("🔗 Then visit: http://localhost:5000")
