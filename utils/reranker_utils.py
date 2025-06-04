from mixedbread_ai.client import MixedbreadAI
import cohere
import time
import os
import requests
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding, SparseEmbedding
import numpy as np
from typing import List, Tuple, Union
import pickle

sparse_model_name = "prithivida/Splade_PP_en_v1"
embedding_model_name = "BAAI/bge-small-en"

sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)

embedding_model = FastEmbedEmbeddings(model_name=embedding_model_name)

client = QdrantClient(url="http://localhost:6333")

RERANK_URL = f"https://api.jina.ai/v1/rerank"
RERANK_MODEL = "jina-colbert-v1-en"


COHERE_KEY=os.environ.get("COHERE_KEY")
co = cohere.Client(COHERE_KEY)

def get_reranked_documents_with_cohere(client, collection_name, query, num_documents=5):
    """
    This function retrieves the desired number of documents from the Qdrant collection given a query.
    It returns a list of the reranked retrieved documents.
    """
    
    search_results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=num_documents+4,
    )
    results = [r.metadata["document"] for r in search_results]
    
    response = co.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=results,
    top_n=num_documents,
    return_documents=True
    )

    time.sleep(6.1)
    
    return [doc.document.text for doc in response.results]


MXBAI_KEY=os.environ.get("MXBAI_API_KEY")
mxbai = MixedbreadAI(api_key=MXBAI_KEY)

def get_reranked_documents_with_mixedbread(client, collection_name, query, num_documents=5):
    """
    This function retrieves the desired number of documents from the Qdrant collection given a query.
    It returns a list of the reranked retrieved documents.
    """
    
    search_results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=num_documents+4,
    )
    results = [r.metadata["document"] for r in search_results]
    
    response = mxbai.reranking(
    model="mixedbread-ai/mxbai-rerank-large-v1",
    query=query,
    input=results,
    top_k=num_documents,
    return_input=True
    )

    time.sleep(1)
    
    return [doc.input for doc in response.data]



def get_reranked_documents_with_jina(client, collection_name, query, num_documents=5):
    """
    This function retrieves the desired number of documents from the Qdrant collection given a query.
    It returns a list of the reranked retrieved documents.
    """
    
    search_results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=num_documents+4,
    )
    results = [r.metadata["document"] for r in search_results]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
    }

    data = {
    "model": RERANK_MODEL,
    "query": query,
    "documents": results,
    "top_n": num_documents
    }

    response = requests.post(RERANK_URL, headers=headers, json=data)
    response = response.json()

    time.sleep(1)
    
    return [doc['document']['text'] for doc in response['results']]





def compute_sparse_vectors(texts: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
    """Tính toán sparse vector từ SPLADE model"""
    indices_list, values_list = [], []
    for text in texts:
        embedding: SparseEmbedding = next(sparse_model.embed([text]))
        indices_list.append(embedding.indices.tolist())
        values_list.append(embedding.values.tolist())
    return indices_list, values_list



def add_hybrid_documents(
    client: QdrantClient,
    collection_name: str,
    langchain_docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model_name: str = "BAAI/bge-small-en",
):
    """
    Chia nhỏ tài liệu, nhúng và thêm vào Qdrant với hỗ trợ hybrid search.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in langchain_docs:
        docs_processed += text_splitter.split_documents([doc])

    texts = [doc.page_content for doc in docs_processed]
    metadatas = [doc.metadata for doc in docs_processed]

    sparse_indices, sparse_values = compute_sparse_vectors(texts)

    for i, meta in enumerate(metadatas):
        meta["sparse_indices"] = sparse_indices[i]
        meta["sparse_values"] = sparse_values[i]

    vectorstore = Qdrant.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        collection_name=collection_name,
        url="http://localhost:6333",
        force_recreate=True,
        prefer_grpc=False,
        hnsw_config={"ef_construct": 128, "m": 16},
        optimizers_config={"indexing_threshold": 10000},
        on_disk_payload=True,
    )
    print(f"✅ Đã thêm {len(docs_processed)} đoạn văn bản vào collection '{collection_name}'")


def get_hybrid_documents(
    client: QdrantClient,
    collection_name: str,
    query: str,
    hybrid_factor: float = 0.4,
    num_documents: int = 5,
) -> List[str]:
    """
    Truy vấn hybrid trên Qdrant dựa trên alpha factor.
    """

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model,
    )

    all_docs = vectorstore.similarity_search(query="", k=1000) 
    all_texts = [doc.page_content for doc in all_docs]
    all_metadatas = [doc.metadata for doc in all_docs]

    dense_scores = [
        float(vectorstore._embedding_retriever._cosine_score(doc, query))
        for doc in all_texts
    ]

    sparse_scores = []
    for i in range(len(all_texts)):
        indices = all_metadatas[i]["sparse_indices"]
        values = all_metadatas[i]["sparse_values"]
        sparse_vec = np.zeros(30522) 
        for idx, val in zip(indices, values):
            sparse_vec[idx] = max(sparse_vec[idx], val)
        query_embedding = next(sparse_model.embed([query]))
        query_vec = np.zeros(30522)
        for idx, val in zip(query_embedding.indices, query_embedding.values):
            query_vec[idx] = val
        sparse_score = np.dot(sparse_vec, query_vec) / (np.linalg.norm(sparse_vec) * np.linalg.norm(query_vec) + 1e-8)
        sparse_scores.append(sparse_score)

    # Kết hợp hybrid score
    combined_scores = [
        (1 - hybrid_factor) * dense + hybrid_factor * sparse
        for dense, sparse in zip(dense_scores, sparse_scores)
    ]
    scored_docs = sorted(zip(combined_scores, all_texts), key=lambda x: x[0], reverse=True)

    # Trả về top-k kết quả
    results = [doc[1] for doc in scored_docs[:num_documents]]
    return results