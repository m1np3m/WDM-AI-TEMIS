# reranker.py

import os
import time
from typing import Callable, List, Optional

import requests
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

DEVICE = get_device()

# === Jina ===
JINA_MODEL = "jina-colbert-v1-en"
JINA_URL = "https://api.jina.ai/v1/rerank"

# === Mixedbread ===
from mixedbread import Mixedbread

mxbai = Mixedbread(api_key=os.getenv("MXBAI_API_KEY"))

# === Cohere ===
import cohere

co = cohere.Client(os.getenv("COHERE_KEY"))

# === BCE ===
from BCEmbedding import RerankerModel

bce_model = RerankerModel("maidalun1020/bce-reranker-base_v1", use_fp16=True, device=DEVICE)

# === FlagEmbedding ColBERT ===
from FlagEmbedding import FlagAutoReranker

colbert_model = FlagAutoReranker.from_finetuned(
    model_name_or_path="BAAI/bge-reranker-v2-minicpm-layerwise",
    max_length=512,
    devices=DEVICE
)

# === Flashrank ===
from flashrank import Ranker, RerankRequest

flashrank_model = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=512)

# === Sentence Transformers ===
from sentence_transformers import CrossEncoder

st_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class Reranker:
    def __init__(self, method: Optional[str] = None):
        self.method = method
        self.rerank_func = self.get_reranker_by_name(method)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if not self.rerank_func:
            return documents[:top_k]
        return self.rerank_func(query, documents, top_k)

    def get_reranker_by_name(self, name: Optional[str]) -> Optional[Callable]:
        if name is None:
            return None
        rerankers = {
            "jina": self.jina_reranker,
            "mixedbread": self.mixedbread_reranker,
            "cohere": self.cohere_reranker,
            "bce": self.bce_reranker,
            "colbert": self.colbert_reranker,
            "flashrank": self.flashrank_reranker,
            "st-crossencoder": self.st_crossencoder_reranker,
        }
        return rerankers.get(name.lower())

    # === Individual Reranker Functions ===
    def jina_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
        }
        data = {
            "model": JINA_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }
        try:
            response = requests.post(JINA_URL, headers=headers, json=data)
            response = response.json()
            return [doc['document']['text'] for doc in response['results']]
        except Exception as e:
            print(f"[JINA Reranker] Error: {e}")
            return documents[:top_k]

    def mixedbread_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            response = mxbai.rerank(
                model="mixedbread-ai/mxbai-rerank-large-v1",
                query=query,
                input=documents,
                top_k=top_k,
                return_input=True
            )
            return [doc.input for doc in response.data]
        except Exception as e:
            print(f"[Mixedbread Reranker] Error: {e}")
            return documents[:top_k]

    def cohere_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=True
            )
            return [documents[doc.index] for doc in response.results]
        except Exception as e:
            print(f"[Cohere Reranker] Error: {e}")
            return documents[:top_k]

    def bce_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            rerank_input = [(query, doc) for doc in documents]
            scores = bce_model.compute_score(rerank_input)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in ranked[:top_k]]
        except Exception as e:
            print(f"[BCE Reranker] Error: {e}")
            return documents[:top_k]

    def colbert_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = colbert_model.compute_score(pairs, normalize=True)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[ColBERT Reranker] Error: {e}")
            return documents[:top_k]

    def flashrank_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            passages = [{"text": doc} for doc in documents]
            request = RerankRequest(query=query, passages=passages)
            results = flashrank_model.rerank(request)
            return [item["text"] for item in results[:top_k]]
        except Exception as e:
            print(f"[Flashrank Reranker] Error: {e}")
            return documents[:top_k]

    def st_crossencoder_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = st_model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[ST CrossEncoder Reranker] Error: {e}")
            return documents[:top_k]
        
    def bge_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = colbert_model.compute_score(pairs, normalize=True)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            return documents[:top_k]

if __name__ == 'main':
    # from reranker import Reranker

    reranker = Reranker(method="flashrank")

    documents = reranker.rerank(
        query="What is hybrid retrieval?",
        documents=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
        top_k=5
    )