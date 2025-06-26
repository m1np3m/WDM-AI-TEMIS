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

# === Constants ===
JINA_MODEL = "jina-colbert-v1-en"
JINA_URL = "https://api.jina.ai/v1/rerank"



class Reranker:
    def __init__(self, method: Optional[str] = None):
        self.method = method
        self.rerank_func = self.get_reranker_by_name(method)
        
        # Lazy loading cache for models
        self._mxbai = None
        self._cohere_client = None
        self._bce_model = None
        self._colbert_model = None
        self._flashrank_model = None
        self._st_model = None
        self._bge_model = None

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
            "pretrained_bge": self.pretrained_bge_reranker,
            "finetune_bge": self.finetune_bge_reranker,
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

    def mixedbread_reranker(self, query: str, documents, top_k: int = 5) -> List[str]:
        if self._mxbai is None:
            from mixedbread import Mixedbread
            self._mxbai = Mixedbread(api_key=os.getenv("MXBAI_API_KEY"))
        
        try:
            response = self._mxbai.rerank(
                model="mixedbread-ai/mxbai-rerank-large-v1",
                query=query,
                input=documents,
                top_k=top_k,
                return_input=True
            )
            return [str(doc.input) for doc in response.data if doc.input is not None]
        except Exception as e:
            print(f"[Mixedbread Reranker] Error: {e}")
            return documents[:top_k]

    def cohere_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if self._cohere_client is None:
            import cohere
            self._cohere_client = cohere.Client(os.getenv("COHERE_KEY"))
        
        try:
            response = self._cohere_client.rerank(
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
        if self._bce_model is None:
            from BCEmbedding import RerankerModel
            self._bce_model = RerankerModel("maidalun1020/bce-reranker-base_v1", use_fp16=True, device=DEVICE)
        
        try:
            rerank_input = [(query, doc) for doc in documents]
            scores = self._bce_model.compute_score(rerank_input)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in ranked[:top_k]]
        except Exception as e:
            print(f"[BCE Reranker] Error: {e}")
            return documents[:top_k]

    def flashrank_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if self._flashrank_model is None:
            from flashrank import Ranker, RerankRequest
            self._flashrank_model = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=512)
        
        try:
            from flashrank import RerankRequest
            passages = [{"text": doc} for doc in documents]
            request = RerankRequest(query=query, passages=passages)
            results = self._flashrank_model.rerank(request)
            return [item["text"] for item in results[:top_k]]
        except Exception as e:
            print(f"[Flashrank Reranker] Error: {e}")
            return documents[:top_k]

    def st_crossencoder_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if self._st_model is None:
            from sentence_transformers import CrossEncoder
            self._st_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._st_model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[ST CrossEncoder Reranker] Error: {e}")
            return documents[:top_k]
        
    def pretrained_bge_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if self._colbert_model is None:
            from FlagEmbedding import FlagAutoReranker
            self._colbert_model = FlagAutoReranker.from_finetuned(
                model_name_or_path="BAAI/bge-reranker-v2-minicpm-layerwise",
                max_length=512,
                devices=DEVICE
            )
        
        try:
            pairs = [(query, doc) for doc in documents]
            scores = self._colbert_model.compute_score(pairs, normalize=True)
            if scores is not None:
                ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in ranked[:top_k]]
            else:
                return documents[:top_k]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            return documents[:top_k]
        
    def finetune_bge_reranker(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if self._bge_model is None:
            from .bge_finetune import BGEv2m3Reranker
            self._bge_model = BGEv2m3Reranker(
                model_path=os.getenv("BGEV3_RE_RANKER_PATH", "src/bge_v2_m3_rerank/bgev2m3_finetune"),
                device=DEVICE
            )
        
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._bge_model.compute_score(pairs, normalize=True)
            if scores is not None:
                ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in ranked[:top_k]]
            else:
                return documents[:top_k]
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