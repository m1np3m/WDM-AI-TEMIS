# reranker.py

import os
import time
from typing import Callable, List, Optional

import requests
import torch

# BGE Finetuned Reranker class (copied from src/bge_v2_m3_rerank/loadmodelfintune.py)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEv2m3FinetunedReranker:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def bgev2m3_rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = []

            with torch.no_grad():
                for q, d in pairs:
                    inputs = self.tokenizer(
                        q,
                        d,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    score = outputs.logits.squeeze().item()
                    scores.append(score)

            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked[:top_k]]
        except Exception as e:
            print(f"[BGE Finetuned Reranker] Error: {e}")
            return documents[:top_k]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"


DEVICE = get_device()


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
            "pretrained_bge": self.pretrained_bge_reranker,
            # "finetune_bge": self.finetune_bge_reranker,
            "flashrank": self.flashrank_reranker,
            "st-crossencoder": self.st_crossencoder_reranker,
        }
        return rerankers.get(name.lower())

    # === Individual Reranker Functions ===
    def jina_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        JINA_MODEL = "jina-colbert-v1-en"
        JINA_URL = "https://api.jina.ai/v1/rerank"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
        }
        data = {
            "model": JINA_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
        try:
            response = requests.post(JINA_URL, headers=headers, json=data)
            response = response.json()
            return [doc["document"]["text"] for doc in response["results"]]
        except Exception as e:
            print(f"[JINA Reranker] Error: {e}")
            return documents[:top_k]

    def mixedbread_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from typing import Any, cast

            from mixedbread import Mixedbread

            mxbai = Mixedbread(api_key=os.getenv("MXBAI_API_KEY", ""))
            response = mxbai.rerank(
                model="mixedbread-ai/mxbai-rerank-large-v1",
                query=query,
                input=cast(Any, documents),  # Cast to satisfy type checker
                top_k=top_k,
                return_input=True,
            )
            return [
                str(doc.input) if doc.input is not None else "" for doc in response.data
            ]
        except Exception as e:
            print(f"[Mixedbread Reranker] Error: {e}")
            return documents[:top_k]

    def cohere_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from cohere import Client

            co = Client(os.getenv("COHERE_KEY", ""))
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=True,
            )
            return [documents[doc.index] for doc in response.results]
        except Exception as e:
            print(f"[Cohere Reranker] Error: {e}")
            return documents[:top_k]

    def bce_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from BCEmbedding import RerankerModel

            bce_model = RerankerModel(
                "maidalun1020/bce-reranker-base_v1", use_fp16=True, device=DEVICE
            )

            rerank_input = [(query, doc) for doc in documents]
            scores = bce_model.compute_score(rerank_input)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in ranked[:top_k]]
        except Exception as e:
            print(f"[BCE Reranker] Error: {e}")
            return documents[:top_k]

    def flashrank_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from flashrank import Ranker, RerankRequest

            flashrank_model = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2", max_length=512
            )

            passages = [{"text": doc} for doc in documents]
            request = RerankRequest(query=query, passages=passages)
            results = flashrank_model.rerank(request)
            return [item["text"] for item in results[:top_k]]
        except Exception as e:
            print(f"[Flashrank Reranker] Error: {e}")
            return documents[:top_k]

    def st_crossencoder_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from sentence_transformers import CrossEncoder

            st_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            pairs = [[query, doc] for doc in documents]
            scores = st_model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[ST CrossEncoder Reranker] Error: {e}")
            return documents[:top_k]

    def pretrained_bge_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            from FlagEmbedding import FlagAutoReranker

            # Initialize BGE reranker with proper dtype handling
            try:
                colbert_model = FlagAutoReranker.from_finetuned(
                    model_name_or_path="BAAI/bge-reranker-v2-minicpm-layerwise",
                    max_length=512,
                    device=DEVICE,
                    use_fp16=False,
                    torch_dtype=torch.float32,  # Explicitly set to float32
                )
                # FIX: Explicitly cast the model to float32 to prevent dtype errors.
                colbert_model.model.to(torch.float32)
                # FIX 2: Force the use_fp16 flag to False to prevent the model.half() call.
                colbert_model.use_fp16 = False
            except Exception as e:
                print(
                    f"[BGE Model Init] Warning: Failed to initialize with torch_dtype, trying fallback: {e}"
                )
                # Fallback initialization without torch_dtype
                colbert_model = FlagAutoReranker.from_finetuned(
                    model_name_or_path="BAAI/bge-reranker-v2-minicpm-layerwise",
                    max_length=512,
                    device=DEVICE,
                    use_fp16=False,
                )
                # FIX: Also apply the fix in the fallback initialization.
                colbert_model.model.to(torch.float32)
                # FIX 2: Also apply the fix in the fallback initialization.
                colbert_model.use_fp16 = False
            pairs = [(query, doc) for doc in documents]
            scores = colbert_model.compute_score(pairs, normalize=True)
            if scores is not None:
                # Ensure scores are in float32 format
                import numpy as np
                import torch

                # Convert to float32 to avoid type mismatch errors
                if torch.is_tensor(scores):
                    # Force conversion to float32 regardless of original dtype
                    scores = scores.to(dtype=torch.float32)
                    # Convert to numpy for easier handling
                    if scores.is_cuda:
                        scores = scores.cpu().numpy()
                    else:
                        scores = scores.numpy()
                elif isinstance(scores, np.ndarray):
                    scores = scores.astype(np.float32)
                else:
                    # Handle list or other types
                    scores = np.array(scores, dtype=np.float32)

                # Ensure documents and scores have same length
                if len(documents) != len(scores):
                    print(
                        f"[BGE Reranker] Warning: Length mismatch - documents: {len(documents)}, scores: {len(scores)}"
                    )
                    min_len = min(len(documents), len(scores))
                    documents = documents[:min_len]
                    scores = scores[:min_len]

                ranked = sorted(
                    zip(documents, scores), key=lambda x: float(x[1]), reverse=True
                )
                return [doc for doc, _ in ranked[:top_k]]
            else:
                print(f"[BGE Reranker] Warning: compute_score returned None")
                return documents[:top_k]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            import traceback

            print(f"[BGE Reranker] Traceback: {traceback.format_exc()}")
            return documents[:top_k]

    def finetune_bge_reranker(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[str]:
        try:
            bge_finetune_model = None
            try:
                bge_finetune_model_path = os.getenv(
                    "BGEV3_RE_RANKER_PATH", "src/bge_v2_m3_rerank/bgev2m3_finetune"
                )
                if os.path.exists(bge_finetune_model_path):
                    bge_finetune_model = BGEv2m3FinetunedReranker(
                        model_path=bge_finetune_model_path, device=DEVICE
                    )
                    print(
                        f"[BGE Finetune] Loaded finetuned model from: {bge_finetune_model_path}"
                    )
                else:
                    # Try alternative path from Hugging Face
                    alt_path = "rakhuynh/bgev2m3_finetune_wdm"
                    print(f"[BGE Finetune] Local path not found, trying: {alt_path}")
                    bge_finetune_model = BGEv2m3FinetunedReranker(
                        model_path=alt_path, device=DEVICE
                    )
                    print(f"[BGE Finetune] Loaded finetuned model from HF: {alt_path}")
            except Exception as e:
                print(
                    f"[BGE Finetune] Warning: Failed to initialize finetuned BGE model: {e}"
                )
                bge_finetune_model = None
            if bge_finetune_model is None:
                print(
                    f"[BGE Finetune Reranker] Warning: Finetuned model not available, using fallback"
                )
                return documents[:top_k]

            # Use the bgev2m3_rerank method from the finetuned model
            reranked_docs = bge_finetune_model.bgev2m3_rerank(query, documents, top_k)
            return reranked_docs
        except Exception as e:
            print(f"[BGE Finetune Reranker] Error: {e}")
            import traceback

            print(f"[BGE Finetune Reranker] Traceback: {traceback.format_exc()}")
            return documents[:top_k]


if __name__ == "main":
    # from reranker import Reranker

    reranker = Reranker(method="flashrank")

    documents = reranker.rerank(
        query="What is hybrid retrieval?",
        documents=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
        top_k=5,
    )
