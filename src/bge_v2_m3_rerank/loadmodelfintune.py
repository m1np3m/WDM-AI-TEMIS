import os
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class BGEv2m3Reranker:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def bgev2m3_rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.bge_reranker.predict(pairs, max_length=500)
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [doc for scores, doc in ranked[:top_k]]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            return documents[:top_k]

if __name__ == '__main__':
    reranker_path = os.getenv("BGEV3_RE_RANKER_PATH", r"\WDM-AI-TEMIS\src\\bge_v2_m3_rerank\\bgev2m3finetune")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = BGEv2m3Reranker(reranker_path, device=device)
    docs = ["Introduction to hybrid retrieval.", "Overview of dense vs sparse.",
            "BGE model details.", "Use cases of reranking.", "Evaluation metrics."]
    top_docs = reranker.bgev2m3_rerank("What is hybrid retrieval?", docs, top_k=3)
    print("Top documents:", top_docs)