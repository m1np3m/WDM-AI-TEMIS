import os
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BGEv2m3Reranker:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def bgev2m3_rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = []

            with torch.no_grad():
                for q, d in pairs:
                    inputs = self.tokenizer(
                        q, d,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=512
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    score = outputs.logits.squeeze().item()
                    scores.append(score)

            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked[:top_k]]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            return documents[:top_k]

if __name__ == '__main__':
    reranker_path = os.getenv("BGEV3_RE_RANKER_PATH", "rakhuynh/bgev2m3_finetune_wdm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = BGEv2m3Reranker(reranker_path, device=device)

    docs = [
        "Introduction to hybrid retrieval.",
        "Overview of dense vs sparse.",
        "BGE model details.",
        "Use cases of reranking.",
        "Evaluation metrics."
    ]
    top_docs = reranker.bgev2m3_rerank("What is hybrid retrieval?", docs, top_k=3)
    print("Top documents:", top_docs)
