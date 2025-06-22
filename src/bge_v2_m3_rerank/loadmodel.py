from FlagEmbedding import FlagReranker
from typing import List

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# Ham trong rerank_utils Loc dinh
def bge_reranker_fn(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.bge_reranker.predict(pairs, max_length=1024)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"[BGE Reranker] Error: {e}")
            return documents[:top_k]
    


