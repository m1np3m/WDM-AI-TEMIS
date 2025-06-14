from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  

def semantic_hit(retrieved_docs, ground_truth_docs, top_k=5, threshold=0.8):
    """Kiểm tra semantic similarity với cosine @top_k"""
    retrieved_docs = retrieved_docs[:top_k]
    for ret in retrieved_docs:
        for gt in ground_truth_docs:
            score = util.cos_sim(model.encode(ret), model.encode(gt)).item()
            if score >= threshold:
                return True
    return False


def calculate_semantic_hit_rate(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', top_k=5, threshold=0.8):
    """Tính semantic Hit Rate @ K bằng cosine similarity"""
    df['semantic_hit'] = df.apply(
        lambda row: semantic_hit(row[retrieved_col], row[ground_truth_col], top_k, threshold),
        axis=1
    )
    return df['semantic_hit'].mean()


def semantic_rr(retrieved_docs, ground_truth_docs, threshold=0.8):
    """Tính Reciprocal Rank theo semantic similarity"""
    for i, ret in enumerate(retrieved_docs):
        for gt in ground_truth_docs:
            score = util.cos_sim(model.encode(ret), model.encode(gt)).item()
            if score >= threshold:
                return 1.0 / (i + 1)
    return 0.0


def calculate_semantic_mrr(df, retrieved_col='retrieved_contexts', ground_truth_col='reference_contexts', threshold=0.8):
    """Tính Mean Reciprocal Rank theo semantic similarity"""
    df['semantic_rr'] = df.apply(
        lambda row: semantic_rr(row[retrieved_col], row[ground_truth_col], threshold),
        axis=1
    )
    return df['semantic_rr'].mean()
