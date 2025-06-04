def is_in_top_k(retrieved_docs, ground_truth_doc, top_k=5):
    """Kiểm tra xem ground truth có nằm trong top-k tài liệu không."""
    for doc in retrieved_docs[:top_k]:
        if doc.strip() == ground_truth_doc.strip():
            return True
    return False


def calculate_hit_rate(df, retrieved_col='contexts', ground_truth_col='ground_truth', top_k=5):
    """Tính Hit Rate @ K"""
    df['hit'] = df.apply(
        lambda row: is_in_top_k(row[retrieved_col], row[ground_truth_col], top_k),
        axis=1
    )
    hit_rate = df['hit'].mean()
    return hit_rate


def reciprocal_rank(retrieved_docs, ground_truth_doc):
    """Tính Reciprocal Rank (1/rank) của ground truth trong danh sách tài liệu."""
    for i, doc in enumerate(retrieved_docs):
        if doc.strip() == ground_truth_doc.strip():
            return 1.0 / (i + 1)
    return 0.0


def calculate_mrr(df, retrieved_col='contexts', ground_truth_col='ground_truth'):
    """Tính Mean Reciprocal Rank (MRR)"""
    df['rr'] = df.apply(
        lambda row: reciprocal_rank(row[retrieved_col], row[ground_truth_col]),
        axis=1
    )
    mrr = df['rr'].mean()
    return mrr