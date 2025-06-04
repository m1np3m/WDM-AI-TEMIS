import pandas as pd
from datasets import Dataset
from ragas.evaluation import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    BleuScore,
    RougeScore,

)
import requests
import json
import matplotlib.pyplot as plt
from .rag_metrics import *
def generate_model_output(user_query, documents):
    """
    Gửi prompt đến Ollama và nhận về một đoạn phản hồi hoàn chỉnh.
    
    Tham số:
        prompt (str): Câu hỏi hoặc yêu cầu bạn muốn gửi đến mô hình.

    Trả về:
        str: Phản hồi hoàn chỉnh từ mô hình.
    """
    full_response = ""
    context = f"""
   You are an assistant can  answer questions based on the content of documents and filter information to give the best answer.
    Here are some document fragments retrieved from a PDF document:
    {documents}
    Please concatenate all the table parts below row by row into one single table, making sure they form a single table. The result should be the concatenated table and any completely separate tables remaining.
    {user_query}
        """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": context,
        "stream": True
    }

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        chunk = data['response']
                        full_response += chunk
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue  

    return full_response.strip()

def run_ragas_eval(
    eval_df,
    collection_name,
    doc_retrieval_function,
    embedding_model_name,
    num_docs=5,
    path="ragas_eval.csv"
):
    eval_df = eval_df.rename(columns={"question": "input", "answer": "ground_truth"})

    eval_df['contexts'] = eval_df['input'].apply(
        lambda q: doc_retrieval_function(collection_name, q, embedding_model_name, num_documents=num_docs)
    )

    if 'output' not in eval_df.columns:
        eval_df['output'] = eval_df.apply(
            lambda row: generate_model_output(row['input'], row['contexts']),
            axis=1
        )

    eval_df = eval_df.rename(columns={
        "input": "user_input",     
        "output": "response",        
        "ground_truth": "ground_truth",  
        "contexts": "contexts"
    })
    dataset = Dataset.from_pandas(eval_df[["user_input", "response", "ground_truth", "contexts"]])

    print("Running RAGAS evaluation...")

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            BleuScore(),
            RougeScore(),
        ]
    )

    df_results = results.to_pandas()
    df_results.to_csv(path, index=False)
    df_results['hit_rate'] = calculate_hit_rate(eval_df)
    df_results['mrr'] = calculate_mrr(eval_df)

    print("Evaluation completed. Results saved to", path)
    return df_results



def plot_experiment_comparison(experiment_results_list, experiment_names, metrics_to_plot):
    """
    experiment_results_list: list các DataFrame kết quả đánh giá từ run_ragas_eval
    experiment_names: list tên experiment tương ứng
    metrics_to_plot: list các tên metric cần vẽ
    
    Hàm sẽ tạo DataFrame trung bình các metric theo experiment và vẽ biểu đồ so sánh.
    """
    stats = []
    for name, df in zip(experiment_names, experiment_results_list):
        means = df[metrics_to_plot].mean().tolist()
        stats.append([name] + means)

    stats_df = pd.DataFrame(stats, columns=['Experiment'] + metrics_to_plot)
    stats_df = stats_df.set_index('Experiment').T 
    standard_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
                        'context_relevancy', 'answer_similarity', 'answer_correctness']
    
    special_metrics = {
        'hit_rate': 'Hit Rate',
        'mrr': 'MRR'
    }
    has_special_metric = any(metric in special_metrics for metric in metrics_to_plot)
    is_all_standard = all(metric in standard_metrics for metric in metrics_to_plot)

    if has_special_metric and not is_all_standard:
        title = "Comparison of Retrieval & RAG Metrics"
        ylabel = "Score / Value"
        ylim = None  # Không giới hạn nếu có metric mở rộng
    else:
        title = "Comparison of RAGAS Evaluation Metrics"
        ylabel = "Average Score"
        ylim = (0, 1)

    ax = stats_df.plot(
        kind='bar',
        figsize=(10, 6),
        ylim=(0, 1),
        width=0.7,
        edgecolor='white',
        linewidth=2,
        colormap='tab10',
        title="Comparison of RAGAS Evaluation Metrics"
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Average Score")
    ax.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", rotation=90)

    plt.tight_layout()
    plt.show()