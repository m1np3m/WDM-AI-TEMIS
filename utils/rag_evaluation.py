import pandas as pd
# from datasets import Dataset
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
from ragas.evaluation import evaluate, RunConfig
from ragas.metrics import (
    NonLLMContextRecall,
    NonLLMContextPrecisionWithReference,
)
from .rag_metrics import *
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_ragas_eval(
    eval_df,
    collection_name,
    doc_retrieval_function,
    embedding_model_name,
    response_generation_function = "",
    num_docs=5,
    reranker_function=None,
    path="ragas_eval.csv",
    use_optimized_metrics=True,
    max_concurrent_requests=20  # Limit concurrent requests to prevent RAM overflow
):
    eval_df = eval_df.rename(columns={"question": "input", "answer": "ground_truth", "context": "reference_contexts"})
    print("Running Context Retrieve...")

    # Optimize context retrieval with controlled concurrency
    async def async_retrieve_contexts():
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def retrieve_single_context(question):
            async with semaphore:  # Limit concurrent operations
                # Run retrieval in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:  # Limit threads per operation
                    return await loop.run_in_executor(
                        executor,
                        lambda: doc_retrieval_function(collection_name, question, embedding_model_name, num_documents=num_docs, reranker=reranker_function)
                    )
        
        # Process all questions with controlled concurrency
        tasks = [retrieve_single_context(question) for question in eval_df['input']]
        print(f"Processing {len(tasks)} questions with max {max_concurrent_requests} concurrent requests...")
        return await asyncio.gather(*tasks)

    # Run async context retrieval
    contexts = asyncio.run(async_retrieve_contexts())
    eval_df['contexts'] = contexts
    
    eval_df["reference_contexts"] = eval_df["reference_contexts"].apply(lambda x: [x] if isinstance(x, str) else x)

    print("Running Oputput Generate...")

    if 'output' not in eval_df.columns:
        eval_df['output'] = eval_df.apply(
            lambda row: response_generation_function, #generate_model_output(row['input'], row['contexts'])
            axis=1
        )

    eval_df = eval_df.rename(columns={
        "input": "user_input",     
        "output": "response",        
        "ground_truth": "ground_truth",  
        "contexts": "retrieved_contexts"
    })
    dataset = Dataset.from_pandas(eval_df[["user_input", "response", "ground_truth", "retrieved_contexts", "reference_contexts"]])
    context_precision =NonLLMContextPrecisionWithReference()
    context_recall = NonLLMContextRecall()

    print("Running RAGAS evaluation...")
    run_config = RunConfig(timeout=120, max_workers=63)

    results = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall
        ]
    )

    df_results = results.to_pandas()
    
    # Use optimized metrics if enabled
    if use_optimized_metrics:
        print("Running optimized semantic metrics calculation (async only)...")
        df_results['hit_rate'] = asyncio.run(async_semantic_hit_rate(eval_df, top_k=num_docs))
        df_results['mrr'] = asyncio.run(async_semantic_mrr(eval_df))
    else:
        print("Running original semantic metrics calculation...")
        df_results['hit_rate'] = calculate_semantic_hit_rate(eval_df, top_k=num_docs)
        df_results['mrr'] = calculate_semantic_mrr(eval_df)
    
    df_results = df_results.rename(columns={
        "non_llm_context_precision_with_reference": "context_precision",     
        "non_llm_context_recall": "context_recall"
    })
    df_results.to_csv(path, index=False)

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
        ylim = None  
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