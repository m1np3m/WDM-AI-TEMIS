from configs.config import *
from libs.common import *
from utils.format_utils import *
# from utils.extract_tables import full_pipeline
from utils.rag_evaluation import *
from utils.rag_qdrant_utils import *
load_dotenv(find_dotenv())

import pandas as pd
import os

csv_files = [file for file in os.listdir("exps") if file.endswith(".csv")]
experiments = []
for file in csv_files:
    df = pd.read_csv(f"exps/{file}")
    experiments.append(df)
    
    
import matplotlib.pyplot as plt

experiment_names = [file.split('_')[-1][:-4] for file in csv_files]
metrics_to_plot = ["context_precision", "context_recall", "hit_rate", "mrr"]


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
        ax.bar_label(container, fmt="%.2f", rotation=60)

    plt.tight_layout()
    plt.savefig(f"exps/comparison_plot.png")
    
    
plot_experiment_comparison(experiments, experiment_names, metrics_to_plot)