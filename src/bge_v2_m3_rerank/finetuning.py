import os
import random
import torch
import pandas as pd
import warnings
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

warnings.filterwarnings("ignore", category=UserWarning)

def build_pairwise_df(eval_df, num_negatives=1):
    """
    (query, doc) with label 1 (positive), 0 (negative).
    """
    pairs = []
    contexts = eval_df['context'].tolist()
    for _, row in eval_df.iterrows():
        q = row['question']
        pos = row['context']
        pairs.append({'query': q, 'doc': pos, 'label': 1})
        negs = random.sample([c for c in contexts if c != pos], num_negatives)
        for neg in negs:
            pairs.append({'query': q, 'doc': neg, 'label': 0})
    return pd.DataFrame(pairs)


def preprocess_function(examples, tokenizer, max_length=256):
    texts = [f"Query: {q} Document: {d}" for q, d in zip(examples['query'], examples['doc'])]
    tok = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    tok['labels'] = [float(l) for l in examples['label']]
    return tok


if __name__ == '__main__':
    dir_ =" "
    data_path = f'{dir_}/fixed_label_QA.json' # lấy data khác để finetune lại vì lúc finetune không có data nên chị lấy data này
    model_name = 'BAAI/bge-reranker-v2-m3'
    output_dir = f'{dir_}/bgev2m3_finetune'
    num_negatives = 2

    # Load data
    eval_df = pd.read_json(data_path, encoding='utf-8')
    if 'label' not in eval_df.columns:
        eval_df['label'] = 1

    # Create pair data
    train_df = build_pairwise_df(eval_df, num_negatives=num_negatives)
    ds = Dataset.from_pandas(train_df)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type='regression',
        num_labels=1
    )
    model.to(device)

    # Tokenize dataset
    tokenized = ds.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=ds.column_names
    )

    # Training arguments (no intermediate checkpoints)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2 if device.type=='cpu' else 8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=500,
        save_strategy='no',
        fp16=device.type=='cuda',
        gradient_checkpointing=device.type=='cuda',
        disable_tqdm=True,
        dataloader_pin_memory=device.type=='cuda',
        dataloader_num_workers=os.cpu_count() or 1,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    
    # Save model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Fine-tune completed. Model and checkpoints saved to {output_dir}")
