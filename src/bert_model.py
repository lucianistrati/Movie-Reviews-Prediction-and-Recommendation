import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
from transformers import BertForSequenceClassification, \
    BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from transformers import TrainingArguments
from datasets import load_metric


def train_semantical_model():
    batch_size = 8

    raw_datasets = load_dataset(
        "src/bert_dataset.py")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                               num_labels=10)

    def tokenize_function(examples):
        tokenizer = CamembertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]
    small_train_dataset = full_train_dataset
    small_eval_dataset = full_eval_dataset

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(evaluation_strategy="epoch",
                                      no_cuda=False, num_train_epochs=10,
                                      output_dir="../checkpoints/bert",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

def main():
    train_semantical_model()

if __name__ == '__main__':
    main()
