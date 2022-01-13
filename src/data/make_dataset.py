# -*- coding: utf-8 -*-
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    output_filepath = "./data/processed"

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    raw_datasets = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    torch.save(small_train_dataset, output_filepath + "/train_small.pt")
    torch.save(small_eval_dataset, output_filepath + "/eval_small.pt")
    torch.save(full_train_dataset, output_filepath + "/train.pt")
    torch.save(full_eval_dataset, output_filepath + "/eval.pt")


if __name__ == "__main__":
    main()
