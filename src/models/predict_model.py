from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from datasets import load_metric

import torch

from tqdm.auto import tqdm


def train_model():
    input_filepath = './data/processed'
    small_eval_dataset = torch.load(input_filepath+'/eval_small.pt')
    print(small_eval_dataset)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == '__main__':
    train_model()
