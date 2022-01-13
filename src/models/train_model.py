from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from datasets import load_metric

import torch

from tqdm.auto import tqdm

from os.path import exists


def train_model():
    input_filepath = './data/processed'
    model_from_path = './models/pretrained_bert'
    model_to_path = './models/finetuned_bert'
    small_train_dataset = torch.load(input_filepath+'/train_small.pt')
    print("The trining set concists of")
    print(small_train_dataset)
    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=8)

    model = None
    if (not exists(model_from_path)):
        print(f"Pretrained model {model_from_path} does not exists")
        print("Downloading pretrained BERT model from Transformers")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2)

        model.save_pretrained(model_from_path)
        print(
            f"Downloaded and saved pretrained model to path {model_from_path}")
    else:
        print(f"Using pretrained BERT model from path {model_from_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_from_path)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.save_pretrained("./models/trained_bert")

    progress_bar = tqdm(range(num_training_steps))

    print(f"Training {num_epochs} epochs")
    model.train()
    for epoch in range(num_epochs):
        metric = load_metric("accuracy")
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions,
                             references=batch["labels"])
        metric.compute()
        print(f"\tFinished epoch {epoch+1} of {num_epochs}")
    model.save_pretrained(model_to_path)
    print(f"Saved trained BERT model to path {model_to_path}")


if __name__ == '__main__':
    train_model()
