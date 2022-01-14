import argparse
from os.path import exists
from os import environ

import torch
import wandb
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler


def wandb_arg_parser():
    environ["WANDB_API_KEY"] = "719c09fb68fba7368ba93fd0b304d7e1a2fb1a4a"
    environ["WANDB_MODE"] = "offline"

    parser = argparse.ArgumentParser()
    args, leftovers = parser.parse_known_args()
    config_defaults = {"learning_rate": 5e-5, "epochs": 2}

    if hasattr(args, "epochs"):
        config_defaults["epochs"] = args.epochs
    if hasattr(args, "learning_rate"):
        config_defaults["learning_rate"] = args.learning_rate

    wandb.init(config=config_defaults, project="MLOps Transformers Sweep")

    config = wandb.config
    return config


def train_model():
    input_filepath = "./data/processed"
    model_from_path = "./models/pretrained_bert"
    model_to_path = "./models/finetuned_bert"
    small_train_dataset = torch.load(input_filepath + "/train_small.pt")
    print("The trining set concists of")
    print(small_train_dataset)
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)

    model = None
    if not exists(model_from_path):
        print(f"Pretrained model {model_from_path} does not exists")
        print("Downloading pretrained BERT model from Transformers")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )

        model.save_pretrained(model_from_path)
        print(f"Downloaded and saved pretrained model to path {model_from_path}")
    else:
        print(f"Using pretrained BERT model from path {model_from_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_from_path)

    # Getting arguments from WandB
    config = wandb_arg_parser()
    learning_rate = config.learning_rate
    num_epochs = config.epochs

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Moving model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Initializing WandB
    wandb.watch(model, log_freq=100)
    progress_bar = tqdm(range(num_training_steps))

    # Training
    print(f"Training {num_epochs} epochs")
    model.train()
    for epoch in range(num_epochs):
        metric = load_metric("accuracy")
        train_loss = 0.0
        num_batches = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # Updating metrics
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            train_loss += loss.item()
            num_batches += 1

            # Continuing to next batch
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        accuracy = metric.compute()
        print(f"\tFinished epoch {epoch+1} of {num_epochs}:")
        print(f"Accuracy:{accuracy}")
        print(f"Loss:{train_loss}")
        wandb.log(
            {"training_loss": train_loss / num_batches, "training_accuracy": accuracy}
        )
    model.save_pretrained(model_to_path)
    print(f"Saved trained BERT model to path {model_to_path}")

    model.save_pretrained("./models/trained_bert")


if __name__ == "__main__":
    train_model()
