from os.path import exists

import torch
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_scheduler)
from wandb_helpers import wandb_arg_parser

import wandb


def train_model():
    """
    Train the model using the small version of the training dataset.
    The accuracy and loss are calculated and logged using WandB.
    """
    input_filepath = "./data/processed"
    model_from_path = "./models/pretrained_bert"
    model_to_path = "./models/finetuned_bert"
    small_train_dataset = torch.load(input_filepath + "/train_small.pt")
    print("The training set concists of")
    print(small_train_dataset)

    # Getting arguments from WandB
    config = wandb_arg_parser()
    learning_rate = config.learning_rate
    num_epochs = config.epochs
    batch_size = config.batch_size

    num_batches = 1000 / batch_size
    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=batch_size
    )

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
    # progress_bar = tqdm(range(num_training_steps))

    # Training
    print(f"Training {num_epochs} epochs")
    model.train()
    for epoch in range(num_epochs):
        print(f"Running epoch {epoch+1} of {num_epochs}")
        # metric = load_metric("accuracy")
        accuracy = 0.0
        train_loss = 0.0
        for ite, batch in enumerate(train_dataloader):
            print(f"\tRunning batch {ite+1} of {num_batches}", end="\r")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            # Updating metrics
            predictions = torch.argmax(outputs.logits, dim=-1)
            # metric.add_batch(predictions=predictions, references=batch["labels"])
            accuracy += sum(predictions == batch["labels"]) / predictions.numel()
            train_loss += loss.item()

            # Continuing to next batch
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)

        # accuracy = metric.compute()
        accuracy = 100 * (accuracy / num_batches)
        print(f"\n\tFinished epoch {epoch+1} of {num_epochs}:")
        print(f"\t\tAccuracy:{accuracy} %")
        print(f"\t\tLoss:{train_loss}\n")
        wandb.log(
            {"training_loss": train_loss / num_batches, "training_accuracy": accuracy}
        )
        if (epoch == 0) or ((epoch + 1) % 5 == 0) or (epoch == (num_epochs - 1)):
            model.to(torch.device("cpu"))
            model.save_pretrained(model_to_path)
            print(f"Saved trained BERT model to path {model_to_path}")
            wandb.log_artifact(
                model_to_path, name=f"finetuned_bert_{epoch+1}", type="model"
            )
            print(f"Uploaded trained BERT model to WandB from path {model_to_path}")
            model.to(device)


if __name__ == "__main__":
    train_model()
