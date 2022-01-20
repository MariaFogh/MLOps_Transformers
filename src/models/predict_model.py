import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from wandb_helpers import wandb_arg_parser

import wandb


def test_model():
    """
    Tests the trained model using the small version of the evaluation dataset.
    The accuracy and loss are calculated and logged using Wandb.
    """
    input_filepath = "./data/processed"
    model_path = "./models/finetuned_bert"
    small_eval_dataset = torch.load(input_filepath + "/eval_small.pt")
    print("The test set concists of")
    print(small_eval_dataset)

    config = wandb_arg_parser()
    batch_size = config.batch_size
    num_batches = 1000 / batch_size
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()
    accuracy = 0.0
    validation_loss = 0.0
    for ite, batch in enumerate(eval_dataloader):
        print(f"\tRunning batch {ite+1} of {num_batches}", end="\r")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy += sum(predictions == batch["labels"]) / predictions.numel()
        loss = outputs.loss
        validation_loss += loss.item()

    accuracy = 100 * (accuracy / num_batches)
    wandb.log(
        {
            "validation_loss": validation_loss / num_batches,
            "validation_accuracy": accuracy,
        }
    )


if __name__ == "__main__":
    test_model()
