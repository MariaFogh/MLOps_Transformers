import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

model_path = "./models/finetuned_bert/"

model = AutoModelForSequenceClassification.from_pretrained(model_path, torchscript=True)

small_train_dataset = torch.load("./data/processed" + "/train_small.pt")

num_batches = 1000 / 1
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=1)

for ite, batch in enumerate(train_dataloader):
    batch = {k: v.to("cpu") for k, v in batch.items()}
    outputs = model(**batch)
    break

inp = batch

script_model = torch.jit.trace(model, list(batch.values()))
script_model.save("deployable_model.pt")
