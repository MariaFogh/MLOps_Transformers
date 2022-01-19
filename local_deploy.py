import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2, torchscript=True
)
script_model = torch.jit.trace(model)
script_model.save("deployable_model.pt")
