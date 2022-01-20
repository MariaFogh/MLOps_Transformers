import pickle

from transformers import AutoModelForSequenceClassification

model_path = "./models/finetuned_bert"

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use pickle for saving the model
with open("./models/model.pkl", "wb") as file:
    pickle.dump(model, file)
