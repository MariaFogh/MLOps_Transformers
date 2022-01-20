from transformers import AutoModelForSequenceClassification
import pickle

model_path = "./models/finetuned_bert"

model = AutoModelForSequenceClassification.from_pretrained(model_path)


with open("./models/model.pkl", "wb") as file:
    pickle.dump(model, file)
