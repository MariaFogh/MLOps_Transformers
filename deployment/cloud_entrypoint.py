from transformers import AutoTokenizer

from google.cloud import storage
import pickle
import torch

BUCKET_NAME = "bert-bucket-mlops"
MODEL_FILE = "model.pkl"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)


def predictor(model, str):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    token_string = tokenizer([str])

    for k in list(token_string.keys()):
        token_string[k] = torch.IntTensor(token_string[k])

    output = model(**token_string)

    logits = output.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions.tolist()[0]


my_model = pickle.loads(blob.download_as_string())
my_string = "Don't Look Up is a political satire, one of the better ones to be honest. You can't help it to compare the different characters with real people, that was probably exactly the point of making this movie."
prediction = predictor(my_model, my_string)

print(prediction)
