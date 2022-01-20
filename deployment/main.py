"""
Script used for deploying the model in Cloud along with requirements.txt
"""

import pickle

import torch
from google.cloud import storage
from transformers import AutoTokenizer

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


def make_html(msg):
    my_model = pickle.loads(blob.download_as_string())
    prediction = predictor(my_model, msg)

    ret = "<h2>You are requesting whether the following review from IMDB is positive or negative:</h2><br>"
    ret += f"<i>{msg}</i><br>"
    if prediction > 0.5:
        ret += "<h2>BERT predicts that the review is: Positive 🤗</h2><br>"
        ret += '<img src="https://i.pinimg.com/736x/6a/50/88/6a508859a9a7fecd93672cf35249f8fb.jpg" width="500">'
    else:
        ret += "<h2>BERT predicts that the review is: Negative</h2><br>"
        ret += '<img src="https://venturephotography.com.au/wp-content/uploads/2017/05/del_badmeme.jpg" width="500">'
    return ret


def main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and "message" in request.args:
        return make_html(request.args.get("message"))
    elif request_json and "message" in request_json:
        return make_html(request_json["message"])
    else:
        return "Please provide URL parameter, messaage!"
