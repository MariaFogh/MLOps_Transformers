# 02476 Machine Learning Operations: Final Project

A project using the [Transformers framework](https://github.com/huggingface/transformers) (repository from the 🤗 Huggingface group)

## Group members
- Enny Tao (s174491)
- Anton Rydahl
- Laurits Fromberg (s174512)
- Maria Fogh (s174498)

## Overall objective

This project will investigate how to use Machine Learning Operations tools on a project. Specifically, we work with the [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge) competition from Kaggle.

## Framework

In this project we intend to use the Transformers framework, which provides an array of pre-trained models,
for various uses such as for instance text, vision and audio. We will in this project work with text data in
order to classify movie-ratings from IMDB, hence the reason for exploiting such framework.
We will more specifically employ the model BERT (Bidirectional Encoder Representations From Transformers) 
developed by Google, which is a transformer-based machine-learning model for Natural Language Processing (NLP).

## Data

The data is found in the [Huggingface Datasets library](https://github.com/huggingface/datasets/) and consists of Imdb ratings. They have provided 25,000 highly polar movie reviews for training and 25,000 for testing. They also include an unlabelled version of the data. Specifically, the dataset contains a ‘text’ column, which is a string of the written review in itself, and a ‘label’ column with possible values 0 and 1, describing whether it is a negative or positive review, respectively.

## Model


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
