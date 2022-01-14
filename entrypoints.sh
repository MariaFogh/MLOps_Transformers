# Making data
python src/data/make_dataset.py
# Parameter estimation
wandb sweep sweep.yaml
# Validation of model
python src/data/predict_model.py
