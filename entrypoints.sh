# Making data
python src/data/make_dataset.py
# Parameter estimation
NUM=10
SWEEPID="axhg8mqz"
wandb agent --count $NUM $SWEEPID
# Validation of model
python src/data/predict_model.py
