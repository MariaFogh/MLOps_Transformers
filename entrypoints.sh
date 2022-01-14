# Making data
python src/data/make_dataset.py
# Parameter estimation
#NUM=5
#SWEEPID="axhg8mqz"
#wandb agent --count $NUM $SWEEPID
python src/models/train_model.py
# Validation of model
python src/models/predict_model.py
