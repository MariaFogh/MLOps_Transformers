from src.data.make_dataset import save_datasets
from src.models.train_model import train_model
from src.models.predict_model import test_model

if __name__ == "__main__":
    print("Running entrypoints.py")
    save_datasets()
    train_model()
    test_model()
