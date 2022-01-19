import sys
sys.path.append('./src/models/')
sys.path.append('./src/data/')
print(sys.path)
from make_dataset import save_datasets
from train_model import train_model
from predict_model import test_model

if __name__ == "__main__":
    print("Running entrypoints.py")
    save_datasets()
    train_model()
    test_model()
