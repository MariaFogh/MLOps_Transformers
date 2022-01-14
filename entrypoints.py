from src.data.make_dataset import main
from src.models.train_model import train_model
from src.models.predict_model import test_model

if __name__ == "__main__":
    main()
    train_model()
    test_model()
