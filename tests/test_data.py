import torch
import os
import pytest

folder = "data/processed"

# Define size of datasets
N_train = 1000
N_test = 1000


@pytest.mark.skipif(not os.path.exists(folder), reason="Data files not found")
def test_data_length():
    """
    Tests whether the data is the correct size to prepare for training.
    """
    train_data = torch.load(folder + "/train_small.pt")
    test_data = torch.load(folder + "/eval_small.pt")
    assert (
        len(train_data) == N_train
    ), "Dataset did not have the correct number of samples"
    assert (
        len(test_data) == N_test
    ), "Dataset did not have the correct number of samples"


@pytest.mark.skipif(not os.path.exists(folder), reason="Data files not found")
def test_dataset_features():
    """
    Tests whether the datasets have all necessary feature keys for the model.
    """
    train_data = torch.load(folder + "/train_small.pt")
    test_data = torch.load(folder + "/eval_small.pt")
    assert set(train_data.features.keys()) == set(
        ["attention_mask", "input_ids", "labels", "token_type_ids"]
    )

    assert set(test_data.features.keys()) == set(
        ["attention_mask", "input_ids", "labels", "token_type_ids"]
    )


@pytest.mark.skipif(not os.path.exists(folder), reason="Data files not found")
def test_dimensions():
    """
    Tests the dimensions of the features.
    """
    train_data = torch.load(folder + "/train_small.pt")
    test_data = torch.load(folder + "/eval_small.pt")

    for dataset in [train_data, test_data]:
        assert list(dataset["attention_mask"].size()) == [1000, 512]
        assert list(dataset["input_ids"].size()) == [1000, 512]
        assert list(dataset["labels"].size()) == [1000]
        assert list(dataset["token_type_ids"].size()) == [1000, 512]


@pytest.mark.skipif(not os.path.exists(folder), reason="Data files not found")
def test_labels():
    """
    Tests whether both positive and negative reviews are included in the datasets.
    """
    train_data = torch.load(folder + "/train_small.pt")
    test_data = torch.load(folder + "/eval_small.pt")

    assert all(i in train_data["labels"] for i in range(2))
    assert all(i in test_data["labels"] for i in range(2))
