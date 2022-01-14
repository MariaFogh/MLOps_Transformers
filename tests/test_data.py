import torch

folder = 'C:/Users/Bruger/Documents/MLops/MLOps_Transformers/data/processed'

train_set = torch.load(folder + '/train.pt')
test_set = torch.load(folder + '/eval.pt')

# Define size of datasets
N_train = 25000
N_test = 25000

def test_data_length():
   assert len(train_set) == N_train, 'Dataset does not have the correct number of samples'
   assert len(test_set) == N_test, 'Dataset does not have the correct number of samples'
   
