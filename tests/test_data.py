import torch

folder = 'C:/Users/Bruger/Documents/MLops/MLOps_Transformers/data/processed'

train_set = torch.load(folder + '/train.pt')
test_set = torch.load(folder + '/eval.pt')


# Define size of datasets
N_train = 25000
N_test = 25000
