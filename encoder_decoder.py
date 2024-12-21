
# load the data from the csv files generated in pretrain

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

# define dataset

print("defining the dataset")
class TranslationDataset(Dataset):
    def __init__(self, vocab = None):
        if vocab != None:
            self.vocab = vocab
        else:
            self.vocab = {"<PAD>": 0, "-->": 1}  # special symbols which may not be in the data but need to be included
    def make_vocab(self, pretrain_data, train_data): # we will include both pretrain and train data in our vocab
        # pretain data is expected to be in the form 

from sklearn.model_selection import train_test_split