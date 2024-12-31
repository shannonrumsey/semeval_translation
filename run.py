from dataset import TranslationDataset, get_semeval_train, collate_fn, get_pretrain
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import json
import sentencepiece as spm
import random
import ast

pretrain_list = get_pretrain()
semeval_train = get_semeval_train()

# # Encode and load pretrain data
pretrain_dataset = TranslationDataset()
pretrain_dataset.make_vocab(pretrain_list, semeval_train)
# pretrain_dataset.encode_pretrain(pretrain_list)
# # No padding in the pretrainig data
# pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
# pretrain_dataset.make_sure_everythings_alligned_properly()

# # Encode and load train data
# semeval_dataset = TranslationDataset()
# semeval_dataset.make_vocab(pretrain_list, semeval_train)
# train_data = get_semeval_train()
# semeval_dataset.encode_semeval(train_data)
# train_loader = DataLoader(semeval_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
# semeval_dataset.make_sure_everythings_alligned_properly()
