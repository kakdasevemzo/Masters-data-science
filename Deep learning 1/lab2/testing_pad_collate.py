from utils import Instance, read_instances_from_file, NLPDataset, freq_dic, Vocab, create_embed_matrix, pad_collate_fn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

batch_size = 2 # Only for demonstrative purposes
shuffle = False # Only for demonstrative purposes
train_dataset = NLPDataset().from_file('data/sst_train_raw.csv')
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=pad_collate_fn)
texts, labels, lengths = next(iter(train_data_loader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")