from utils import Instance, read_instances_from_file, NLPDataset, freq_dic, Vocab, create_embed_matrix
import torch
import torch.nn as nn

train_instances = read_instances_from_file('data\sst_train_raw.csv')
# val_instances = read_instances_from_file('data\sst_valid_raw.csv')
# test_instances = read_instances_from_file('data\sst_test_raw.csv')

# print(train_instances[-1])

text_freq, label_freq = freq_dic(train_instances)

text_vocab = Vocab(text_freq, max_size=-1, min_freq=0)
label_vocab = Vocab(label_freq, max_size=-1, min_freq=0, labels=True)

print(f"Text: {train_instances[3].instance_text}")
print(f"Label: {train_instances[3].instance_label}")

print(text_vocab.encode(train_instances[3].instance_text))
print(label_vocab.encode(train_instances[3].instance_label))

print(len(text_vocab.itos))

embed_matrix = create_embed_matrix(text_vocab)
print(create_embed_matrix(text_vocab)[0,:])
print(create_embed_matrix(text_vocab)[2,:])

embedding = nn.Embedding(embeddings=embed_matrix, padding_idx=0, freeze=True)