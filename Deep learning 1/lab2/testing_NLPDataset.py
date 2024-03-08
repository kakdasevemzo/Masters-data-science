from utils import Instance, read_instances_from_file, NLPDataset, freq_dic, Vocab, create_embed_matrix, pad_collate_fn
import torch
import torch.nn as nn
train_dataset = NLPDataset().from_file('data\sst_train_raw.csv')

instance_text, instance_label = train_dataset.instances[3]
# Referenciramo atribut klase pa se ne zove nadjačana metoda
print(f"Text: {instance_text}")
print(f"Label: {instance_label}")

numericalized_text, numericalized_label = train_dataset[3]
# Koristimo nadjačanu metodu indeksiranja
print(f"Numericalized text: {numericalized_text}")
print(f"Numericalized label: {numericalized_label}")

print(f"Numericalized text: {train_dataset.text_vocab.encode(instance_text)}")
print(f"Numericalized label: {train_dataset.label_vocab.encode(instance_label)}")
numbers = train_dataset.text_vocab.encode(instance_text)

embed_matrix= create_embed_matrix(vocab=train_dataset.text_vocab)

for el in numbers:
    print(embed_matrix[el])