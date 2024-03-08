import csv
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import numpy as np


@dataclass
class Instance:
    instance_text: str
    instance_label: str

class NLPDataset(Dataset):

    def __init__(self):
        self.instances = []
        self.text_vocab = 0
        self.label_vocab = 0

    def from_file(self, path):
        self.instances = []
        instances_i = read_instances_from_file(path)
        for instance in instances_i:
            self.instances.append((instance.instance_text,instance.instance_label))
        text_freq, label_freq = freq_dic(instances_i)

        self.text_vocab = Vocab(text_freq, max_size=-1, min_freq=0)
        self.label_vocab = Vocab(label_freq, max_size=-1, min_freq=0, labels=True)   
        return self

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        text,label = self.instances[idx]
        text_numer = self.text_vocab.encode(text)
        label_numer = self.label_vocab.encode(label)
        return text_numer, label_numer

def freq_dic(instances):
    text_dict = {}
    label_dict = {}
    for instance in instances:
        for word in instance.instance_text:
            if word in text_dict:
                text_dict[word] += 1
            else:
                text_dict[word] = 1
        
        if instance.instance_label in label_dict:
            label_dict[instance.instance_label] += 1
        else:
            label_dict[instance.instance_label] = 1
    return dict(sorted(text_dict.items(), key=lambda item: item[1], reverse=True)), dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))

class Vocab:
    def __init__(self, freq, max_size=-1, min_freq=0, labels=False):
        if not labels:
            self.itos = {0: '<pad>', 1: '<unk>'}
            for idx, key in enumerate(freq.keys()):
                self.itos[idx+2] = key
        else:
            self.itos = {}
            for idx, key in enumerate(freq.keys()):
                self.itos[idx] = key
        self.stoi = {y: x for x, y in self.itos.items()}

    def encode(self, text):
        if isinstance(text, str):
            return torch.tensor(self.stoi[text])
        else:    
            encode_list = []
            for word in text:
                if word in self.stoi:
                    encode_list.append(self.stoi[word])
                else:
                    encode_list.append(1)
        return torch.tensor(encode_list)

def read_instances_from_file(path):
    instances = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            text = row[0].lower().split()
            label = row[1]
            instances.append(Instance(text, label[1:]))
    return instances

def create_embed_matrix(vocab, random=False):
    n = len(vocab.stoi)
    width = 300
    embed_matrix = torch.randn(n,width)

    embed_matrix[0] = torch.zeros_like(embed_matrix[0])
    if not random:
        with open('vekRep.txt') as f:
            lines = f.readlines()
            for row in lines:
                text = row.split()
                word = text[0]
                if word in vocab.stoi:
                    idx = vocab.stoi[word]
                    float_list = list(map(float, text[1:]))
                    embed_matrix[idx] = torch.tensor(float_list)
    return embed_matrix

def pad_collate_fn(batch, pad_index=0):
    texts = []
    labels = []
    lengths = []
    max_length = 0
    for el in batch:
        lengths.append(len(el[0]))
        if len(el[0]) > max_length:
            max_length = len(el[0])
    for el in batch:
        array_el = el[0]
        array_el_labels = el[1]
        if len(array_el) < max_length:
            array_el = torch.tensor(np.pad(array_el, (0, max_length-len(array_el)), 'constant'))
        texts.append(array_el)
        labels.append(array_el_labels)
    lengths = torch.tensor(torch.tensor(lengths))
    texts = torch.stack(texts, dim=0)
    labels = torch.stack(labels, dim=0)

    return texts, labels, lengths
    