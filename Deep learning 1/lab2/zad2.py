
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import NLPDataset, create_embed_matrix, pad_collate_fn
import sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class Args:
    def __init__(self):
        self.seed = None
        self.lr = None
        self.batch_size_train = 10
        self.batch_size_valid = 32
        self.epochs = 5


class MeanPoolModel(nn.Module):
    def __init__(self, args, embed_matrix):
        super(MeanPoolModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_matrix,padding_idx=0,freeze=True)
        self.fc1 = nn.Linear(300, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)
    def forward(self, x):
        x = torch.mean(x,dim=1).T
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train(model, data, optimizer, criterion, args):
  model.train()
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    x,y,z = batch
    indexed_data = torch.zeros(300, torch.max(z), len(z))
    for data_slice, tensor in enumerate(x):
       for row, index in enumerate(tensor):
          indexed_data[ : , row , data_slice] = model.embedding.weight[index]
    x = indexed_data
    logits = model(x).squeeze()
    y = y.to(torch.float32)
    loss = criterion(logits, y)
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    # ...
    


def evaluate(model, data, criterion, args):
  model.eval()
  total_loss = 0
  total_correct = 0
  total_samples = 0
  predicted_labels = []
  true_labels = []  
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      x,y,z = batch
      indexed_data = torch.randn(300, torch.max(z), len(z))
      for data_slice, tensor in enumerate(x):
        for row, index in enumerate(tensor):
            indexed_data[ : , row , data_slice] = model.embedding.weight[index]
      x = indexed_data
      logits = model(x).squeeze()
      logits = torch.sigmoid(logits)
      loss = criterion(logits, y.to(torch.float32))
      pred_labels = (logits > 0.5).int()
      total_loss += loss.item() * len(z)
      predicted_labels.extend(pred_labels.tolist())
      true_labels.extend(y.tolist())
      
      # ...
  predicted_labels = torch.tensor(predicted_labels)
  true_labels = torch.tensor(true_labels)
  accuracy = (predicted_labels == true_labels).float().mean()
  print("Accuracy:", accuracy.item())

def load_datasets(batch_size_train=10 , batch_size_valid=32 ,shuffle=True):
    
    train_dataset = NLPDataset().from_file('data/sst_train_raw.csv')
    valid_dataset = NLPDataset().from_file('data/sst_valid_raw.csv')
    test_dataset = NLPDataset().from_file('data/sst_test_raw.csv')
    valid_dataset.text_vocab = train_dataset.text_vocab
    test_dataset.text_vocab = train_dataset.text_vocab
    valid_dataset.label_vocab = train_dataset.label_vocab
    test_dataset.label_vocab = train_dataset.label_vocab

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, 
                              shuffle=shuffle, collate_fn=pad_collate_fn)  
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size_valid, 
                              shuffle=shuffle, collate_fn=pad_collate_fn) 
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_valid, 
                              shuffle=shuffle, collate_fn=pad_collate_fn)
    
    return train_data_loader, valid_data_loader, test_data_loader

def main(args):
  seed = args.seed
  np.random.seed(seed)
  torch.manual_seed(seed)
  train_dataset_embed = NLPDataset().from_file('data/sst_train_raw.csv')
  train_dataset, valid_dataset, test_dataset = load_datasets(args.batch_size_train,args.batch_size_valid)
  embed_matrix = create_embed_matrix(vocab=train_dataset_embed.text_vocab)
  model = MeanPoolModel(args=args, embed_matrix=embed_matrix)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  for epoch in range(args.epochs):
    train(model=model, data=train_dataset, optimizer=optimizer, criterion=criterion,args=args)
    evaluate(model=model, data=valid_dataset, criterion=criterion, args=args)
  predicted_labels = []
  true_labels = []  
  for batch_num, batch in enumerate(test_dataset):
      x,y,z = batch
      indexed_data = torch.randn(300, torch.max(z), len(z))
      for data_slice, tensor in enumerate(x):
        for row, index in enumerate(tensor):
            indexed_data[ : , row , data_slice] = model.embedding.weight[index]
      x = indexed_data
      logits = model(x).squeeze()
      logits = torch.sigmoid(logits)
      loss = criterion(logits, y.to(torch.float32))
      pred_labels = (logits > 0.5).int()
      predicted_labels.extend(pred_labels.tolist())
      true_labels.extend(y.tolist())
      
      # ...
  predicted_labels = torch.tensor(predicted_labels)
  true_labels = torch.tensor(true_labels)
  accuracy = (predicted_labels == true_labels).float().mean()
  print("Test accuracy:", accuracy.item())

    



args = Args()
for arg in sys.argv[1:]:
  print(arg)
  if arg.startswith("--seed="):
    _, value = arg.split("=")
    args.seed = int(value)
  if arg.startswith("--lr="):
    _, value = arg.split("=")
    args.lr = float(value)
  if arg.startswith("--batch_size_train="):
    _, value = arg.split("=")
    args.batch_size_train = int(value)
  if arg.startswith("--batch_size_valid="):
    _, value = arg.split("=")
    args.batch_size_valid = int(value)
  if arg.startswith("--epochs="):
    _, value = arg.split("=")
    args.epochs = int(value)
    # Call the main function with the provided arguments
main(args)