import torch
from torch import nn

from pathlib import Path
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter()

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['learning_rate'] = 1e-3
config['weight_decay'] = 1e-3

train_data = MNIST(
    root = DATA_DIR,
    train = True,                         
    transform = ToTensor(),         
)
test_data = MNIST(
    root = DATA_DIR, 
    train = False, 
    transform = ToTensor()
)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)

test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)
    
class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_in, fc1_width, class_count):
    super(CovolutionalModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool1 = nn.MaxPool2d(2,2)


    self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool2 = nn.MaxPool2d(2,2)


    self.flatten1 = nn.Flatten()

    # potpuno povezani slojevi
    self.fc1 = nn.Linear(fc1_in, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool1(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.conv2(h)
    h = self.maxpool2(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.flatten1(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits
    
model = CovolutionalModel(in_channels=1, conv1_width=16, conv2_width=32,fc1_in=7*7*32,fc1_width=512,class_count=10)

    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])

total_steps = len(train_loader)

loss_list = list()

for epoch in range(config['max_epochs']):
    for i, (images,labels) in enumerate(train_loader):
        
        outputs = model(images)
        loss = criterion(outputs,labels)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'epoch: {epoch+1}, step {i+1}/{total_steps}, batch loss {loss.item():.4f}')
    loss_list.append(loss.item())
    weight = model.conv1.weight.data.numpy()

epochs = list(range(1,config['max_epochs']+1))
plt.plot(epochs,loss_list)
plt.show()
writer.flush()
print("Finished training")

