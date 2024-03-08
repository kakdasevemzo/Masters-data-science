from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
import numpy as np
import torch


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train',remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            remove_indices = [i for i, target in enumerate(self.targets) if target == remove_class]
            self.images = torch.tensor(np.delete(self.images, remove_indices, axis=0))
            self.targets = torch.tensor(np.delete(self.targets, remove_indices, axis=0))
            self.classes.pop(remove_class)

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        target_id = self.targets[index].item()
        negative_classes = [c for c in self.classes if c != target_id]
        negative_class = choice(negative_classes)
        negative_indices = self.target2indices[negative_class]
        negative_index = choice(negative_indices)
        return negative_index


    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        positive_indices = self.target2indices[target_id]
        positive_index = choice(positive_indices)
        return positive_index

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)