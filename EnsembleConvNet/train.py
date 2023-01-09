import torchvision
from model import LeNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data.dataloader import DataLoader

batch_size=120

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
)

train_dl = DataLoader(train_data, batch_size, shuffle=True)
val_dl = DataLoader(test_data, batch_size*2)

optim = optim.Adam()
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

