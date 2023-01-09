import torchvision
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
from m3_model import M3_Model
from utils import ImageClassificationBase
from m5_model import M5_Model
from m7_model import M7_Model

batch_size=120

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform= ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform= ToTensor(),
)

train_dl = DataLoader(train_data, batch_size, shuffle=True)
val_dl = DataLoader(test_data, batch_size*2)

optim = optim.Adam()
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.001

model_1 = M3_Model()
model_2 = M5_Model()
model_3 = M7_Model()

history_1 = fit(num_epochs, lr, model_2, train_dl, val_dl, opt_func)

