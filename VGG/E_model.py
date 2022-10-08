import torch
import numpy as np
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.activation(self.conv_layer(x))

class E_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(E_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x

def train():
    x = torch.rand((3,3,224,224))
    model = E_VGG()
    y = model(x)
    print(y)
    print(y.shape)

if __name__ == "__main__":
    train()