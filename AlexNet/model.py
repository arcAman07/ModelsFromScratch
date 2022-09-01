import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ImageClassificationBase

class AlexNet(ImageClassificationBase):
  def __init__(self, input_channels = 3, num_classes = 1000):
    super(AlexNet, self).__init__()
    self.block_1 = nn.Sequential(
      nn.Conv2d(input_channels, 96, kernel_size = 11, stride = 4, padding = 0, bias = False),
      nn.BatchNorm2d(96),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 3, stride = 2),
      nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2, bias = False),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 3, stride = 2),
      nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1, bias = False),
      nn.BatchNorm2d(384),
      nn.ReLU(),
      nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1, bias = False),
      nn.BatchNorm2d(384),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 3, stride = 2),
      nn.Dropout(dropout = 0.5),
    )
    self.fc_1 = nn.Linear(256 * 6 * 6, 4096)
    self.relu = nn.ReLU()
    self.dropout_1 = nn.Dropout(dropout = 0.5)
    self.fc_2 = nn.Linear(4096, 4096)
    self.fc_3 = nn.Linear(4096, num_classes)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, x):
    x = self.block_1(x)
    x = self.fc_1(x)
    x = self.relu(x)
    x = self.dropout_1(x)
    x = self.fc_2(x)
    x = self.relu(x)
    x = self.fc_3(x)
    x = self.softmax(x)
    return x


