import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ImageClassificationBase

class LeNet(ImageClassificationBase):
  def __init__(self, input_channels = 1):
    super(LeNet, self).__init__()
    self.block_1 = nn.Sequential(
      nn.Conv2d(input_channels, 6, kernel_size = 5, stride = 1, padding = 0, bias = False),
      nn.BatchNorm2d(6),
      nn.Tanh(),
      nn.AvgPool2d(kernel_size = 2, stride = 2),
      nn.Conv2d(6, 16, kernel_size = 5, stride = 1, padding = 0, bias = False),
      nn.BatchNorm2d(16),
      nn.Tanh(),
      nn.AvgPool2d(kernel_size = 2, stride = 2),
      nn.Conv2d(16, 120, kernel_size = 5, stride = 1, padding = 0, bias = False),
      nn.BatchNorm2d(120),
      nn.Tanh(),
    )
    self.linear_1 = nn.Linear(120, 84)
    self.linear_2 = nn.Linear(84, 10)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, x):
    x = self.block_1(x)
    x = x.view(x.size(0), -1)
    x = self.linear_1(x)
    x = self.linear_2(x)
    x = self.softmax(x)
    return x