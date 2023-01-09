import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ImageClassificationBase

class M7_Model(ImageClassificationBase):
  def __init__(self, in_channels = 1, out_channels = 48, kernel_size = 7, padding = 0, bias = False, stride = 1):
    super(M7_Model, self).__init__()
    self.network = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, 96, kernel_size,stride, padding, bias = False),
      nn.BatchNorm2d(96),
      nn.ReLU(),
      nn.Conv2d(96, 144, kernel_size,stride, padding, bias = False),
      nn.BatchNorm2d(144),
      nn.ReLU(),
      nn.Conv2d(144, 192, kernel_size, stride, padding, bias = False),
      nn.BatchNorm2d(192),
      nn.ReLU(),
    )

    self.fc = nn.Linear(192*4*4, 10, bias = False)
    self.bn = nn.BatchNorm1d(10)

  def forward(self, x):
    x = self.network(x)
    x = x.view(x.size(0), -1)
    x = self.bn(self.fc(x))
    return x

def test():
  model = M7_Model()
  inputs = torch.randn((3,1,28,28))
  output = model(inputs)
  print(output.shape)
  print(output)

if __name__ == '__main__':
  test()
