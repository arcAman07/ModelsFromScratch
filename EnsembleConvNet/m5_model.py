import torch
import torch.nn as nn
from utils import ImageClassificationBase
import torch.nn.functional as F

class M5_Model(ImageClassificationBase):
  def __init__(self, in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding  = 0, bias = False):
    super(M5_Model, self).__init__()
    self.network = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, bias, stride, padding),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size, bias, stride, padding),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 96, kernel_size, bias, stride, padding),
      nn.BatchNorm2d(96),
      nn.ReLU(),
      nn.Conv2d(96, 128, kernel_size, bias, stride, padding),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 160, kernel_size, bias, stride, padding),
      nn.BatchNorm2d(160),
      nn.ReLU(),
    )

    self.fc = nn.Linear(10240, 10),
    self.bn = nn.BatchNorm1d(10),

  def forward(self, x):
    x = self.network(x)
    x = x.view(x.size(0, -1))
    x = self.bn(self.fc(x))
    return x
  
def test():
  model = M3_Model()
  inputs = torch.randn((3,1,28,28))
  output = model(inputs)
  print(output.shape)
  print(output)

if __name__ == '__main__':
  test()
