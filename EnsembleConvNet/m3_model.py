import torch
import torch.nn as nn
import torch.nn.functional as F

class M3_Model(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 32):
    super(M3_Model, self).__init__()
    self.network = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, 48, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(48),
      nn.ReLU(),
      nn.Conv2d(48, 64, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 80, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(80),
      nn.ReLU(),
      nn.Conv2d(80, 96, kernel_size = 3, bias=False, stride = 1, padding = 0),
      nn.BatchNorm2d(96),
      nn.ReLU(),
      nn.Conv2d(96, 112, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(112),
      nn.ReLU(),
      nn.Conv2d(112, 128, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 144, kernel_size = 3,bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(144),
      nn.ReLU(),
      nn.Conv2d(144, 160, kernel_size = 3, bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(160),
      nn.ReLU(),
      nn.Conv2d(160, 176, kernel_size = 3,bias = False, stride = 1, padding = 0),
      nn.BatchNorm2d(176),
      nn.ReLU(),
    )
    self.fc = nn.Linear(11264, 10)
    self.bn = nn.BatchNorm1d(10)
  
  def forward(self, x):
    x = self.network(x)
    x = x.view(x.size(0), -1)
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

