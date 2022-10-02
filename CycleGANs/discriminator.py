import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size = 4, padding = 1, stride = 1, **kwargs):
    super(Block, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding = padding, stride = stride, bias = False
    , padding_mode = "reflect")
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu =  nn.LeakyReLU(0.2)

  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

class Discriminator(nn.Module):
  def __init__(self, in_channels, features=[64, 128, 256, 512]):
    super(Discriminator, self).__init__()
    self.initial = nn.Sequential(
      nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
    )

    layers = []
    in_channels = features[0]
    for feature in features[1:]:
      layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
      in_channels = feature
    layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    x = self.initial(x)
    return torch.sigmoid(self.model(x))
  
def test():
  x = torch.randn((5, 3, 256, 256))
  model = Discriminator(in_channels=3)
  preds = model(x)
  print(preds.shape)


if __name__ == "__main__":
    test()
