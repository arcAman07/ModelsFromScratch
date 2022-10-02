import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, down = True, act = True, **kwargs):
        super(Block, self).__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode = "reflect" ,bias = False, **kwargs)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        if act:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.block = nn.Sequential(
      Block(channels, channels, kernel_size = 3, stride = 1, padding = 1, down = True, act = True),
      Block(channels, channels, kernel_size = 3, stride = 1, padding = 1, down = True, act = False),
    )

  def forward(self, x):
    return self.block(x) + x


class Generator(nn.Module):
  def __init__(self, input_channels = 3, num_features = 64, residual_blocks = 9):
    super(Generator, self).__init__()
    self.initial = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
    self.downsample_blocks = nn.ModuleList(
      [
                Block(num_features, num_features*2, kernel_size=3, stride=2, padding=1, act = False),
                Block(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1, act = False),
            ]
    )

    self.residual_blocks = nn.Sequential(
      *[ResidualBlock(num_features*4) for _ in range(residual_blocks)]
    )

    self.upsample_blocks = nn.ModuleList(
      [
        Block(num_features*4, num_features*2, kernel_size=3, stride= 2, padding=1, output_padding = 1,  down = False, act = False),
        Block(num_features*2, num_features*1, kernel_size=3, stride= 2, padding=1, output_padding = 1,  down = False, act = False),
      ]
    )

    self.last = nn.Conv2d(num_features*1, input_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

  def forward(self, x):
    x = self.initial(x)
    for downsample_block in self.downsample_blocks:
      x = downsample_block(x)
    x = self.residual_blocks(x)
    for upsample_block in self.upsample_blocks:
      x = upsample_block(x)
    return torch.tanh(self.last(x))

def test():
    img_size = 256
    x = torch.randn((2, 3, img_size, img_size))
    gen = Generator(input_channels = 3, num_features = 9)
    print(gen(x).shape)

if __name__ == "__main__":
    test()