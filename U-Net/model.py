import torch
import torch.nn as nn

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.downsample_block_1 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size= 3, )
    )