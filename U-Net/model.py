import torch
import torch.nn as nn

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.downsample_block_1 = _make_downsample_block(in_channels, 64, 0.25)
    self.downsample_block_2 = _make_downsample_block(64, 128, 0.5)
    self.downsample_block_3 = _make_downsample_block(128, 256, 0.5)
    self.downsample_block_4 = _make_downsample_block(256, 512, 0.5)
    self.middle_block_5 = _make_middle_block(512, 1024)
    self.upsample_layer_1 = upsample_block()
    self.upsample_block_1 = _make_upsample_block

  def _make_downsample_block(in_channels, out_channels, dropout):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride = 1, padding = 0 ),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride = 1, padding = 0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size= 2, stride= 2),
      nn.Dropout(dropout),
    )

  def _make_middle_block(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(out_channels, in_channels, kernel_size= 3, stride=1, padding=0),
      nn.ReLU(),
    )

  def upsample_block():
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

  def _make_upsample_block(in_channels, out_channels, dropout):
    return nn.Sequential(
      nn.Dropout(dropout),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
      nn.ReLU(),
    )
