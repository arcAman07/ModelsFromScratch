import torch
import torch.nn as nn

# self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels = 2):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.downsample_block_1 = self._make_downsample_block(in_channels, 64)
    self.downsample_block_2 = self._make_downsample_block(64, 128)
    self.downsample_block_3 = self._make_downsample_block(128, 256)
    self.downsample_block_4 = self._make_downsample_block(256, 512)
    self.downsample_max_pool_1 = self.downsample_max_pool2d(dropout = 0.25)
    self.downsample_max_pool_2 = self.downsample_max_pool2d(dropout = 0.5)
    self.downsample_max_pool_3 = self.downsample_max_pool2d(dropout = 0.5)
    self.downsample_max_pool_4 = self.downsample_max_pool2d(dropout = 0.5)
    self.middle_block_5 = self._make_middle_block(512, 1024)
    self.skip_upsample_1 = self.upsample_block(scale_factor = 56/64)
    self.skip_upsample_2 = self.upsample_block(scale_factor = 104/136)
    self.skip_upsample_3 = self.upsample_block(scale_factor = 200/280)
    self.skip_upsample_3 = self.upsample_block(scale_factor = 392/568)
    self.conv1d_upsample_block_2 = self._make_conv1d(512, 256)
    self.conv1d_upsample_block_3 = self._make_conv1d(256, 128)
    self.conv1d_upsample_block_4 = self._make_conv1d(128, 64)
    self.upsample_block_1 = nn.Sequential(
      self.upsample_block(scale_factor = 2),
      self._make_conv1d(1024, 512),
    )
    self.upsample_block_2 = nn.Sequential(
      self.upsample_block(scale_factor = 2),
      self._make_conv1d(512, 256),
    )
    self.upsample_block_3 = nn.Sequential(
      self.upsample_block(scale_factor = 2),
      self._make_conv1d(256, 128),
    )
    self.upsample_block_4 = nn.Sequential(
      self.upsample_block(scale_factor = 2),
      self._make_conv1d(128, 64),
    )
    self.upsample_layer_1 = self._make_upsample_block(1024, 512, 0.5)
    self.upsample_layer_2 = self._make_upsample_block(512, 256, 0.5)
    self.upsample_layer_3 = self._make_upsample_block(256, 128, 0.5)
    self.upsample_layer_4 = self._make_upsample_block(128, 64, 0.25)
    self.final_conv1d = self._make_conv1d(64, out_channels)

  def _make_downsample_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride = 1, padding = 0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride = 1, padding = 0, bias= False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )

  def downsample_max_pool2d(self, dropout):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size= 2, stride= 2),
    )

  def _make_middle_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride=1, padding=0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )

  def _make_conv1d(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )


  def upsample_block(self, scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

  def _make_upsample_block(self, in_channels, out_channels, dropout):
    return nn.Sequential(
      nn.Dropout(dropout),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
    )

  def forward(self, x):
    downsample_block_1 = self.downsample_block_1(x)
    downsample_max_pool_1 = self.downsample_max_pool2d(downsample_block_1)
    downsample_block_2 = self.downsample_block_2(downsample_block_1)
    downsample_max_pool_2 = self.downsample_max_pool2d(downsample_block_2)
    downsample_block_3 = self.downsample_block_3(downsample_block_2)
    downsample_max_pool_3 = self.downsample_max_pool2d(downsample_block_3)
    downsample_block_4 = self.downsample_block_4(downsample_block_3)
    downsample_max_pool_4 = self.downsample_max_pool2d(downsample_block_4)
    middle_block_5 = self.middle_block_5(downsample_block_4)
    upsample_block_1 = self.upsample_block_1(middle_block_5)
    upsample_layer_1 = self.upsample_layer_1(torch.cat((self.skip_upsample_1(downsample_block_4), upsample_block_1), dim = 1))
    upsample_block_2 = self.upsample_block_2(upsample_layer_1)
    upsample_layer_2 = self.upsample_layer_2(torch.cat((self.skip_upsample_2(downsample_block_3), upsample_block_2), dim = 1))
    upsample_block_3 = self.upsample_block_3(upsample_layer_2)
    upsample_layer_3 = self.upsample_layer_3(torch.cat((self.skip_upsample_3(downsample_block_2), upsample_block_3), dim = 1))
    upsample_block_4 = self.upsample_block_4(upsample_layer_3)
    upsample_layer_4 = self.upsample_layer_4(torch.cat((self.skip_upsample_4(downsample_block_1), upsample_block_4), dim = 1))
    final_output = self.final_conv1d(upsample_layer_4)
    return final_output


model = UNet(in_channels = 1, out_channels = 2)
from torchvision import models
from torchsummary import summary

print(summary(model,(1, 572, 572)))
