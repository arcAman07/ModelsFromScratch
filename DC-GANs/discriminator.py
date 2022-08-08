import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from base import ImageClassificationBase

class Discriminator(nn.Module):
  def __init__(self, channels_img, features_d):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
      nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      self._block(features_d, features_d*2, 4, 2, 1, bias=False),
      self._block(features_d*2, features_d*4, 4, 2, 1, bias=False),
      self._block(features_d*4, features_d*8, 4, 2, 1, bias=False),
      nn.Conv2d(features_d*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
      nn.Sigmoid(),
    )

  
  def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2, inplace=True),
    )

  def forward(self, x):
    return self.disc(x)
