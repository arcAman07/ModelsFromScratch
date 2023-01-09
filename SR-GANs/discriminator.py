import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = False):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride, padding, bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 2, features = [64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size = 3),
            nn.LeakyReLU(0.2),
        )