import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride , padding, use_act = True, downsample = True, **kwargs):
    super(Block, self).__init__()
    self.use_act = use_act
    self.downsample = downsample
    if downsample:
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False, **kwargs)
    else:
      self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, momentum=0.1,  eps=0.8)
    if use_act:
      self.act = nn.ReLU()
    else:
      self.act = nn.LeakyReLU(0.2)

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

class Generator(nn.Module):
  def __init__(self, noise_dim = 100, n_classes = 10, embedding_dim = 100):
    super(Generator, self).__init__()
    self.initial_layer = nn.Sequential(
      nn.Embedding(n_classes, embedding_dim),
      nn.Linear(embedding_dim, 16),
    )

    self.latent = nn.Sequential(
      nn.Linear(noise_dim, 4*4*512),
      nn.LeakyReLU(0.2, inplace = True),
    )

    self.model = nn.Sequential(
      *[
        Block(513, 64*8, 4, 2, 1, use_act = True, downsample = False),
        Block(64*8, 64*4, 4, 2, 1, use_act = True, downsample = False),
        Block(64*4, 64*2, 4, 2, 1, use_act = True, downsample = False),
        Block(64*2, 64, 4, 2, 1, use_act = True, downsample = False),
        nn.ConvTranspose2d(64*1, 3, 4, 2, 1, bias=False),
        nn.Tanh(),
      ]
    )

  def forward(self, x, label):
    x_1 = self.initial_layer(label)
    x_1 = x_1.view(x_1.shape[0], 1, 4, 4)
    x_2 = self.latent(x)
    x_2 = x_2.view(x_2.shape[0], 512, 4, 4)
    x = torch.cat([x_1, x_2], dim = 1)
    return self.model(x)
  
def test():
    img_size = 128
    x = torch.randn((1,1,100))
    label = torch.randint(0, 10, (1,))
    print(label)
    gen = Generator()
    print(gen(x, label).shape)

if __name__ == "__main__":
    test()