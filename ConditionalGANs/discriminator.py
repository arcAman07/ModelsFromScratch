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

class Discriminator(nn.Module):
  def __init__(self, noise_dim = 100, n_classes = 10, embedding_dim = 100):
    super(Discriminator, self).__init__()
    self.label_embedding = nn.Embedding(n_classes, embedding_dim)
    self.linear_layer = nn.Linear(embedding_dim, 3*128*128)

    self.model = nn.Sequential(
      *[
        Block(6, 64, 4, 2, 1, use_act = False, downsample = True),
        Block(64, 64*2, 4, 3, 2, use_act = True, downsample = True),
        Block(64*2, 64*4, 4, 3, 2, use_act = True, downsample = True),
        Block(64*4, 64*8, 4, 3, 2, use_act = True, downsample = True),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(4608, 1),
        nn.Sigmoid(),
      ]
    )
  
  def forward(self, x, label):
    x_1 = self.label_embedding(label)
    x_1 = self.linear_layer(x_1)
    x_1 = x_1.view(x_1.shape[0], 3, 128, 128)
    x = torch.cat([x, x_1], dim = 1)
    return self.model(x)
  
def test():
    img_size = 128
    x = torch.randn((1,3,img_size, img_size))
    label = torch.randint(0, 10, (1,))
    print(label)
    disc = Discriminator()
    print(disc(x, label).shape)

if __name__ == "__main__":
    test()
