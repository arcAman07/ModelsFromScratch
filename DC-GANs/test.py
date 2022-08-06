from discriminator import Discriminator
from generator import Generator
from initialize import initialize_weights

def test():
  N, in_channels, H, W = 8, 3, 64, 64
  z_dim = 100
  x = torch.randn((N, in_channels, H, W))
  disc = Discriminator(in_channels, 8)
  initialize_weights(disc)
  assert disc(x).shape == (N, 1, 1 ,1)
  gen = Generator(z_dim, in_channels, 8)
  z = torch.randn((N, z_dim, 1 ,1))
  assert gen(z).shape == (N, in_channels, H, W)
  print("Success")

  