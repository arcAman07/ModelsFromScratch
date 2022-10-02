import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
import config
from tqdm import tqdm
from torchvision.utils import save_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

train_data = datasets.STL10(root='./data', split='train', download=True, transform=transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
))
train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
      shuffle=True,
      num_workers=config.NUM_WORKERS,
      pin_memory=True)

test_data = datasets.STL10(root='./data', train='test', download=True, transform=transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
))
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE,
      shuffle=True,
      num_workers=config.NUM_WORKERS,
      pin_memory=True)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# img, label = test_data[0]
# plt.imshow(img[0])
# print('Label: ',label)

batch_size = config.BATCH_SIZE
num_epochs = config.NUM_EPOCHS
latent_size = 100
def create_image():
  y = torch.randn(batch_size,latent_size)
  return y

def denorm(x):
  out = (x + 1) / 2
  return out.clamp(0, 1)

def train_func(disc, gen, opt_disc, opt_gen, loader, l1, mse, d_scaler, g_scaler, num_epochs):
  loop = tqdm(loader, leave=True)
  for idx, (image, label) in enumerate(loop):
    image = image.to(config.DEVICE)
    label = label.to(config.DEVICE)
    # Train Discriminator for the particular label
    with torch.cuda.amp.autocast():
      fake = gen(create_image().to(config.DEVICE), label)
      disc_fake = disc(fake, label)
      disc_real = disc(image, label)
      disc_loss = (mse(disc_real, torch.ones_like(disc_real)) + mse(disc_fake, torch.zeros_like(disc_fake))) / 2
    opt_disc.zero_grad()
    d_scaler.scale(disc_loss).backward()
    d_scaler.step(opt_disc)
    d_scaler.update()

    # Train Generator
    with torch.cuda.amp.autocast():
      fake = gen(create_image().to(config.DEVICE), label)
      disc_fake = disc(fake, label)
      gen_loss = mse(disc_fake, torch.ones_like(disc_fake))
      # l1_loss = l1(fake, image) * config.LAMBDA_CYCLE
      total_loss = gen_loss
    opt_gen.zero_grad()
    g_scaler.scale(total_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()

    loop.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

    if idx % 200 == 0:
      save_image(denorm(fake.detach()), f"images/{idx}.png")

    loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))

def test_func(gen, loader):
  loop = tqdm(loader, leave=True)
  for idx, (image, label) in enumerate(loop):
    image = image.to(config.DEVICE)
    label = label.to(config.DEVICE)
    fake = gen(create_image().to(config.DEVICE), label)
    save_image(denorm(fake.detach()), f"test_images/{idx}.png")

def main():
  disc = Discriminator(n_classes = 196).to(config.DEVICE)
  gen = Generator(n_classes = 196).to(config.DEVICE)
  opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
  opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
  l1 = nn.L1Loss()
  mse = nn.MSELoss()
  d_scaler = torch.cuda.amp.GradScaler()
  g_scaler = torch.cuda.amp.GradScaler()
  for epoch in range(config.NUM_EPOCHS):
    train_func(disc, gen, opt_disc, opt_gen, train_loader, l1, mse, d_scaler, g_scaler, config.NUM_EPOCHS)
  test_func(gen, test_loader)
  PATH = "./model.pt"
  torch.save({
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            }, PATH)

if __name__ == "__main__":
  main()