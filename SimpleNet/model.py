import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ImageClassificationBase

class SimpleNet(ImageClassificationBase):
  def __init__(self, in_channels, out_channels)