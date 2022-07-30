import torch
import numpy as np
import torch.nn as nn

"""
Model architecture: Inverted Residual Block (IRB)

Input( b * h * c)    Operator    t( expansion factor / Width multiplier)    c( channels )    s( stride )    n( Repeating layers )

224 ^2 * 3         Conv2d               None                                 32                2                 1

112 ^2 * 32        Bottleneck           1                                    16                1                 1

112 ^2 * 16        Bottleneck           6                                    24                2                 2

56 ^2 * 24         Bottleneck           6                                    32                2                 3

28 ^2 * 32         Bottleneck           6                                    64                2                 4

14 ^2 * 64         Bottleneck           6                                    96                1                 3

14 ^2 * 96         Bottleneck           6                                    160               2                 3

7 ^2 * 160         Bottleneck           6                                    320               1                 1

7 ^2 * 320         Conv2d 1X1           None                                 1280              1                 1

7 ^2 * 1280        avgPool 7*7          None                                  -                -                 -

1 ^2 * 1280        Conv2d 1X1           None                                  k                -                 -

"""

"""
Model : Linear Bottleneck

Input                        Operator                   Output

h * w * k       Conv2d 1*1 , BatchNorm2d, Relu6        h * w * tk

h * w * tk      Depthwise Conv2d 3*3 stride = s , BatchNorm1d, Relu6        h/s * w/s * tk

h/s * w/s * tk  Pointwise Conv2d 1*1 , BatchNorm2d, Relu6                   h/s * w/s * d'

"""

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class LinearBottleneck(nn.Module):
  def __init__(self, in_channels, out_channels, stride, expansion, groups, bias=False):
    super(LinearBottleneck, self).__init__()
    self.stride = stride
    self.expansion = expansion
    self.groups = groups
    self.sequential_conv = nn.Sequential(
      nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=bias, groups=1),
      nn.BatchNorm2d(in_channels * expansion),
      nn.ReLU6(inplace=True),
    )
    self.depthwise_conv = nn.Sequential(
      nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups),
      nn.BatchNorm2d(in_channels * expansion),
      nn.ReLU6(inplace=True),
    )
    self.pointwise_conv = nn.Sequential(
      nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=bias, groups=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU6(inplace=True),
    )

  def forward(self, x):
    if self.stride == 1:
      x = self.sequential_conv(x)
    else:
      x = self.sequential_conv(x)
      x = self.depthwise_conv(x)
    x = self.pointwise_conv(x)
    return x


class MobileNetV2(ImageClassificationBase):
  def __init__(self, num_classes=10, width_mult=1.0, groups=8, bias=False):
    super(MobileNetV2, self).__init__()
    self.width_mult = width_mult
    self.groups = groups
    self.bias = bias
    self.first_layer = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=bias),
      nn.BatchNorm2d(32),
      nn.ReLU6(inplace=True),
    )
    self.first_bottleneck = nn.Sequential(
      LinearBottleneck(32, 16, 1, 1, groups=32, bias=bias),
    )
    self.second_bottleneck = nn.Sequential(
      LinearBottleneck(16, 24, 2, 6, groups=16, bias=bias),
      LinearBottleneck(24, 24, 1, 6, groups=24, bias=bias),
    )
    self.third_bottleneck = nn.Sequential(
      LinearBottleneck(24, 32, 2, 6, groups=24, bias=bias),
      LinearBottleneck(32, 32, 1, 6, groups=32, bias=bias),
      LinearBottleneck(32, 32, 1, 6, groups=32, bias=bias),
    )
    self.fourth_bottleneck = nn.Sequential(
      LinearBottleneck(32, 64, 2, 6, groups=32, bias=bias),
      LinearBottleneck(64, 64, 1, 6, groups=64, bias=bias),
      LinearBottleneck(64, 64, 1, 6, groups=64, bias=bias),
      LinearBottleneck(64, 64, 1, 6, groups=64, bias=bias),
    )
    self.fifth_bottleneck = nn.Sequential(
      LinearBottleneck(64, 96, 1, 6, groups=64, bias=bias),
      LinearBottleneck(96, 96, 1, 6, groups=96, bias=bias),
      LinearBottleneck(96, 96, 1, 6, groups=96, bias=bias),
    )
    self.sixth_bottleneck = nn.Sequential(
      LinearBottleneck(96, 160, 2, 6, groups=96, bias=bias),
      LinearBottleneck(160, 160, 1, 6, groups=160, bias=bias),
      LinearBottleneck(160, 160, 1, 6, groups=160, bias=bias),
    )
    self.seventh_bottleneck = nn.Sequential(
      LinearBottleneck(160, 320, 1, 6, groups=160, bias=bias),
    )
    self.pointwise_conv = nn.Sequential(
      nn.Conv2d(320, 1280, kernel_size=1, bias=bias),
      nn.BatchNorm2d(1280),
      nn.ReLU6(inplace=True),
    )
    self.avgpool = nn.AvgPool2d(kernel_size=7)
    self.classifier = nn.Linear(1280, num_classes)

  def forward(self, x):
    self.x = x
    x = self.first_layer(x)
    x = self.first_bottleneck(x)
    x = self.second_bottleneck(x)
    x = self.third_bottleneck(x)
    x = self.fourth_bottleneck(x)
    x = self.fifth_bottleneck(x)
    x = self.sixth_bottleneck(x)
    x = self.seventh_bottleneck(x)
    x = self.pointwise_conv(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
