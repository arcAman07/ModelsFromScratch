import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299

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

class DepthwiseSeperableConvolution(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding=0, bias=False, dilation = 1):
      super(DepthwiseSeperableConvolution, self).__init__()
      self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
      self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
      self.bn = nn.BatchNorm2d(out_channels)
        
  def forward(self, x):
      x = self.depthwise_conv(x)
      x = self.pointwise_conv(x)
      x = self.bn(x)
      return x

class BasicBlockEntry(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 1, bias = False ):
      super(BasicBlockEntry, self).__init__()
      self.basic_sequential = nn.Sequential(
        nn.ReLU(),
        DepthwiseSeperableConvolution(in_channels, out_channels, kernel_size, stride, padding, bias),
        nn.ReLU(),
        DepthwiseSeperableConvolution(out_channels, out_channels, kernel_size, stride, padding, bias),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      )
  
  def forward(self, x):
      return self.basic_sequential(x)


class BasicBlockMiddle(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, bias = False):
    super(BasicBlockMiddle, self).__init__()
    self.basic_sequential = nn.Sequential(
      nn.ReLU(),
      DepthwiseSeperableConvolution(in_channels, out_channels, kernel_size, stride, padding, bias),
      nn.ReLU(),
      DepthwiseSeperableConvolution(out_channels, out_channels, kernel_size, stride, padding, bias),
      nn.ReLU(),
      DepthwiseSeperableConvolution(out_channels, out_channels, kernel_size, stride, padding, bias),
    )

  def forward(self, x):
    return self.basic_sequential(x)

class BasicBlockExit(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, bias = False):
    super(BasicBlockExit, self).__init__()
    self.basic_sequential = nn.Sequential(
      nn.ReLU(),
      DepthwiseSeperableConvolution(in_channels, out_channels, kernel_size, stride, padding, bias),
    )

  def forward(self, x):
    return self.basic_sequential(x)

class PointwiseSeperableConvolution(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride= 1, padding = 0 , bias = False):
    super(PointwiseSeperableConvolution, self).__init__()
    self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, bias=bias)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.pointwise_conv(x)
    x = self.bn(x)
    return x

class Xception(ImageClassificationBase):
  """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf

  """
  def __init__(self, num_classes = 1000, input_channels = 3, width_mult = 1, bias = False ):
    super(Xception, self).__init__()
    self.num_classes = num_classes
    self.input_channels = input_channels
    self.width_mult = width_mult
    self.bias = bias
    self.relu = nn.ReLU(inplace=True)
    self.first_block = nn.Sequential(
        nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=0, bias=self.bias),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=self.bias),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    self.first_pointwise_block = PointwiseSeperableConvolution(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 2, padding = 0, bias = False)
    self.second_block = nn.Sequential(
        DepthwiseSeperableConvolution(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding=1, bias=False, dilation = 1 ),
        nn.ReLU(),
        DepthwiseSeperableConvolution(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding=1, bias=False, dilation = 1 ),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    self.third_block = BasicBlockEntry(128, 256, kernel_size = 3, stride=1, padding=1, bias= False)
    self.second_pointwise_block = PointwiseSeperableConvolution(in_channels = 128, out_channels = 256, kernel_size = 1, stride= 2, padding = 0 , bias = False)
    self.fourth_block = BasicBlockEntry(256, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.third_pointwise_block = PointwiseSeperableConvolution(in_channels = 256, out_channels = 728, kernel_size = 1, stride= 2, padding=0, bias = False)
    self.fifth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.sixth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.seventh_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.eighth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.ninth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.tenth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.eleventh_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.twelfth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.thirteenth_block = BasicBlockMiddle(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.fourteenth_block = BasicBlockExit(728, 728, kernel_size = 3, stride=1, padding=1, bias= False)
    self.fifteenth_block = BasicBlockExit(728, 1024, kernel_size = 3, stride=1, padding=1)
    self.fifteenth_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.sixteenth_block = BasicBlockExit(1024, 1536, kernel_size = 3, stride=1, padding=1, bias= False)
    self.seventeenth_block = BasicBlockExit(1536, 2048, kernel_size = 3, stride=1, padding=1, bias= False)
    self.fourteenth_pointwise_block = PointwiseSeperableConvolution(in_channels = 728, out_channels = 1024, kernel_size = 1, stride=2, padding=0, bias = False)
    self.fc = nn.Linear(2048, num_classes)

  def forward(self, x):
      x = self.first_block(x)
      y = x.clone()
      x = self.first_pointwise_block(x)
      y = self.second_block(y)
      x = x + y
      y = x.clone()
      x = self.third_block(x)
      y = self.second_pointwise_block(y)
      x = x + y
      y = x.clone()
      x = self.fourth_block(x)
      y = self.third_pointwise_block(y)
      x = x + y
      y = self.fifth_block(x)
      x = x + y
      y = self.sixth_block(x)
      x = x + y
      y = self.seventh_block(x)
      x = x + y
      y = self.eighth_block(x)
      x = x + y
      y = self.ninth_block(x)
      x = x + y
      y = self.tenth_block(x)
      x = x + y
      y = self.eleventh_block(x)
      x = x + y
      y = self.twelfth_block(x)
      x = x + y
      y = self.thirteenth_block(x)
      x = x + y
      y = x.clone()
      x = self.fourteenth_block(x)
      x = self.fifteenth_block(x)
      x = self.fifteenth_pooling(x)
      y = self.fourteenth_pointwise_block(y)
      x = x + y
      x = self.sixteenth_block(x)
      x = self.seventeenth_block(x)
      x = self.relu(x)
      x = F.adaptive_avg_pool2d(x, (1, 1))
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      return x