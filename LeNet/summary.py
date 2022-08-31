from model import LeNet

model = LeNet()
from torchvision import models
from torchsummary import summary

print(summary(model.cuda(),(1, 32, 32)))

"""

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             150
       BatchNorm2d-2            [-1, 6, 28, 28]              12
              Tanh-3            [-1, 6, 28, 28]               0
         AvgPool2d-4            [-1, 6, 14, 14]               0
            Conv2d-5           [-1, 16, 10, 10]           2,400
       BatchNorm2d-6           [-1, 16, 10, 10]              32
              Tanh-7           [-1, 16, 10, 10]               0
         AvgPool2d-8             [-1, 16, 5, 5]               0
            Conv2d-9            [-1, 120, 1, 1]          48,000
      BatchNorm2d-10            [-1, 120, 1, 1]             240
             Tanh-11            [-1, 120, 1, 1]               0
           Linear-12                   [-1, 84]          10,164
           Linear-13                   [-1, 10]             850
          Softmax-14                   [-1, 10]               0
================================================================
Total params: 61,848
Trainable params: 61,848
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.16
Params size (MB): 0.24
Estimated Total Size (MB): 0.40
----------------------------------------------------------------
None


"""