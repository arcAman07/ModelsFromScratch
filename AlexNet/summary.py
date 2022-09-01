from model import AlexNet

model = AlexNet()
from torchvision import models
from torchsummary import summary

print(summary(model,(3, 227 , 227)))

"""

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 96, 55, 55]          34,848
       BatchNorm2d-2           [-1, 96, 55, 55]             192
              ReLU-3           [-1, 96, 55, 55]               0
         MaxPool2d-4           [-1, 96, 27, 27]               0
            Conv2d-5          [-1, 256, 27, 27]         614,400
       BatchNorm2d-6          [-1, 256, 27, 27]             512
              ReLU-7          [-1, 256, 27, 27]               0
         MaxPool2d-8          [-1, 256, 13, 13]               0
            Conv2d-9          [-1, 384, 13, 13]         884,736
      BatchNorm2d-10          [-1, 384, 13, 13]             768
             ReLU-11          [-1, 384, 13, 13]               0
           Conv2d-12          [-1, 384, 13, 13]       1,327,104
      BatchNorm2d-13          [-1, 384, 13, 13]             768
             ReLU-14          [-1, 384, 13, 13]               0
           Conv2d-15          [-1, 256, 13, 13]         884,736
      BatchNorm2d-16          [-1, 256, 13, 13]             512
             ReLU-17          [-1, 256, 13, 13]               0
        MaxPool2d-18            [-1, 256, 6, 6]               0
          Dropout-19                 [-1, 9216]               0
           Linear-20                 [-1, 4096]      37,752,832
             ReLU-21                 [-1, 4096]               0
          Dropout-22                 [-1, 4096]               0
           Linear-23                 [-1, 4096]      16,781,312
             ReLU-24                 [-1, 4096]               0
           Linear-25                 [-1, 1000]       4,097,000
          Softmax-26                 [-1, 1000]               0
================================================================
Total params: 62,379,720
Trainable params: 62,379,720
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 16.06
Params size (MB): 237.96
Estimated Total Size (MB): 254.60
----------------------------------------------------------------
None


"""