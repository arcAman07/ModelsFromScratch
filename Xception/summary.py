from model import Xception
model = Xception()
from torchvision import models
from torchsummary import summary

"""

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 149, 149]             864
       BatchNorm2d-2         [-1, 32, 149, 149]              64
              ReLU-3         [-1, 32, 149, 149]               0
            Conv2d-4         [-1, 64, 147, 147]          18,432
       BatchNorm2d-5         [-1, 64, 147, 147]             128
              ReLU-6         [-1, 64, 147, 147]               0
            Conv2d-7          [-1, 128, 74, 74]           8,192
       BatchNorm2d-8          [-1, 128, 74, 74]             256
PointwiseSeperableConvolution-9          [-1, 128, 74, 74]               0
           Conv2d-10         [-1, 64, 147, 147]             576
           Conv2d-11        [-1, 128, 147, 147]           8,192
      BatchNorm2d-12        [-1, 128, 147, 147]             256
DepthwiseSeperableConvolution-13        [-1, 128, 147, 147]               0
             ReLU-14        [-1, 128, 147, 147]               0
           Conv2d-15        [-1, 128, 147, 147]           1,152
           Conv2d-16        [-1, 128, 147, 147]          16,384
      BatchNorm2d-17        [-1, 128, 147, 147]             256
DepthwiseSeperableConvolution-18        [-1, 128, 147, 147]               0
        MaxPool2d-19          [-1, 128, 74, 74]               0
             ReLU-20          [-1, 128, 74, 74]               0
           Conv2d-21          [-1, 128, 74, 74]           1,152
           Conv2d-22          [-1, 256, 74, 74]          32,768
      BatchNorm2d-23          [-1, 256, 74, 74]             512
DepthwiseSeperableConvolution-24          [-1, 256, 74, 74]               0
             ReLU-25          [-1, 256, 74, 74]               0
           Conv2d-26          [-1, 256, 74, 74]           2,304
           Conv2d-27          [-1, 256, 74, 74]          65,536
      BatchNorm2d-28          [-1, 256, 74, 74]             512
DepthwiseSeperableConvolution-29          [-1, 256, 74, 74]               0
             ReLU-30          [-1, 256, 74, 74]               0
        MaxPool2d-31          [-1, 256, 37, 37]               0
  BasicBlockEntry-32          [-1, 256, 37, 37]               0
           Conv2d-33          [-1, 256, 37, 37]          32,768
      BatchNorm2d-34          [-1, 256, 37, 37]             512
PointwiseSeperableConvolution-35          [-1, 256, 37, 37]               0
             ReLU-36          [-1, 256, 37, 37]               0
           Conv2d-37          [-1, 256, 37, 37]           2,304
           Conv2d-38          [-1, 728, 37, 37]         186,368
      BatchNorm2d-39          [-1, 728, 37, 37]           1,456
DepthwiseSeperableConvolution-40          [-1, 728, 37, 37]               0
             ReLU-41          [-1, 728, 37, 37]               0
           Conv2d-42          [-1, 728, 37, 37]           6,552
           Conv2d-43          [-1, 728, 37, 37]         529,984
      BatchNorm2d-44          [-1, 728, 37, 37]           1,456
DepthwiseSeperableConvolution-45          [-1, 728, 37, 37]               0
             ReLU-46          [-1, 728, 37, 37]               0
        MaxPool2d-47          [-1, 728, 19, 19]               0
  BasicBlockEntry-48          [-1, 728, 19, 19]               0
           Conv2d-49          [-1, 728, 19, 19]         186,368
      BatchNorm2d-50          [-1, 728, 19, 19]           1,456
PointwiseSeperableConvolution-51          [-1, 728, 19, 19]               0
             ReLU-52          [-1, 728, 19, 19]               0
           Conv2d-53          [-1, 728, 19, 19]           6,552
           Conv2d-54          [-1, 728, 19, 19]         529,984
      BatchNorm2d-55          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-56          [-1, 728, 19, 19]               0
             ReLU-57          [-1, 728, 19, 19]               0
           Conv2d-58          [-1, 728, 19, 19]           6,552
           Conv2d-59          [-1, 728, 19, 19]         529,984
      BatchNorm2d-60          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-61          [-1, 728, 19, 19]               0
             ReLU-62          [-1, 728, 19, 19]               0
           Conv2d-63          [-1, 728, 19, 19]           6,552
           Conv2d-64          [-1, 728, 19, 19]         529,984
      BatchNorm2d-65          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-66          [-1, 728, 19, 19]               0
 BasicBlockMiddle-67          [-1, 728, 19, 19]               0
             ReLU-68          [-1, 728, 19, 19]               0
           Conv2d-69          [-1, 728, 19, 19]           6,552
           Conv2d-70          [-1, 728, 19, 19]         529,984
      BatchNorm2d-71          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-72          [-1, 728, 19, 19]               0
             ReLU-73          [-1, 728, 19, 19]               0
           Conv2d-74          [-1, 728, 19, 19]           6,552
           Conv2d-75          [-1, 728, 19, 19]         529,984
      BatchNorm2d-76          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-77          [-1, 728, 19, 19]               0
             ReLU-78          [-1, 728, 19, 19]               0
           Conv2d-79          [-1, 728, 19, 19]           6,552
           Conv2d-80          [-1, 728, 19, 19]         529,984
      BatchNorm2d-81          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-82          [-1, 728, 19, 19]               0
 BasicBlockMiddle-83          [-1, 728, 19, 19]               0
             ReLU-84          [-1, 728, 19, 19]               0
           Conv2d-85          [-1, 728, 19, 19]           6,552
           Conv2d-86          [-1, 728, 19, 19]         529,984
      BatchNorm2d-87          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-88          [-1, 728, 19, 19]               0
             ReLU-89          [-1, 728, 19, 19]               0
           Conv2d-90          [-1, 728, 19, 19]           6,552
           Conv2d-91          [-1, 728, 19, 19]         529,984
      BatchNorm2d-92          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-93          [-1, 728, 19, 19]               0
             ReLU-94          [-1, 728, 19, 19]               0
           Conv2d-95          [-1, 728, 19, 19]           6,552
           Conv2d-96          [-1, 728, 19, 19]         529,984
      BatchNorm2d-97          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-98          [-1, 728, 19, 19]               0
 BasicBlockMiddle-99          [-1, 728, 19, 19]               0
            ReLU-100          [-1, 728, 19, 19]               0
          Conv2d-101          [-1, 728, 19, 19]           6,552
          Conv2d-102          [-1, 728, 19, 19]         529,984
     BatchNorm2d-103          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-104          [-1, 728, 19, 19]               0
            ReLU-105          [-1, 728, 19, 19]               0
          Conv2d-106          [-1, 728, 19, 19]           6,552
          Conv2d-107          [-1, 728, 19, 19]         529,984
     BatchNorm2d-108          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-109          [-1, 728, 19, 19]               0
            ReLU-110          [-1, 728, 19, 19]               0
          Conv2d-111          [-1, 728, 19, 19]           6,552
          Conv2d-112          [-1, 728, 19, 19]         529,984
     BatchNorm2d-113          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-114          [-1, 728, 19, 19]               0
BasicBlockMiddle-115          [-1, 728, 19, 19]               0
            ReLU-116          [-1, 728, 19, 19]               0
          Conv2d-117          [-1, 728, 19, 19]           6,552
          Conv2d-118          [-1, 728, 19, 19]         529,984
     BatchNorm2d-119          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-120          [-1, 728, 19, 19]               0
            ReLU-121          [-1, 728, 19, 19]               0
          Conv2d-122          [-1, 728, 19, 19]           6,552
          Conv2d-123          [-1, 728, 19, 19]         529,984
     BatchNorm2d-124          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-125          [-1, 728, 19, 19]               0
            ReLU-126          [-1, 728, 19, 19]               0
          Conv2d-127          [-1, 728, 19, 19]           6,552
          Conv2d-128          [-1, 728, 19, 19]         529,984
     BatchNorm2d-129          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-130          [-1, 728, 19, 19]               0
BasicBlockMiddle-131          [-1, 728, 19, 19]               0
            ReLU-132          [-1, 728, 19, 19]               0
          Conv2d-133          [-1, 728, 19, 19]           6,552
          Conv2d-134          [-1, 728, 19, 19]         529,984
     BatchNorm2d-135          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-136          [-1, 728, 19, 19]               0
            ReLU-137          [-1, 728, 19, 19]               0
          Conv2d-138          [-1, 728, 19, 19]           6,552
          Conv2d-139          [-1, 728, 19, 19]         529,984
     BatchNorm2d-140          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-141          [-1, 728, 19, 19]               0
            ReLU-142          [-1, 728, 19, 19]               0
          Conv2d-143          [-1, 728, 19, 19]           6,552
          Conv2d-144          [-1, 728, 19, 19]         529,984
     BatchNorm2d-145          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-146          [-1, 728, 19, 19]               0
BasicBlockMiddle-147          [-1, 728, 19, 19]               0
            ReLU-148          [-1, 728, 19, 19]               0
          Conv2d-149          [-1, 728, 19, 19]           6,552
          Conv2d-150          [-1, 728, 19, 19]         529,984
     BatchNorm2d-151          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-152          [-1, 728, 19, 19]               0
            ReLU-153          [-1, 728, 19, 19]               0
          Conv2d-154          [-1, 728, 19, 19]           6,552
          Conv2d-155          [-1, 728, 19, 19]         529,984
     BatchNorm2d-156          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-157          [-1, 728, 19, 19]               0
            ReLU-158          [-1, 728, 19, 19]               0
          Conv2d-159          [-1, 728, 19, 19]           6,552
          Conv2d-160          [-1, 728, 19, 19]         529,984
     BatchNorm2d-161          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-162          [-1, 728, 19, 19]               0
BasicBlockMiddle-163          [-1, 728, 19, 19]               0
            ReLU-164          [-1, 728, 19, 19]               0
          Conv2d-165          [-1, 728, 19, 19]           6,552
          Conv2d-166          [-1, 728, 19, 19]         529,984
     BatchNorm2d-167          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-168          [-1, 728, 19, 19]               0
            ReLU-169          [-1, 728, 19, 19]               0
          Conv2d-170          [-1, 728, 19, 19]           6,552
          Conv2d-171          [-1, 728, 19, 19]         529,984
     BatchNorm2d-172          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-173          [-1, 728, 19, 19]               0
            ReLU-174          [-1, 728, 19, 19]               0
          Conv2d-175          [-1, 728, 19, 19]           6,552
          Conv2d-176          [-1, 728, 19, 19]         529,984
     BatchNorm2d-177          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-178          [-1, 728, 19, 19]               0
BasicBlockMiddle-179          [-1, 728, 19, 19]               0
            ReLU-180          [-1, 728, 19, 19]               0
          Conv2d-181          [-1, 728, 19, 19]           6,552
          Conv2d-182          [-1, 728, 19, 19]         529,984
     BatchNorm2d-183          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-184          [-1, 728, 19, 19]               0
            ReLU-185          [-1, 728, 19, 19]               0
          Conv2d-186          [-1, 728, 19, 19]           6,552
          Conv2d-187          [-1, 728, 19, 19]         529,984
     BatchNorm2d-188          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-189          [-1, 728, 19, 19]               0
            ReLU-190          [-1, 728, 19, 19]               0
          Conv2d-191          [-1, 728, 19, 19]           6,552
          Conv2d-192          [-1, 728, 19, 19]         529,984
     BatchNorm2d-193          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-194          [-1, 728, 19, 19]               0
BasicBlockMiddle-195          [-1, 728, 19, 19]               0
            ReLU-196          [-1, 728, 19, 19]               0
          Conv2d-197          [-1, 728, 19, 19]           6,552
          Conv2d-198          [-1, 728, 19, 19]         529,984
     BatchNorm2d-199          [-1, 728, 19, 19]           1,456
DepthwiseSeperableConvolution-200          [-1, 728, 19, 19]               0
  BasicBlockExit-201          [-1, 728, 19, 19]               0
            ReLU-202          [-1, 728, 19, 19]               0
          Conv2d-203          [-1, 728, 19, 19]           6,552
          Conv2d-204         [-1, 1024, 19, 19]         745,472
     BatchNorm2d-205         [-1, 1024, 19, 19]           2,048
DepthwiseSeperableConvolution-206         [-1, 1024, 19, 19]               0
  BasicBlockExit-207         [-1, 1024, 19, 19]               0
       MaxPool2d-208         [-1, 1024, 10, 10]               0
          Conv2d-209         [-1, 1024, 10, 10]         745,472
     BatchNorm2d-210         [-1, 1024, 10, 10]           2,048
PointwiseSeperableConvolution-211         [-1, 1024, 10, 10]               0
            ReLU-212         [-1, 1024, 10, 10]               0
          Conv2d-213         [-1, 1024, 10, 10]           9,216
          Conv2d-214         [-1, 1536, 10, 10]       1,572,864
     BatchNorm2d-215         [-1, 1536, 10, 10]           3,072
DepthwiseSeperableConvolution-216         [-1, 1536, 10, 10]               0
  BasicBlockExit-217         [-1, 1536, 10, 10]               0
            ReLU-218         [-1, 1536, 10, 10]               0
          Conv2d-219         [-1, 1536, 10, 10]          13,824
          Conv2d-220         [-1, 2048, 10, 10]       3,145,728
     BatchNorm2d-221         [-1, 2048, 10, 10]           4,096
DepthwiseSeperableConvolution-222         [-1, 2048, 10, 10]               0
  BasicBlockExit-223         [-1, 2048, 10, 10]               0
            ReLU-224         [-1, 2048, 10, 10]               0
          Linear-225                 [-1, 1000]       2,049,000
================================================================
Total params: 24,469,928
Trainable params: 24,469,928
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.02
Forward/backward pass size (MB): 788.40
Params size (MB): 93.35
Estimated Total Size (MB): 882.76
----------------------------------------------------------------

"""