from model import AlexNet

model = AlexNet()
from torchvision import models
from torchsummary import summary

print(summary(model.cuda(),(3, 227 , 227)))