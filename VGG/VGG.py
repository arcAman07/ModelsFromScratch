#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torch.nn as nn


# In[6]:


a = torch.tensor([1.])


# In[7]:


a


# In[8]:


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


# In[9]:


x_data


# In[10]:


from PIL import Image


# In[11]:


image = Image.open("/Users/deepaksharma/Desktop/VGG.png")


# In[12]:


image


# In[64]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.activation(self.conv_layer(x))


# In[68]:


class B_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(B_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[69]:


def train():
    x = torch.rand((3,3,224,224))
    model = B_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[70]:


train()


# In[71]:


class C_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(C_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2], kernel_size = 1, padding = 0, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3], kernel_size = 1, padding = 0, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3], kernel_size = 1, padding = 0, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[72]:


def train():
    x = torch.rand((3,3,224,224))
    model = C_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[73]:


train()


# In[76]:


class D_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(D_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[77]:


def train():
    x = torch.rand((3,3,224,224))
    model = D_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[78]:


train()


# In[79]:


class E_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(E_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[80]:


def train():
    x = torch.rand((3,3,224,224))
    model = E_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[81]:


train()


# In[82]:


class A_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(A_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[83]:


def train():
    x = torch.rand((3,3,224,224))
    model = A_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[84]:


train()


# In[85]:


class A_LRN_VGG(nn.Module):
    def __init__(self, in_channels = 3, out_classes = 10, channels = [64, 128, 256, 512]):
        super(A_LRN_VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_network = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            nn.BatchNorm2d(channels[0]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[0], channels[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[2]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ConvBlock(channels[3], channels[3]),
            ConvBlock(channels[3], channels[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.outputs = nn.Sequential(
            nn.Linear(4096, out_classes),
            nn.Softmax(),
        )
    def forward(self, x):
        x = self.conv_network(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.outputs(x)
        return x


# In[86]:


def train():
    x = torch.rand((3,3,224,224))
    model = A_LRN_VGG()
    y = model(x)
    print(y)
    print(y.shape)


# In[87]:


train()


# In[ ]:




