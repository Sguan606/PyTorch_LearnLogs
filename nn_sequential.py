'''
Author: 星必尘Sguan
Date: 2025-04-25 14:16:59
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 14:55:21
FilePath: \test_TrainPytorch\nn_sequential.py
Description: 简单的实战搭建,使用到Sequential训练一个完整的CIFAR10_Module

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.tensorboard import SummaryWriter

class Sguan(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024,out_features=64),
            Linear(in_features=64,out_features=10)
        )

    def forward(self,input_py):
        output = self.module1(input_py)
        return output

sguan_py = Sguan()
input_x = torch.ones((64,3,32,32))
output = sguan_py(input_x)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(model=sguan_py,input_to_model=input_x)
writer.close()

