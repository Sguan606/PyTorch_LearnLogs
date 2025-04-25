'''
Author: 星必尘Sguan
Date: 2025-04-25 15:37:21
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 17:42:47
FilePath: \test_TrainPytorch\nn_loss_network.py
Description: 神经网络结合损失函数的反向传播实战学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# 测试的数据集
dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

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

loss = CrossEntropyLoss()
sguan_loss = Sguan()
for data in dataloader:
    imgs,targets = data
    img_out = sguan_loss(imgs)
    # print(img_out)
    result = loss(img_out,targets)
    # print(result)
    result.backward()


