'''
Author: 星必尘Sguan
Date: 2025-04-24 20:32:50
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 21:39:46
FilePath: \test_TrainPytorch\nn_Conv2D.py
Description: 卷积神经网络的学习CNN|Conv2D

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torch.nn import Conv2d
from torch import nn

dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

# 卷积层CNN函数搭建
class Sguan(nn.Module):
    def __init__(self):
        super(Sguan,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.conv1(x)
        return x

# 具体卷积操作CNN
sguan_cnn = Sguan()
Step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs,targes = data
    output = sguan_cnn(imgs)
    # print(imgs.shape) 
    # print(output.shape)
    output_show = output[:, :3, :, :]  # 选择前3个通道
    writer.add_images("nn_input",imgs,Step)
    writer.add_images("nn_output",output_show,Step)
    Step += 1

writer.close()
