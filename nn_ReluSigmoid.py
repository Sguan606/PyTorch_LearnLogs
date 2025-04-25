'''
Author: 星必尘Sguan
Date: 2025-04-25 13:24:34
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 13:40:26
FilePath: \test_TrainPytorch\nn_ReluSigmoid.py
Description: 对神经网络非线性层激活函数的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.nn import ReLU 
from torch.nn import Sigmoid
import torchvision

dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

input_relu = torch.tensor([[1,-0.5],
                            [-1,3]])

# 首个reshape参数中-1代表黑1代表自
input_relu = torch.reshape(input_relu,(1,1,2,2))

# 非线性激活层函数的搭建
class Sguan(nn.Module):
    def __init__(self):
        super(Sguan,self).__init__()
        # self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,x):
        x = self.sigmoid1(x)
        return x

sguan_sigmoid = Sguan()
# output = sguan_sigmoid(input_relu)

# 显示图像的训练集中
writer = SummaryWriter("logs")
Step = 0
for data in dataloader:
    imgs,targes = data
    img_out = sguan_sigmoid(imgs)
    writer.add_images("Sigmoid_in",imgs,Step)
    writer.add_images("Sigmoid_out",img_out,Step)
    Step += 1

writer.close()
