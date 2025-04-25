'''
Author: 星必尘Sguan
Date: 2025-04-25 22:10:54
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 22:20:05
FilePath: \test_TrainPytorch\train_gpu1.py
Description: 利用GPU训练神经网络模型

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn
from torch import cuda
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 1.准备的一个测试集
train_data = torchvision.datasets.CIFAR10(root=".\\dataset",train=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,transform=torchvision.transforms.ToTensor())

# 2.length长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 3.利用DataLoader加载数据集
train_loader = DataLoader(dataset=train_data,batch_size=4,shuffle=False,num_workers=0,drop_last=True)
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=False,num_workers=0,drop_last=True)

# 4.搭建神经网络
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

# 5.创建网络模型
sguan_run = Sguan()
if torch.cuda.is_available():
    sguan_run = sguan_run.cuda()

# 6.损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 7.优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(sguan_run.parameters(),lr=learning_rate)

# 8.设置训练网络的参数
total_train_step = 0    # 记录训练的次数
total_test_step = 0     # 记录测试的次数
epoch = 2               # 神经网络训练的轮数

# 9.训练Loss损失函数可视化
writer = SummaryWriter("logs")

# 10.神经网络模型的训练and测试
for i in range(epoch):
    print("-------第 {} 轮训练开始runing...-------".format(i+1))
    for data in train_loader:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = sguan_run(imgs)
        # 1运行损失函数
        loss = loss_fn(outputs,targets)
        # 2清零函数梯度
        optimizer.zero_grad()
        # 3反向传播
        loss.backward()
        # 4调用优化器（参数优化）
        optimizer.step()
        total_train_step += 1
        if total_train_step % 1000 == 0:
            print("训练次数: {}次,Loss: {}".format(total_train_step,loss.item()))
            # 5可视化训练参数Loss
            writer.add_scalar("sguan_myTrainLog",loss.item(),total_train_step)
        
    # 6测试步骤开始
    total_test_loss = 0     # 整体训练的Loss损失函数
    total_accuracy = 0      # 整体训练的准确个数（基于测试集test_loader）
    with torch.no_grad():
        for test_data in test_loader:
            imgs,targets = test_data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = sguan_run(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 7计算训练网络的一个准确率
            accuracy = (outputs.argmax(1) == targets).sum
            total_accuracy += accuracy
    print("整体测试集上的loss: {}".format(total_test_loss))
    print("整体测试集的准确率: {}".format(total_accuracy/test_data_size))
    # 8可视化测试参数Loss
    writer.add_scalar("sguan_myTestLog",loss.item(),total_train_step)
    writer.add_scalar("sguan_myTestMain",total_accuracy/test_data_size,total_train_step)

    # 9整体测试次数记录
    total_train_step += 1