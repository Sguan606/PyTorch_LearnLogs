'''
Author: 星必尘Sguan
Date: 2025-04-25 19:22:44
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 21:04:26
FilePath: \test_TrainPytorch\model_save.py
Description: 网络模型的保存与读取(重在保存)

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式一
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式二（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

# 陷阱(不能使用第一种方式的save函数保存自己的模型)
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
sguan_run = Sguan()
torch.save(sguan_run,"sguan_method1.pth")