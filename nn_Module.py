'''
Author: 星必尘Sguan
Date: 2025-04-24 17:35:55
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 19:41:11
FilePath: \test_TrainPytorch\nn_Module.py
Description: 神经网络基本骨架nn.Module的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn

class Sguan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

sguan = Sguan()
x = torch.tensor(1,0)
output = sguan(x)
print(output)
# print神经网络输出
