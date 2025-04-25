'''
Author: 星必尘Sguan
Date: 2025-04-24 19:40:10
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 20:32:32
FilePath: \test_TrainPytorch\nn_conv.py
Description: 神经网络卷积核CNN的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
import torch.nn.functional as F

img_input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 图片重定义reshap
img_input = torch.reshape(img_input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

# 打印shape参数，方便我们观看
print(img_input.shape)
print(kernel.shape)

# 卷积核函数conv操作(不带Padding边沿填充)
output1 = F.conv2d(img_input,kernel,stride=1)
print(output1)

# 卷积核conv(带Padding边沿填充)
output2 = F.conv2d(img_input,kernel,stride=1,padding=1)
print(output2)


