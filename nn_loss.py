'''
Author: 星必尘Sguan
Date: 2025-04-25 15:12:11
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 15:35:36
FilePath: \test_TrainPytorch\nn_loss.py
Description: 神经网络Loss损失函数的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch.nn import CrossEntropyLoss

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))

loss = CrossEntropyLoss()
result = loss(x,y)
print(result)


