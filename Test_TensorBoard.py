'''
Author: 星必尘Sguan
Date: 2025-04-24 11:34:09
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 12:24:30
FilePath: \test_TrainPytorch\Test_TensorBoard.py
Description: TensorBoard的使用

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

image_path = "data\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(image_path)
img = np.array(img_PIL)

writer.add_image("sguanTest",img,1,dataformats='HWC')
# y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x",2*i,i)

writer.close()

