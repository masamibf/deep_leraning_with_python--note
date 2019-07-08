#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: b_tensor.py
@time: 2019/3/30 10:02 
"""

import numpy as np

x = np.array(12)             #  0D张量(标量)
# print(x.ndim)
y = np.array([12,6,3,14,7])  #  1D张量(向量)

z = np.array([[5,78,2,34,0], #  2D张量(矩阵)
              [6,79,3,35,1],
              [7,80,4,36,2]])

m = np.array([[[5,78,2,34,0],#  3D张量(立方体)
               [6,79,3,35,1],
               [7,80,4,36,2]],
              [[5,78,2,34,0],
               [6,79,3,35,1],
               [7,80,4,36,2]],
              [[5,78,2,34,0],
               [6,79,3,35,1],
               [7,80,4,36,2]]])

"""====加载Keras中的mnist数据集===="""
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

"""====显示第5个数字===="""
import matplotlib.pyplot as plt
# digit = train_images[5]
# print(digit)
# plt.imshow(digit,cmap = plt.cm.binary)
# plt.show()

"""====张量切片===="""
my_slice1 = train_images[10:100]           #选择第10~100个数字(不包括第100个)
my_slice2 = train_images[10:100, :, :]     #同1
my_slice3 = train_images[10:100,0:28,0:28] #同1
my_slice4 = train_images[:,14: ,14: ]      #选择每张图片的右下角14*14像素
my_slice5 = train_images[:,7:-7,7:-7]      #在图像中心裁剪出14*14像素


"""====张量广播===="""
#较小的张量会被广播，以匹配较大张量的形状
xx = np.random.random((2,2))
yy = np.random.random((1,1))

zz = np.maximum(xx,yy)
print('xx = ',xx,
      'yy = ',yy,
      'zz = ',zz)

"""====张量变形===="""

xxx = np.zeros((300,20))
xxx = np.transpose(xxx)
print(xxx.shape)
