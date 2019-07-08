#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: g_generalization.py 
@time: 2019/5/29 15:45 
"""

"""
权重正则化: 
    L1正则化:添加的成本与权重系数的绝对值(权重的L1范数)成正比
    L2正则化:添加的成本与权重系数的平方(权重的L2范数)成正比
    神经网络的L2正则化也叫 权重衰减
"""

"""
dropout正则化:
    在训练过程中随机将该层的一些输出特征舍弃(设为0)
    概率值通常为0.2-0.5
"""


"""================= 向模型添加L2权重正则化 =================="""
from keras import regularizers
from keras.models import Sequential
from keras import layers

model = Sequential()
# l2(0.001) 的意思是该层权重矩阵的每个系数都会使网络总损失增加 0.001×weight_coefficient_value
model.add(layers.Dense(16,
                       kernel_regularizer = regularizers.l2(0.001),
                       # kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001)
                       activation = 'relu',
                       input_shape = (10000,)))
model.add(layers.Dense(16,
                       kernel_regularizer = regularizers.l2(0.001),
                       activation = 'relu'))

"""================= 添加dropout正则化 =================="""
model2 = Sequential()
model2.add(layers.Dense(16,
                        activation = 'relu',
                        input_shape = (10000,)))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(16,
                        activation = 'relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1,
                        activation = 'sigmoid'))
