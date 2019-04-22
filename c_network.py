#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @长泽雅美男友
@contact: houxiang_fan@163.com 
@file: c_network.py
@time: 2019/4/21 15:12 
"""

"""
    2D张量通常用 密集连接层，也叫 全连接层 或 密集层，对应于Keras的 Dense 类
    3D张量通常用 循环层，用Keras的 LSTM 处理
    4D张量通常用 二维卷积层，用Keras的 Conv2D 处理
    
    在Keras中，构建深度学习模型就是将相互兼容的多个层拼接在一起，以建立有用的数据变换流程
"""


"""======================= 层 深度学习的基础组件 ========================"""
from keras import layers
#创建一个层，有32个输出单元的密集层，只接受第一个维度大小为784的2D张量作为输入
#这个层返回一个张量，第一个维度的大小变成32
#因此这个层后面只能连接一个接受32维向量作为输入的层
layer = layers.Dense(32,input_shape=(784,))


from keras import models
from keras import layers
#其中第二层没有输入形状 input_shape 的参数，它可以自动推导出输入形状等于上一层的输出形状
model = models.Sequential()
model.add(layers.Dense(32,input_shape=(784,)))
model.add(layers.Dense(32))

"""======================= 模型 层构成的网络 ========================"""
#常见网络拓扑结构如下:
#   1.双分支网络(two-branch)
#   2.多头网络  (multihead)
#   3.Inception模块

#选定了网络拓扑结构，意味着将 可能性空间 限定为一系列特定的张量运算，将输入数据映射为输出数据
#然后为这些张量运算的权重张量找到一组合适的值

"""================ 损失函数与优化器 配置学习过程的关键 =================="""
#一旦确定了网络架构，还需要选择两个参数
#   1.损失函数(目标函数)————在训练过程将其最小化，它衡量当前任务是否完成
#   2.优化器————决定如何基于损失函数对网络进行更新，它执行的是SGD的某个变体

#多个输出的神经网络可能具有多个损失函数，但是梯度下降过程必须基于单个标量损失值。
#对于多个损失函数的网络，需要将所有损失函数取平均，变为一个标量值

#常见损失函数的选择：
#   二分类问题---二元交叉熵    多分类问题---分类交叉熵
#   回归问题---均方误差        序列学习问题---联结主义时序分类

"""======================= 使用Keras开发 ========================"""
#定义模型有两种方法: 1.使用sequential类(仅用于层的线性堆叠)
#                  2.使用函数式API(用于层组成的有向无环图，可构建任意形式架构)

#第一种
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(784,)))
model.add(layers.Dense(10,activation='softmax'))

#第二种
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32,activation='relu')(input_tensor)
output_tensor = layers.Dense(10,activation='softmax')(x)
model = models.Model(inputs = input_tensor,outpus = output_tensor)

#单一损失函数
from keras import optimizers

model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss = 'mse',
              metrics=['accuracy'])

model.fit(input_tensor,output_tensor,batch_size=128,epochs=10)

