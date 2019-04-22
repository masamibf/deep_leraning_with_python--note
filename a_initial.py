#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @长泽雅美男友
@contact: 374454765@qq.com 
@file: a_initial.py
@time: 2019/3/27 10:55 
"""


"""====加载Keras中的mnist数据集===="""
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

# print(test_images.shape)
# print(len(train_labels))
# print(test_labels)

"""====网络架构===="""
from keras import models
from keras import layers

network = models.Sequential()
# 模型第一层需知道输入数据的shape参数(input_shape),后面各层自动的推导中间数据的shape
network.add(layers.Dense(512,activation = 'relu',input_shape = (28*28,)))
network.add(layers.Dense(10,activation = 'softmax'))

"""====编译步骤===="""
network.compile(optimizer = 'rmsprop',             #优化器
                loss = 'categorical_crossentropy', #损失函数
                metrics = ['accuracy'])            #指标列表

"""====准备图像数据===="""
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

"""====准备标签===="""
from keras.utils import  to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 调用fit方法在训练数据上拟合模型
# 结果:训练精度acc = 98.89% ,测试精度acc = 98.02%
network.fit(train_images,train_labels,epochs = 5,batch_size = 128)
test_loss,test_acc = network.evaluate(test_images,test_labels)
print(test_acc)


