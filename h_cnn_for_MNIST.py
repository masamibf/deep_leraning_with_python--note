#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: h_cnn_for_MNIST.py 
@time: 2019/5/30 16:30 
"""

"""
搭建卷积神经网络，对MNIST数字分类
"""

"""================= 实例化一个小型的CNN =================="""
from keras import layers
from keras.models import Sequential

def build_cnn_model():
    model = Sequential()
    # filters:32 卷积核数量  kernel_size:(3,3) 卷积核大小
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    # pool_size:(2,2) 输入张量沿两个维度缩小一半
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))

    #添加分类器
    model.add(layers.Flatten())  #多维输出一维化，一般用在卷积层与全链接层之间

    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))

    return model

# print(build_cnn_model().summary())

"""================= 在MNIST图像上训练CNN =================="""
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = build_cnn_model()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)





