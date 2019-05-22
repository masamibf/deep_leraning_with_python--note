#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @长泽雅美男友
@contact: houxiang_fan@163.com 
@file: e_multi_classification.py 
@time: 2019/4/22 21:49 
"""

"""
    新闻分类：多分类问题
    一共46个不同主题的新闻，每个主题至少10个样本
"""

"""================= 加载路透社数据集 =================="""
from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))

"""================= 将索引解码为新闻文本 =================="""
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decode_newswire = ' '.join([reverse_word_index.get(i - 3,'?') for i in train_data[0]])

"""================= 准备数据 =================="""
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)   #将训练数据向量化
x_test = vectorize_sequences(test_data)     #将测试数据向量化

#将标签向量化有两种方法：可以将标签列表转换为整数向量；或者使用one-hot编码

def to_one_hot(labels,dimension = 46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#keras内置方法也可以实现该操作
from keras.utils.np_utils import to_categorical

one_hot_train_labels_2 = to_categorical(train_labels)
one_hot_test_labels_2 = to_categorical(test_labels)

"""================= 构建网络 =================="""
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

"""================= 编译模型 =================="""
model.compile(
    optimizer='rmsprop',
    loss = 'categorical_crossentropy',    #分类交叉熵，用于衡量两个概率分布之间的距离（网络输出与真实分布）
    metrics = ['accuracy']
)

"""================= 留出验证集 =================="""
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

"""================= 训练模型 =================="""
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,       #epochs = 9
    batch_size=512,
    validation_data=(x_val,y_val)
)

"""================= 绘制训练损失和验证损失 =================="""
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""================= 绘制训练精度和验证精度 =================="""
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#模型在9个epoch后开始过拟合，重新训练，共9个epoch

"""================= 生成预测结果 =================="""
predictions = model.predict(x_test)    #predictions 中每个元素都是长度为46的向量

print(np.argmax(predictions[0]))       #输出概率最大的类别

