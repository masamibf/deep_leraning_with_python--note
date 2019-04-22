#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @长泽雅美男友
@contact: houxiang_fan@163.com 
@file: d_binary_classification.py 
@time: 2019/4/21 17:26 
"""

"""电影评论分类:二分类问题"""


"""===================== 加载IMDB数据集 ======================"""
from keras.datasets import imdb
#仅保留训练数据中前1000个最常出现单词，低频词将被舍弃
#data是 单词索引 组成的评论列表  labels 是由0 1组成的态度列表 0表示负面 1表示积极
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=1000)
# print(train_data[0])

#word_index是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()
#键值颠倒，将整数索引映射为单词
reverse_word_index = dict(
    [(value,key) for (key,value) in word_index.items()])
#将评论解码，索引减3，因为0 1 2 是填充词
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3,'?') for i in train_data[0]])
#第一条评论
print(decoded_review)

"""======================= 准备数据 ========================"""
#将整数序列编码为二进制矩阵
import numpy as np

def vectorize_sequences(sequences,dimension = 1000):
    """创建一个形状为 (len(sequences),dimension) 的零矩阵"""
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

#训练数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

"""======================= 模型定义 ========================"""
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (1000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))

"""======================= 编译模型 ========================"""
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(
    optimizer = optimizers.RMSprop(lr = 0.001),   #配置优化器
    loss = losses.binary_crossentropy,            #自定义损失和指标
    metrics = [metrics.binary_accuracy]
)

"""======================= 留出验证集 ========================"""
#为了验证，需将原始训练数据留出10000个样本作为验证集

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

"""======================= 训练模型 ========================"""
model.compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['acc']
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = (x_val,y_val)
)

history_dict = history.history
print(history_dict.keys())

"""================= 绘制训练损失和验证损失 =================="""
import matplotlib.pyplot as plt

history_dict = history.history
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_value) + 1)

plt.plot(epochs,loss_value,'bo',label = 'Training loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


"""================= 绘制训练损失和验证损失 =================="""
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

"""===================== 生成预测结果 ======================"""
res = model.predict(x_test)
print(res)
