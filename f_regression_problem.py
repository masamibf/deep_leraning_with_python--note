#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: f_regression_problem.py 
@time: 2019/5/16 16:41 
"""

"""
    波士顿房价预测:回归问题
    已知一些数据点，比如犯罪率、当地房产税率等
    404个训练样本，102个测试样本
"""

"""================= 加载波士顿房价数据集 =================="""
from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

# print(train_data.shape)

"""================= 数据标准化 =================="""
mean = train_data.mean(axis=0)
train_data -= mean           #减去特征平均值
std = train_data.std(axis=0)
train_data /= std            #除以特征标准差

test_data -= mean            #均值和标准差都是在训练数据上计算得到,不得使用测试数据
test_data /= std

"""================= 构建网络 =================="""
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))    #无激活函数，这是标量回归的典型设置，激活函数会限制输出范围
    model.compile(
        optimizer='rmsprop',
        loss='mse',          #均方误差:预测值与目标值之差的平方
        metrics=['mae']      #平均绝对误差:预测值与目标值之差的绝对值
    )
    return model

"""================= 利用K折交叉验证来验证方法 =================="""
import numpy as np

k = 4
num_val_samples = len(train_data) // k     #   /:浮点数除法 //:整数除法
num_epochs = 500
all_mae_histories = []
all_scores = []
for i in range(k):
    print('processing fold #', i)
    #准备验证数据：第k个分区的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    #准备训练数据：其他分区所有数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],       #向量拼接
                                         train_data[(i + 1) * num_val_samples:]],
                                         axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]],
                                            axis=0)

    #构建Keras模型(已编译)
    model = build_model()
    # 训练模型 (静默模式, verbose=0)
    history = model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)

    # 在验证数据上评估模型
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # print(all_scores.append(val_mae))

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

#计算所有轮次中的K折验证分数平均值
average_mae_history = [
    np.mean([x(i) for x in all_mae_histories]) for i in range(num_epochs)
]
