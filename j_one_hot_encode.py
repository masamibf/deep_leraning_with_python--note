#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: j_one_hot_encode.py
@time: 2019/6/25 16:08 
"""

"""单词级的one-hot编码"""
import numpy as np

samples = ['The cat sat on the mat.','The dog ate my homework.']
token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10
results = np.zeros(shape = (len(samples),
                            max_length,
                            max(token_index.values()) + 1))
# print(results)
for i,sample in enumerate(samples):
    # print(i,sample)
    for j,word in list(enumerate(sample.split()))[:max_length]:
        # print(j,word)
        index = token_index.get(word)
        # print(index,word)
        results[i,j,index] = 1.
# print(results)

