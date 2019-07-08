#!/usr/bin/env python  
# encoding: utf-8  

""" 
@author: @樊厚翔
@contact: houxiang_fan@163.com 
@file: i_dogs_vs_cats.py 
@time: 2019/6/24 11:19 
"""

import os,shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#原始数据集路径
original_dataset_dir = 'D://my_python_code//python_with_deep_learning//kaggle//train'
#保存较小数据集路径
base_dir = 'D://my_python_code//python_with_deep_learning//cats_and_dogs_small'

def MakeDir(name):
    os.mkdir(base_dir)

    os.mkdir(os.path.join(base_dir,'train'))
    os.mkdir(os.path.join(base_dir,'val'))
    os.mkdir(os.path.join(base_dir,'test'))

    os.mkdir(os.path.join(base_dir,'train',name))
    os.mkdir(os.path.join(base_dir,'val',name))
    os.mkdir(os.path.join(base_dir,'test',name))

def CopyFile(name,type,start,end):
    n = name.strip('s')
    fnames = ['{0}.{1}.jpg'.format(n,i) for i in range(start,end)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir,fname)
        dir = os.path.join(base_dir,type,name)
        dst = os.path.join(dir,fname)
        shutil.copyfile(src,dst)

def BuildCnnModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.RMSprop(lr=1e-4),
                  metrics = ['acc'])

    return model

def PreProcessImage():
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,      #随机旋转角度范围
        width_shift_range = 0.2,  #水平方向平移范围
        height_shift_range = 0.2, #垂直方向平移范围
        shear_range = 0.2,        #随机错切变换角度
        zoom_range = 0.2,         #随机缩放范围
        horizontal_flip = True,   #随机将一半图像水平翻转
    )
    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_dir = base_dir + '//train'
    val_dir = base_dir + '//val'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150,150),
        batch_size = 20,
        class_mode = 'binary')

    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = 'binary')

    return train_generator,val_generator

def FitCnnModel():
    train_generator,val_generator = PreProcessImage()
    model = BuildCnnModel()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = val_generator,
        validation_steps = 50
    )
    model.save('cats_and_dogs_small.h5')

    return history

def Plot():
    history = FitCnnModel()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # #猫和狗的训练、验证、测试目录
    # MakeDir('cats')
    # MakeDir('dogs')
    #
    # #复制图像集
    # CopyFile('cats','train',0,1000)
    # CopyFile('cats','val',1000,1500)
    # CopyFile('cats','test',1500,2000)
    # CopyFile('dogs','train',0,1000)
    # CopyFile('dogs','val',1000,1500)
    # CopyFile('dogs','test',1500,2000)

    # import os
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # BuildCnnModel()
    # PreProcessImage()
    # FitCnnModel()
    # Plot()

    # import tensorflow as tf
    #
    # a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    # b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    # c = tf.matmul(a, b)
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # print(sess.run(c))
    pass

