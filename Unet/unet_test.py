################神经网络模型--Unet网络测试#################

# -*-coding: utf-8 -*-

#输入格式 [b,h,w,cin]  (数量，size，通道数)
#卷积核格式 [k,k,cin,cout]  (size，通道数， 核数量)

import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder  

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output

from PIL import Image 



filepath='data/Unet/unet_train/'
seed = 7  
np.random.seed(seed)  

#输入大小
image_w=256
image_h=256


def data_load(path,grayscale=False):
    if grayscale :
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img=cv2.imread(path)
        img=np.array(img,dtype="float")/255.0
    return img
    
def get_train_val(val_rate=0.2):
    train_url=[]
    train_set=[]
    val_set=[]
    #得到所有图片名
    for pic in os.listdir(filepath+'src'):
        train_url.append(pic)
    random.shuffle(train_url)   #打乱
    toltal_num=len(train_url)   
    val_num=int(val_rate*toltal_num)

    #划分数据集
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

#获取图片和标签   
def generateData(batch_size,data=[]):
    while True:
        train_data=[]
        train_label=[]
        batch=0
        for i in range(len(data)):
            url=data[i]
            batch+=1
            img=data_load(filepath+'src/'+url)
            #img=img_to_array(img)
            train_data.append(img)
            label=data_load(filepath+'label/'+url)
            #label=img_to_array(label)
            train_label.append(label)
            if batch==batch_size: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)
                train_data=[]
                train_label=[]
                batch=0  



#unet 网络
def unet():
    inputs=input((3,image_w,image_h))

    conv1=tf.nn.conv2d(64,(3,3),padding="same")


#训练网络
def train():
    Epochs=10
    BS=20
    
    train_set,val_set=get_train_val(val_rate=0.2)
    train_set_len=len(train_set)
    val_set_len=len(val_set)
    print("lenth of train set:%d",train_set_len)
    print("lenth of val set:%d",val_set_len)

    #获取数据迭代器
    generate_traindata=generateData(20,train_set)       #train
    generate_valdata=generateData(20,val_set)           #val

    # 画图
    plt.style.use("ggplot")
    plt.figure()
    N=Epochs

def display(generatedata):
    
    plt.figure(figsize=(10,10))

    title=["image","mask"]
    
    sample=next(generatedata)
    image=np.squeeze(sample[0])
    label=np.squeeze(sample[1])
    plt.subplot(1,2,1)
    plt.title(title[0])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.subplot(1,2,2)
    plt.title(title[1])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(label))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":


    gpus=tf.config.experimental.list_physical_devices('GPU0')
    if gpus :
        try:
            for gpu in gpus:
                #需要时申请显存空间，增长式占用
                tf.config.experimental.set_memory_growth(gpu,True) 
        except RuntimeError as e:
            print(e)
    
    train_set,val_set=get_train_val(val_rate=0.2)  #获取训练集和验证集
    train_set_len=len(train_set)    
    val_set_len=len(val_set)    
    print("lenth of train set:",train_set_len)
    print("lenth of val set:",val_set_len)        #打印训练集和验证机长度
    
    generate_traindata=generateData(1,train_set)       #获取train迭代数据
    
    #display(generate_traindata)  #显示



