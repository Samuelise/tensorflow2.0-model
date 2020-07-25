################������ģ��--Unet�������#################

# -*-coding: utf-8 -*-

#�����ʽ [b,h,w,cin]  (������size��ͨ����)
#����˸�ʽ [k,k,cin,cout]  (size��ͨ������ ������)

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

#�����С
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
    #�õ�����ͼƬ��
    for pic in os.listdir(filepath+'src'):
        train_url.append(pic)
    random.shuffle(train_url)   #����
    toltal_num=len(train_url)   
    val_num=int(val_rate*toltal_num)

    #�������ݼ�
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

#��ȡͼƬ�ͱ�ǩ   
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



#unet ����
def unet():
    inputs=input((3,image_w,image_h))

    conv1=tf.nn.conv2d(64,(3,3),padding="same")


#ѵ������
def train():
    Epochs=10
    BS=20
    
    train_set,val_set=get_train_val(val_rate=0.2)
    train_set_len=len(train_set)
    val_set_len=len(val_set)
    print("lenth of train set:%d",train_set_len)
    print("lenth of val set:%d",val_set_len)

    #��ȡ���ݵ�����
    generate_traindata=generateData(20,train_set)       #train
    generate_valdata=generateData(20,val_set)           #val

    # ��ͼ
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
                #��Ҫʱ�����Դ�ռ䣬����ʽռ��
                tf.config.experimental.set_memory_growth(gpu,True) 
        except RuntimeError as e:
            print(e)
    
    train_set,val_set=get_train_val(val_rate=0.2)  #��ȡѵ��������֤��
    train_set_len=len(train_set)    
    val_set_len=len(val_set)    
    print("lenth of train set:",train_set_len)
    print("lenth of val set:",val_set_len)        #��ӡѵ��������֤������
    
    generate_traindata=generateData(1,train_set)       #��ȡtrain��������
    
    #display(generate_traindata)  #��ʾ



