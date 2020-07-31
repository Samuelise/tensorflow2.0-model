
import os
import numpy as np 
import matplotlib.pyplot as plt
import struct
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Sequential, layers, losses, datasets, optimizers


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)


filepath='data/Mnist'


def load_data(filepath,kind='train'):
    label_path = filepath+'/'+kind+'_label.idx1-ubyte'
    image_path = filepath+'/'+kind+'_image.idx3-ubyte'

    with open(label_path,'rb') as lbpath:
        magic,n=struct.unpack('>II',lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)

    with open(image_path,'rb') as impath:
        magic,num,rows,cols=struct.unpack('>IIII',impath.read(16))
        images=np.fromfile(impath,dtype=np.uint8).reshape(len(labels),784)

    return images,labels

def preprocess(x,y):
    print(x.shape,y.shape)
    x=tf.cast(x,dtype=tf.float32)/255.
    x=tf.reshape(x,[-1,28,28])
    y=tf.cast(y,dtype=tf.int64)
    return x,y

def load_dataset():
    image_train,label_train=load_data(filepath,kind='train')
    image_test,label_test=load_data(filepath,kind='test')

    image_train=tf.convert_to_tensor(image_train,dtype=tf.float32)
    label_train=tf.convert_to_tensor(label_train,dtype=tf.int64)

    image_test=tf.convert_to_tensor(image_test,dtype=tf.float32)
    label_test=tf.convert_to_tensor(label_test,dtype=tf.int64)

    batchsz=128
    train_db=tf.data.Dataset.from_tensor_slices((image_train,label_train))
    train_db = train_db.shuffle(1000).batch(batchsz).map(preprocess)

    test_db=tf.data.Dataset.from_tensor_slices((image_test,label_test))
    test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)

    return train_db,test_db

def build_network():
    network = Sequential([  # 网络容器
        layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
        layers.BatchNormalization(),   # BN层
        layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        layers.ReLU(),  # 激活函数

        layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
        layers.ReLU(),  # 激活函数
        layers.Flatten(),  # 打平层，方便全连接层处理
        layers.Dense(120, activation='relu'),  # 全连接层， 120 个节点
        layers.BatchNormalization(),
        layers.Dense(84, activation='relu'),  # 全连接层， 84 节点
        layers.BatchNormalization(),
        layers.Dense(10)  # 全连接层， 10 个节点
    ])
    # build 一次网络模型，给输入 X 的形状，其中 4 为随意给的 batchsz
    network.build(input_shape=(4, 28, 28,1))
    # 统计网络信息
    network.summary()
    return network

def train(train_db, network, criteon, optimizer, epoch_num):
    for epoch in range(epoch_num):
        correct, total, loss = 0, 0, 0
        for step, (x, y) in enumerate(train_db):
            # 构建梯度记录环境
            with tf.GradientTape() as tape:
                # 插入通道维度， =>[b,28,28,1]
                x = tf.expand_dims(x, axis=3)
                # 前向计算，获得 10 类别的概率分布， [b, 784] => [b, 10]
                out = network(x,training=True)
                pred = tf.argmax(out, axis=-1)
                # 真实标签 one-hot 编码， [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)
                # 计算交叉熵损失函数，标量
                loss += criteon(y_onehot, out)
                # 统计预测正确数量
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
                # 统计预测样本总数
                total += x.shape[0]


            # 自动计算梯度
            grads = tape.gradient(loss, network.trainable_variables)
            # 自动更新参数
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

        print(epoch, 'loss=', float(loss), 'acc=', correct / total)

    return network

def predict(test_db, network):
    # 记录预测正确的数量，总样本数量
    correct, total = 0, 0
    for x, y in test_db:  # 遍历所有训练集样本
        # 插入通道维度， =>[b,28,28,1]
        x = tf.expand_dims(x, axis=3)
        # 前向计算，获得 10 类别的预测分布， [b, 784] => [b, 10]
        out = network(x,training=False)
        # 真实的流程时先经过 softmax，再 argmax
        # 但是由于 softmax 不改变元素的大小相对关系，故省去
        pred = tf.argmax(out, axis=-1)
        y = tf.cast(y, tf.int64)
        # 统计预测正确数量
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        # 统计预测样本总数
        total += x.shape[0]

    # 计算准确率
    print('test acc:', correct / total)


if __name__ == "__main__":

    epoch_num = 20

    train_db,test_db = load_dataset()
    network = build_network()
    # 创建损失函数的类，在实际计算时直接调用类实例即可
    criteon = losses.CategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.RMSprop(0.001)
    network = train(train_db, network, criteon, optimizer, epoch_num)
    predict(test_db, network)

