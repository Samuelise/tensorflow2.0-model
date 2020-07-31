
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
        # ���� GPU Ϊ����ʽռ��
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # ��ӡ�쳣
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
    network = Sequential([  # ��������
        layers.Conv2D(6, kernel_size=3, strides=1),  # ��һ�������, 6 �� 3x3 �����
        layers.BatchNormalization(),   # BN��
        layers.MaxPooling2D(pool_size=2, strides=2),  # �߿������ĳػ���
        layers.ReLU(),  # �����

        layers.Conv2D(16, kernel_size=3, strides=1),  # �ڶ��������, 16 �� 3x3 �����
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),  # �߿������ĳػ���
        layers.ReLU(),  # �����
        layers.Flatten(),  # ��ƽ�㣬����ȫ���Ӳ㴦��
        layers.Dense(120, activation='relu'),  # ȫ���Ӳ㣬 120 ���ڵ�
        layers.BatchNormalization(),
        layers.Dense(84, activation='relu'),  # ȫ���Ӳ㣬 84 �ڵ�
        layers.BatchNormalization(),
        layers.Dense(10)  # ȫ���Ӳ㣬 10 ���ڵ�
    ])
    # build һ������ģ�ͣ������� X ����״������ 4 Ϊ������� batchsz
    network.build(input_shape=(4, 28, 28,1))
    # ͳ��������Ϣ
    network.summary()
    return network

def train(train_db, network, criteon, optimizer, epoch_num):
    for epoch in range(epoch_num):
        correct, total, loss = 0, 0, 0
        for step, (x, y) in enumerate(train_db):
            # �����ݶȼ�¼����
            with tf.GradientTape() as tape:
                # ����ͨ��ά�ȣ� =>[b,28,28,1]
                x = tf.expand_dims(x, axis=3)
                # ǰ����㣬��� 10 ���ĸ��ʷֲ��� [b, 784] => [b, 10]
                out = network(x,training=True)
                pred = tf.argmax(out, axis=-1)
                # ��ʵ��ǩ one-hot ���룬 [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)
                # ���㽻������ʧ����������
                loss += criteon(y_onehot, out)
                # ͳ��Ԥ����ȷ����
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
                # ͳ��Ԥ����������
                total += x.shape[0]


            # �Զ������ݶ�
            grads = tape.gradient(loss, network.trainable_variables)
            # �Զ����²���
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

        print(epoch, 'loss=', float(loss), 'acc=', correct / total)

    return network

def predict(test_db, network):
    # ��¼Ԥ����ȷ������������������
    correct, total = 0, 0
    for x, y in test_db:  # ��������ѵ��������
        # ����ͨ��ά�ȣ� =>[b,28,28,1]
        x = tf.expand_dims(x, axis=3)
        # ǰ����㣬��� 10 ����Ԥ��ֲ��� [b, 784] => [b, 10]
        out = network(x,training=False)
        # ��ʵ������ʱ�Ⱦ��� softmax���� argmax
        # �������� softmax ���ı�Ԫ�صĴ�С��Թ�ϵ����ʡȥ
        pred = tf.argmax(out, axis=-1)
        y = tf.cast(y, tf.int64)
        # ͳ��Ԥ����ȷ����
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        # ͳ��Ԥ����������
        total += x.shape[0]

    # ����׼ȷ��
    print('test acc:', correct / total)


if __name__ == "__main__":

    epoch_num = 20

    train_db,test_db = load_dataset()
    network = build_network()
    # ������ʧ�������࣬��ʵ�ʼ���ʱֱ�ӵ�����ʵ������
    criteon = losses.CategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.RMSprop(0.001)
    network = train(train_db, network, criteon, optimizer, epoch_num)
    predict(test_db, network)

