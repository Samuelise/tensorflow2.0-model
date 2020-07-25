################官方例程#############
#模型：Unet改良版
#数据集：oxford_iiit_pet 3.2.0

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
 
import numpy as np
import os
import random

from PIL import Image
 
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#从磁盘加载原始数据
def load_tensor_from_file(img_file):
    img = Image.open(img_file)
    sample_image = np.array(img)
    #检查输入是否是彩色图像，格式不一致做通道转换
    if len(sample_image.shape) != 3 or sample_image.shape[2] == 4:
        img = img.convert("RGB")
        sample_image = np.array(img)
    sample_image = tf.image.resize(sample_image,[128, 128])
    return sample_image
 
#从磁盘加载mask数据
def load_ann_from_file(img_file):
    sample_image = tf.image.decode_image(tf.io.read_file(img_file))
    #检查输入是否是单通道图，不是进行通道转换
    if sample_image.shape[2] != 1:
        img = Image.open(img_file)
        img = img.convert("L")#转为灰度图
        sample_image = np.array(img)
    sample_image = tf.image.resize(sample_image,[128, 128])
    return sample_image

#加载图片并转换为训练集和验证集，同时输出训练集、验证集样本数量
def load(img_path):
    trainImageList = []
    path = img_path + "/ann"
    files = os.listdir(path)
    cnt = 0
    for imgFile in files:
        if os.path.isdir(imgFile):
            continue
        file = path + "/" + imgFile
        print("load image ", file)
        cnt += 1
        img = load_tensor_from_file(file)
        img = tf.squeeze(img)
        trainImageList.append(img)
        #加载1000张样本，机器配置有限，样本过多，报00M错误
        if cnt > 100:
            break
 
    trainAnnList = []
    path = img_path + "/images"
    files = os.listdir(path)
    cnt = 0
    for imgFile in files:
        if os.path.isdir(imgFile):
            continue
        file = path + "/" + imgFile
        print("load image ", file)
        img = load_ann_from_file(file)
        cnt+=1
        trainAnnList.append(img)
        #加载1000张样本，机器配置有限，样本过多，报00M错误
        if cnt > 100:
            break
    train_x, val_x, train_y, val_y = train_test_split(trainImageList, trainAnnList, test_size=0.2, random_state=3)
    train_num = len(train_x)
    val_num = len(val_x)
    x = tf.convert_to_tensor(train_x, dtype=tf.float32)
    y = tf.convert_to_tensor(train_y, dtype=tf.float32)
    dataset_train = tf.data.Dataset.from_tensor_slices((x, y))
    x_val = tf.convert_to_tensor(val_x, dtype=tf.float32)
    y_val = tf.convert_to_tensor(val_y, dtype=tf.float32)
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    return dataset_train, train_num, dataset_val, val_num
 
train_dataset, train_num, val_dataset, val_num = load("data/Unet/oxford_iiit_pet")

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 128.0 - 1
    #mask数据需要根据具体内容进行转换，否则容易导致loss=nan
    #此处：[0 1 2]=[内部  外部  边缘]
    input_mask = input_mask-1
    return input_image, input_mask
 
@tf.function
def load_image_train(x, y):
    input_image = tf.image.resize(x, (128, 128))
    input_mask = tf.image.resize(y, (128,128))
    if tf.random.uniform(()) -0.5>0 :
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask
 
def load_image_test(x, y):
    input_image = tf.image.resize(x, (128, 128))
    input_mask = tf.image.resize(y, (128,128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask
 
TRAIN_LENGTH = train_num
#根据硬件性能调整batch_size大小
BATCH_SIZE = 16#64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH= TRAIN_LENGTH
 
train = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(load_image_test)
 
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
 
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
 
np.set_printoptions(threshold=128*128)
for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    print(tf.reduce_min(mask), tf.reduce_max(mask), tf.reduce_mean(mask))
    display([sample_image,sample_mask])
 
OUTPUT_CHANNELS = 3
 
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3], include_top=False)
layer_names = [
    'block_1_expand_relu', #64x64
    'block_3_expand_relu', #32x32
    'block_6_expand_relu', #16x16
    'block_13_expand_relu',#8x8
    'block_16_project',    #4x4
]
 
layers = [base_model.get_layer(name).output for name in layer_names]
#创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = True
 
up_stack =[
    pix2pix.upsample(512, 3),#4x4 -> 8x8
    pix2pix.upsample(256, 3),#8x8 -> 16x16
    pix2pix.upsample(128, 3),#16x16 -> 32x32
    pix2pix.upsample(64, 3), #32x32 -> 64x64
]
 
def unet_model(output_channels):
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2,padding='same', activation='softmax')
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    x = inputs
 
    #降频采样
    skips = down_stack(x)
    x = skips[-1]#??????????
    skips=reversed(skips[:-1])
 
    #升频采样
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
 
    x = last(x)
 
    return tf.keras.Model(inputs=inputs, outputs=x)
 
model = unet_model(OUTPUT_CHANNELS)
adam = tf.keras.optimizers.Adam(lr=1e-3)
#optimizer = optimizers.Adam(lr=1e-3)
model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
 
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, 
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
 
show_predictions()
 
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=False)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
 
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = val_num//BATCH_SIZE//VAL_SUBSPLITS
 
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=val_dataset,
                          callbacks=[DisplayCallback()])

model.save("poker.h5")
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()