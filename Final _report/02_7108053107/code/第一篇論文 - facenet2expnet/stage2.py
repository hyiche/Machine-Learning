import numpy as np
import cv2
import glob
import tensorflow as tf
from keras.layers import Dropout, Conv2D, MaxPooling2D,AveragePooling2D
from keras.models import Sequential
#########################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#########################################

def load_keras_model():
    model = tf.contrib.keras.models.load_model('vgg_pretrain.h5')
    weights_list = model.get_weights()
    for i, weights in enumerate(weights_list[0:16]):
        model.layers[i].set_weights(weights)


    return

def l2_norm(x):
    return x
def MSE_loss(emotion_net_conv,keras_net_conv):
    if emotion_net_conv.shape == keras_net_conv.shape:
        delta = tf.subtract(emotion_net_conv,keras_net_conv,name='matrix_loss')
        delta = l2_norm(delta)
        loss = tf.reduce_mean(AveragePooling2D(pool_size=(emotion_net_conv.shape[0], emotion_net_conv.shape[1]), strides=None, padding='valid', data_format=None)(delta))
        return loss

def emotion_model(train_image,train_keras_conv,test_image,test_keras_conv):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape = (224, 224, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', input_shape = (112, 112, 64), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', input_shape = (56, 56, 128), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape = (28, 28, 256), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', input_shape = (14, 14, 512), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    loss = MSE_loss(model.output,test_keras_conv)
    model.compile(loss=loss, optimizer ='adam', metrics = ['accuracy'])
    train_history = model.fit(x=train_image, y=train_keras_conv, validation_data=(test_image, test_keras_conv), epochs=30)


def load_data(folder):
    select_image = []
    for j in glob.glob(folder + '/*'):
        select_image.append(j)
    select_image = select_image[len(select_image) - 5:]  # 最後五張

    images = []
    for image in select_image:
        img = cv2.imread(image)
        img = cv2.resize(img,(224,224))
        images.append(img)
    return images

def load_data_and_label(folder_name):
    #folder_name = 'CK+ data\Emotion' #找有答案的照片檔
    folders = glob.glob(folder_name)
    output_data = []
    for folder in folders:
        for f in glob.glob(folder + '/*'):
            for i in glob.glob(f + '/*'):
                for j in glob.glob(i + '/*'):
                    num = [1, 2, 3, 4, 5, 6, 7, 8] # 總共有八種表情
                    with open(j, 'r') as f:
                        txt = f.read()
                        txt = eval(txt)
                        num.append(txt)
                        image_root = i.replace('Emotion', 'CK_image')
                        image = load_data(image_root)
                        for times in range(5):
                            output_data.append(image[times])
    return output_data
folders_name = 'Emotion'
data = load_data_and_label(folders_name)
train_image = np.array(data[:len(data)-100])
test_image = np.array(data[len(data)-100:])
