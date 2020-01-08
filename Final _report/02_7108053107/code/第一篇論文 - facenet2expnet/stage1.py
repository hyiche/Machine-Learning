
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

import glob
import pandas as pd
import numpy as np
import cv2
#####################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#####################################

folders_name = 'Emotion'
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
    output_label = []
    for folder in folders:
        for f in glob.glob(folder + '/*'):
            for i in glob.glob(f + '/*'):
                for j in glob.glob(i + '/*'):
                    num = [1, 2, 3, 4, 5, 6, 7, 8] # 總共有八種表情
                    with open(j, 'r') as f:
                        txt = f.read()
                        txt = eval(txt)
                        num.append(txt)
                        label = pd.get_dummies(num)
                        label = label.values.tolist()
                        label = np.asarray(label[-1])
                        image_root = i.replace('Emotion', 'CK_image')
                        image = load_data(image_root)
                        for times in range(5):
                            output_data.append(image[times])
                            output_label.append(label)
    return output_data,output_label

def ck_vgg(train_image,train_label,test_image,test_label,epochs):
    model_vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    model_vgg.summary()
    for layer in model_vgg.layers:
        layer.trainable = True  # 去調整之前的卷積層的參數
    model = Flatten()(model_vgg.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(1024, activation='relu', name='fc2')(model)
    model = Dense(256, activation='relu', name='fc3')(model)
    model = Dropout(0.5)(model)
    model = Dense(8, activation='softmax', name='prediction')(model)
    model_vgg_CK = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    adam = Adam(lr=1e-4, decay=1e-7)  # lr 學習率 decay 梯度的逐漸減小 每叠代一次梯度就下降 0.05*（1-（10的-5））這樣來變
    model_vgg_CK.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model_vgg_CK.fit(train_image, train_label, validation_data=(test_image, test_label), epochs=epochs)
    model_vgg_CK.save('vgg_pretrain.h5')
    model_vgg_CK.save_weights('pretrain_weight.h5')
    json_string = model_vgg_CK.to_json()
    print('finish')
data,label = load_data_and_label(folders_name)

train_image = np.array(data[:len(data)-100])
train_label = np.array(label[:len(label)-100])
test_image = np.array(data[len(data)-100:])
test_label = np.array(label[len(label)-100:])
ck_vgg(train_image,train_label,test_image,test_label,epochs=50)