import numpy as np
import cv2
import glob
import pandas as pd
from keras.optimizers import SGD,Adam
from keras.layers import Conv2D,Flatten,Dense,Dropout
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Model,Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



#############################################################
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.keras.backend.set_session(sess)
################################################################

def load_data(folder):
    select_image = []
    for j in glob.glob(folder + '/*'):
        select_image.append(j)
    select_image = select_image[len(select_image) - 6:]  # 最後五張

    images = []
    for image in select_image:
        # print(image)
        img = cv2.imread(image)
        img = cv2.resize(img,(224,224))
        img = img / 255
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
                    num = [1,2, 3, 4, 5, 6, 7,8] # 總共有八種表情
                    with open(j, 'r') as f:
                        txt = f.read()
                        txt = eval(txt)
                        num.append(txt)
                        label = pd.get_dummies(num)
                        label = label.values.tolist()
                        label = np.asarray(label[-1])
                        image_root = i.replace('Emotion', 'CK_image')
                        image = load_data(image_root)
                        for times in range(6):
                            output_data.append(image[times])
                            output_label.append(label)
    return output_data,output_label


def load_pretrain_emotion_model():
    # load model
    model = load_model('emotion_net_model.h5')
    model = Model(inputs = model.input,outputs = model.output)
    # model.summary()
    for layer in model.layers:
        layer.trainable = True
    # freeze all layer and return all layer output
    return model

def emotion_model():
    model = Sequential()
    model.add(load_pretrain_emotion_model())
    model.add(Conv2D(filters=4, kernel_size=1, strides=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax', name='prediction'))
    model = Model(inputs=model.input, outputs=model.output)
    # for layer in model.layers:
    #     layer.trainable = True
    # model.summary()
    return model

def main():
    folders_name = 'Emotion'
    data, label = load_data_and_label(folders_name)
    train_image, test_image, train_label, test_label = train_test_split(np.array(data), np.array(label), test_size=150,random_state=42)
    # train_image = np.array(data[:len(data)-200])
    # train_label = np.array(label[:len(data)-200])
    # test_image = np.array(data[len(data)-200:])
    # test_label = np.array(label[len(label)-200:])
    model = emotion_model()
    model.summary()
    for layer in model.layers:
        layer.trainable = True
    optimizer = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_image, train_label, verbose=2, batch_size=32, validation_split=0.2,epochs=50)

    loss,acc = model.evaluate(test_image,test_label)
    print('loss:',loss,'accuracy:',acc)

    y_pred = model.predict(test_image)
    y_pred = np.argmax(y_pred, axis=1)
    test_label = np.argmax(test_label,axis=1)
    # print(y_pred,test_label)

    cm = confusion_matrix(test_label, y_pred)
    cm = cm / cm.sum(axis=1)
    sns.heatmap(cm, annot=True)
    plt.title('confuse matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



    plt.plot(history.history['accuracy']) #+ history_2.history['accuracy'])
    plt.plot(history.history['val_accuracy']) # + history_2.history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('train acc')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['loss']) #+ history_2.history['loss'])
    plt.plot(history.history['val_loss']) #+ history_2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('train loss')
    plt.xlabel('epoch')
    plt.show()



main()