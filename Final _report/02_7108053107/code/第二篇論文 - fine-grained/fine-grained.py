import glob
import numpy as np
import cv2
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model,load_model
from keras.layers import Flatten,Conv2D,Dense,Dropout,MaxPooling2D,BatchNormalization,Reshape,AveragePooling2D
from keras.optimizers import Adam,SGD

from sklearn.model_selection import train_test_split
import keras.losses as loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss

####################################################################################
# import tensorflow as tf
#
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# # 設定 Keras 使用的 TensorFlow Session
# tf.keras.backend.set_session(sess)
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
###################################################################################
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





def main():
    # image = np.array(load_image('CUB_200_2011'))
    # label = load_label()
    # image,label = load_data_and_label('Emotion')
    # print(type(image),type(label))
    # train_image, test_image, train_label, test_label = train_test_split(np.array(image),np.array(label), test_size=100, random_state=42)
    folders_name = 'Emotion'
    data, label = load_data_and_label(folders_name)
    train_image, test_image, train_label, test_label = train_test_split(np.array(data), np.array(label), test_size=150,random_state=42)


    model_vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    model_vgg_output = model_vgg.get_layer('block4_conv3').output
    for layer in model_vgg.layers:
        layer.trainable = True

    # G - stream
    G_Stream = Conv2D(filters=200, kernel_size=(1, 1), strides=(1, 1), input_shape=(28, 28, 512), padding='same',activation='relu', name='conv5_G')(model_vgg_output)
    G_Stream = BatchNormalization()(G_Stream)
    G_Stream = Flatten()(G_Stream)
    G_Stream = Dense(200, activation='relu', name='fc1_Gstream')(G_Stream)
    G_Stream = Dropout(0.1)(G_Stream)
    G_Stream = Dense(8, activation='softmax', name='Gstream')(G_Stream)
    G_Stream_model = Model(inputs=model_vgg.input, outputs=G_Stream)

    # # P - stream
    P_Stream = Conv2D(filters=2000, kernel_size=(1, 1), strides=(1, 1), input_shape=(28, 28, 512), padding='same',activation='relu', name='conv6_P')(model_vgg_output)
    # P_Stream = BatchNormalization()(P_Stream)
    P_Stream_GMP = MaxPooling2D(pool_size = (28,28))(P_Stream)
    P_Stream = Flatten()(P_Stream_GMP)
    P_Stream = Dense(1024, activation='relu', name='fc1_Pstream')(P_Stream)
    P_Stream = Dense(512, activation='relu', name='fc2_Pstream')(P_Stream)
    P_Stream = Dense(256, activation='relu', name='fc3_Pstream')(P_Stream)
    # P_Stream = Dropout(0.5)(P_Stream)
    P_Stream = Dense(8, activation='softmax', name='Pstream')(P_Stream)
    P_Stream_model = Model(inputs=model_vgg.input, outputs=P_Stream)

    # Side - Branch
    side_branch_pool = Reshape((1,250,8))(P_Stream_GMP)
    cross_channel_pool = AveragePooling2D(pool_size = (1,250),strides=1,padding='valid')(side_branch_pool)
    cross_channel_pool = Flatten()(cross_channel_pool)
    Side_Branch_output = Dense(8, activation='softmax', name='side_branch')(cross_channel_pool)
    side_branch_model = Model(inputs=model_vgg.input, outputs=Side_Branch_output)

    # total_output = Add()([G_Stream_model.output,P_Stream_model.output]) #,0.1*side_branch_model.output])

    main_model = Model(inputs=model_vgg.input, outputs=[G_Stream_model.output,P_Stream_model.output,side_branch_model.output],name='main_model')
    main_model.summary()
    # for layer in main_model.layers:
    #     layer.trainable = False

    optimizer = Adam(lr=1e-4,decay=0.000000005)
    main_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy','accuracy','accuracy'],loss_weights=[1,1,0.1])
    history = main_model.fit(train_image,[train_label,train_label,train_label],validation_split=0.1,epochs=50,batch_size=16,verbose=2)
    main_model.save('main_model.h5')


    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(loc='upper left')
    # plt.title('model P-stream acc')
    # plt.ylabel('acc')
    # plt.xlabel('epoch')
    # plt.show()
    #
    plt.plot(history.history['Pstream_accuracy'])
    plt.plot(history.history['val_Pstream_accuracy'])
    plt.legend(loc='upper left')
    plt.title('model acc')
    plt.ylabel('train acc')
    plt.xlabel('epoch')
    plt.show()
    #
    plt.plot(history.history['Gstream_accuracy'])
    plt.plot(history.history['val_Gstream_accuracy'])
    plt.legend(loc='upper left')
    plt.title('model acc')
    plt.ylabel('train acc')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['side_branch_accuracy'])
    plt.plot(history.history['val_side_branch_accuracy'])
    plt.legend(loc='upper left')
    plt.title('model acc')
    plt.ylabel('train acc')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(loc='upper left')
    plt.title('model loss')
    plt.ylabel('train loss')
    plt.xlabel('epoch')
    plt.show()

    # loss, acc = main_model.evaluate(test_image, test_label,verbose=2)
    # print('Loss for test image:',loss,'\n'+'accuracy for test image:',acc)
# def categorical_crossentropy(y_true, y_pred):
#     return loss.categorical_crossentropy(y_true, y_pred)

def predict():
    folders_name = 'Emotion'
    data, label = load_data_and_label(folders_name)
    train_image, test_image, train_label, test_label = train_test_split(np.array(data), np.array(label), test_size=150,random_state=42)

    model = load_model('main_model_Pstream_only.h5')
    # loss, acc = model.evaluate(test_image, test_label,verbose=2)
    # print('Loss for test image:',loss,'\n'+'accuracy for test image:',acc)
    model_g,model_p,model_s = model.predict(test_image)
    print(model_g.shape,model_p.shape,model_s.shape)
    total_output = (model_g + model_p + 0.1*model_s)/2.1
    y_pred = np.argmax(total_output, axis=1)
    y_true = np.argmax(test_label, axis=1)
    print(test_label,'\n',y_pred)
    print('accuracy = ',accuracy_score(y_true,y_pred))
    print('loss = ',log_loss(test_label,total_output))
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1)
    sns.heatmap(cm, annot=True)
    plt.title('confuse matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # print(model_pred[0].shape)

predict()


