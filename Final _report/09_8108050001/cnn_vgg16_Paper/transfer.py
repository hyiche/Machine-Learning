#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings   # 消除警告
warnings.filterwarnings("ignore")


# In[29]:


import tensorflow as tf


# In[30]:


import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import Audio
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras import models


# In[31]:


IMG_DIR = 'D:/mgc/spectrogram_images/'
IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 7
NUM_EPOCHS = 10
BATCH_SIZE = 32
L2_LAMBDA = 0.001


# In[32]:


label_dict = {'Hip':0,
              'Pop':1,
              'Vocal':2,
              'Rhythm':3,
              'Reggae':4,
              'Rock':5,
              'Techno':6,
             }


# In[33]:


one_hot = OneHotEncoder(n_values=NUM_CLASSES)


# In[35]:


all_files = os.listdir(IMG_DIR)


# In[247]:


label_array = []
for file_ in all_files:
    vals = file_[:-4].split('_')
    label_array.append(label_dict[vals[1]])


# In[248]:


train_files, test_files, train_labels, test_labels = train_test_split(all_files, 
                                                                      label_array,
                                                                      random_state = 8, 
                                                                      test_size = 0.2
                                                                     )


# In[249]:


test_files,val_files, test_labels,  val_labels = train_test_split(test_files, test_labels,
                                                                  random_state = 6, 
                                                                  test_size = 0.5
                                                                 )


# In[250]:


conv_base = tf.keras.applications.VGG16(include_top = False, 
                                            weights = 'imagenet', 
                                            input_shape = (216, 216, 3) # 3 channels - RGB
                                           ) 
# The weights are for the CONV filters - hence you can pass any pre-set image size to this VGG network
# Need not be 224 x 224 x 3 (Although does it work better for 224 size? Need to check)


# In[251]:


conv_base


# In[252]:


conv_base.summary()


# In[253]:


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten()) # Flatten output and send it to MLP

# 1-layer MLP with Dropout, BN 
model.add(layers.Dense(512, name='dense_1', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
model.add(layers.Dropout(rate=0.3, name='dropout_1')) # Can try varying dropout rates
model.add(layers.Activation(activation='relu', name='activation_1'))

model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='dense_output'))
model.summary()


# In[254]:


# Set the convolution base to be not trainable
conv_base.trainable = False
model.summary()


# In[255]:


def load_batch(file_list):
    img_array = []
    idx_array = []
    label_array = []

    for file_ in file_list:
        im = Image.open(IMG_DIR + file_)
        im = im.resize((216, 216), Image.ANTIALIAS)
        img_array.append(np.array(im))

        vals = file_[:-4].split('_')
        idx_array.append(vals[0])
        label_array.append([label_dict[vals[1]]])

    label_array = one_hot.fit_transform(label_array).toarray()
    img_array = np.array(img_array)/255.0 # Normalize RGB
    
    return img_array, np.array(label_array), np.array(idx_array)


# In[256]:


def batch_generator(files, BATCH_SIZE):
    L = len(files)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < L:
            
            limit = min(batch_end, L)
            file_list = files[batch_start: limit]
            batch_img_array, batch_label_array, batch_idx_array = load_batch(file_list)

            yield (batch_img_array, batch_label_array) # a tuple with two numpy arrays with batch_size samples     

            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE


# In[257]:


optimizer = optimizers.Adam(lr=1e-5)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[258]:


STEPS_PER_EPOCH = len(train_files)//BATCH_SIZE


# 所以，如果列表元素可以按照某種演算法推算出來，那我們是否可以在迴圈的過程中不斷推算出後續的元素呢？這樣就不必建立完整的list，從而節省大量的空間。在Python中，這種一邊迴圈一邊計算的機制，稱為生成器：generator。

# In[259]:


history = model.fit_generator(generator = batch_generator(train_files, BATCH_SIZE),
                              epochs     = 10,
                              steps_per_epoch = STEPS_PER_EPOCH,
                              validation_data = batch_generator(val_files, BATCH_SIZE), 
                              validation_steps = 1)


# In[261]:


history.history.keys()
test1 = pd.DataFrame(history.history, index=range(1,11))


# In[262]:


#test1.to_csv('D:/VGG16.CSV')


# In[263]:


test1 = pd.read_csv('D:/VGG16.CSV')
test1


# In[264]:


plt.xticks(range(1,11))
plt.plot(test1['loss'], marker='o', label='training_loss')
plt.plot(test1['val_loss'], marker='d', label='validation_loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()


# In[265]:


plt.xticks(range(1,11))
plt.plot(test1['categorical_accuracy'], marker='o', label='training_accuracy')
plt.plot(test1['val_categorical_accuracy'], marker='d', label='validation_accuracy')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()


# In[274]:


history = model.fit_generator(generator = batch_generator(train_files, BATCH_SIZE),
                              epochs     = 4,
                              steps_per_epoch = STEPS_PER_EPOCH,
                              validation_data = batch_generator(val_files, BATCH_SIZE), 
                              validation_steps = 1)


# In[290]:


history.history.keys()
test1 = pd.DataFrame(history.history)
test1


# In[283]:


#TEST_STEPS = len(test_files)//BATCH_SIZE

pred_probs = model.predict_generator(generator = batch_generator(test_files, BATCH_SIZE), steps=2)
pred = np.argmax(pred_probs, axis=-1)


# In[284]:


mi = dict(zip(label_dict.values(), label_dict.keys()))


# In[285]:


test_labels_name = []
for i in range(0,len(test_labels),1):
    test_labels_name.append(mi[test_labels[i]])


# In[286]:


pred_name = []
for i in range(0,len(pred),1):
    pred_name.append(mi[pred[i]])


# In[287]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp
import itertools
from itertools import cycle    
def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels


# In[288]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_name, pred_name,labels=['Hip', 'Pop', 'Vocal', 'Rhythm', 'Reggae', 'Rock', 'Techno'])
cm
import seaborn as sns
import matplotlib.pyplot as plt     
plt.figure(figsize = (10,8))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Hip', 'Pop', 'Vocal', 'Rhythm', 'Reggae', 'Rock', 'Techno']); ax.yaxis.set_ticklabels(['Hip', 'Pop', 'Vocal', 'Rhythm', 'Reggae', 'Rock', 'Techno'])


# In[289]:


print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_true=test_labels[:len(pred)], y_pred=pred)))
print('Test Set F-score =  {0:.2f}'.format(f1_score(y_true=test_labels[:len(pred)], y_pred=pred, average='macro')))


# In[ ]:




