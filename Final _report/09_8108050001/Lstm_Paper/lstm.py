#!/usr/bin/env python
# coding: utf-8

# In[32]:


from IPython.display import Audio
import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
from PIL import Image
from matplotlib.pyplot import imshow
import os
import librosa
from librosa import display
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pylab
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
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt    
from tensorflow.keras import layers


# In[33]:


WAV_DIR = 'D:/mgc2/'
wav_files = os.listdir(WAV_DIR)


# In[34]:


label_dict = {'blues':0,
              'classical':1,
              'country':2,
              'disco':3,
              'hiphop':4,
              'jazz':5,
              'metal':6,
              'pop':7,
              'reggae':8,
              'rock':9
             }


# In[35]:


one_hot = OneHotEncoder(n_values=10)


# In[36]:


label_array = []

y, sr = librosa.load(WAV_DIR+wav_files[0], sr = 22050)
M= librosa.feature.mfcc(y, sr,n_mfcc=13,hop_length=512)
mfcc_array = M.T.reshape(1,1293,13)
label_array.append([label_dict[wav_files[0][:-10]]])    

for f in wav_files[1:]:
    try:
        y, sr = librosa.load(WAV_DIR+f, sr = 22050)
        M = librosa.feature.mfcc(y, sr,n_mfcc=13,hop_length=512)
        M = M.T.reshape(1,1293,13)
        mfcc_array = np.concatenate((mfcc_array , M))
        vals = f[:-10]
        label_array.append([label_dict[vals]])
    except:
        print(f)
        pass
label_array_one = one_hot.fit_transform(label_array).toarray()   


# In[41]:


label_array_one.shape


# In[42]:


mfcc_array.shape,label_array_one.shape


# In[71]:


train_mfcc, test_mfcc, train_labels, test_labels = train_test_split(mfcc_array, 
                                                                      label_array_one,
                                                                      random_state = 10, 
                                                                      test_size = 0.1
                                                                     )


# In[72]:


val_mfcc, test_mfcc, val_labels, test_labels = train_test_split(test_mfcc, test_labels,
                                                                  random_state = 10, 
                                                                  test_size = 0.5
                                                                 )


# In[322]:





# In[13]:


#x = np.load('D:/data_validation_input.npy')
#y= np.load('D:/data_validation_target.npy')


# In[47]:


input_shape = (train_mfcc.shape[1], train_mfcc.shape[2])
input_shape


# In[98]:


#input_shape = (1293, 13)
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(units = 128,return_sequences = True,input_shape= input_shape,name='d1'))
model.add(layers.LSTM(units = 32,activation = 'tanh' ,return_sequences = False ,name='d2'))
# Add a Dense layer with 10 units and softmax activation.
model.add(layers.Dense(train_labels.shape[1], activation='softmax',name='o1'))
model.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(), 
                   metrics=['accuracy'])
model.summary()


# In[ ]:





# In[100]:


num_epochs = 10
batch_size =35
history = model.fit(train_mfcc,train_labels, epochs=num_epochs,
    batch_size=batch_size,validation_data =(val_mfcc,val_labels) )


# In[86]:





# In[101]:


history.history.keys()
test1 = pd.DataFrame(history.history, index=range(1,11))


# In[103]:


test1


# In[104]:


plt.xticks(range(1,11))
plt.plot(test1['loss'], marker='o', label='training_loss')
plt.plot(test1['val_loss'], marker='d', label='validation_loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()


# In[105]:


plt.xticks(range(1,11))
plt.plot(test1['accuracy'], marker='o', label='training_accuracy')
plt.plot(test1['val_accuracy'], marker='d', label='validation_accuracy')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()


# In[106]:


pred_probs = model.predict(test_mfcc, steps=2)
pred = np.argmax(pred_probs, axis=-1)


# In[111]:


test_labels
test_labels_new = [np.where(r==1)[0][0] for r in test_labels]


# In[112]:


mi = dict(zip(label_dict.values(), label_dict.keys()))
test_labels_name = []
for i in range(0,len(test_labels_new),1):
    test_labels_name.append(mi[test_labels_new[i]])
pred_name = []
for i in range(0,len(pred),1):
    pred_name.append(mi[pred[i]])    


# In[115]:


label_dict.keys()


# In[116]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_name, pred_name,labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
cm


# In[117]:



plt.figure(figsize = (10,8))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']); ax.yaxis.set_ticklabels(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])


# In[333]:


print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_true=test_labels_new[:len(pred)], y_pred=pred)))
print('Test Set F-score =  {0:.2f}'.format(f1_score(y_true=test_labels_new[:len(pred)], y_pred=pred, average='macro')))


# In[151]:


#mfcc_array[1]                                                                      
label_array
c = {"a":label_array}
c = pd.DataFrame(c)
c.a.value_counts()


# In[331]:


len(label_8)


# In[165]:


label_0 = label_array[0:100]
label_1 = label_array[100:192]
label_2 = label_array[192:283]
label_3 = label_array[283:375]#3
label_4 = label_array[375:457]#4
label_5 = label_array[457:549]#5
label_6 = label_array[549:649]#6
label_7 = label_array[649:749]#7
label_8 = label_array[749:849]#8
label_9 = label_array[849:944]#9
mfcc_0 = mfcc_array[0:100]
mfcc_1 = mfcc_array[100:192]
mfcc_2 = mfcc_array[192:283]
mfcc_3 = mfcc_array[283:375]#3
mfcc_4 = mfcc_array[375:457]#4
mfcc_5 = mfcc_array[457:549]#5
mfcc_6 = mfcc_array[549:649]#6
mfcc_7 = mfcc_array[649:749]#7
mfcc_8 = mfcc_array[749:849]#8
mfcc_9 = mfcc_array[849:944]#9



# In[167]:


def get_v_t_t(data):
    return data[0:15],data[15:30],data[30:]


# In[170]:


mfcc_0_val,mfcc_0_test,mfcc_0_train = get_v_t_t(mfcc_0)
mfcc_1_val,mfcc_1_test,mfcc_1_train = get_v_t_t(mfcc_1)
mfcc_2_val,mfcc_2_test,mfcc_2_train = get_v_t_t(mfcc_2)
mfcc_3_val,mfcc_3_test,mfcc_3_train = get_v_t_t(mfcc_3)
mfcc_4_val,mfcc_4_test,mfcc_4_train = get_v_t_t(mfcc_4)
mfcc_5_val,mfcc_5_test,mfcc_5_train = get_v_t_t(mfcc_5)
mfcc_6_val,mfcc_6_test,mfcc_6_train = get_v_t_t(mfcc_6)
mfcc_7_val,mfcc_7_test,mfcc_7_train = get_v_t_t(mfcc_7)
mfcc_8_val,mfcc_8_test,mfcc_8_train = get_v_t_t(mfcc_8)
mfcc_9_val,mfcc_9_test,mfcc_9_train = get_v_t_t(mfcc_9)
label_0_val,label_0_test,label_0_train = get_v_t_t(label_0)
label_1_val,label_1_test,label_1_train = get_v_t_t(label_1)
label_2_val,label_2_test,label_2_train = get_v_t_t(label_2)
label_3_val,label_3_test,label_3_train = get_v_t_t(label_3)
label_4_val,label_4_test,label_4_train = get_v_t_t(label_4)
label_5_val,label_5_test,label_5_train = get_v_t_t(label_5)
label_6_val,label_6_test,label_6_train = get_v_t_t(label_6)
label_7_val,label_7_test,label_7_train = get_v_t_t(label_7)
label_8_val,label_8_test,label_8_train = get_v_t_t(label_8)
label_9_val,label_9_test,label_9_train = get_v_t_t(label_9)


# In[ ]:


###lstm1


# In[196]:


strong_train =  np.concatenate((mfcc_4_train,mfcc_6_train,mfcc_7_train,mfcc_9_train,mfcc_8_train))
mild_train = np.concatenate((mfcc_5_train,mfcc_3_train,mfcc_2_train,mfcc_1_train,mfcc_0_train))
lstm1_label_train = [[0]]*len(strong_train)+[[1]]*len(mild_train)
one_hot = OneHotEncoder(n_values=2)
lstm1_label_one_train = one_hot.fit_transform(lstm1_label_train).toarray()   
lstm1_train = np.concatenate((strong_train,mild_train))

strong_val =  np.concatenate((mfcc_4_val,mfcc_6_val,mfcc_7_val,mfcc_9_val,mfcc_8_val))
mild_val = np.concatenate((mfcc_5_val,mfcc_3_val,mfcc_2_val,mfcc_1_val,mfcc_0_val))
lstm1_label_val = [[0]]*len(strong_val)+[[1]]*len(mild_val)
one_hot = OneHotEncoder(n_values=2)
lstm1_label_one_val = one_hot.fit_transform(lstm1_label_val).toarray()
lstm1_val = np.concatenate((strong_val,mild_val))


# In[214]:


input_shape = (1293, 13)
model_lstm1 = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# Add a LSTM layer with 128 internal units.
model_lstm1.add(layers.LSTM(units = 128,return_sequences = True,input_shape= input_shape,name='d1'))
model_lstm1.add(layers.LSTM(units = 32,activation = 'tanh' ,return_sequences = False ,name='d2'))
# Add a Dense layer with 10 units and softmax activation.
model_lstm1.add(layers.Dense(lstm1_label_one_train.shape[1], activation='softmax',name='o1'))
model_lstm1.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(), 
                   metrics=['accuracy'])
model_lstm1.summary()


# In[215]:


num_epochs = 5
batch_size =35
history = model_lstm1.fit(lstm1_train,lstm1_label_one_train, epochs=num_epochs,
    batch_size=batch_size,validation_data =(lstm1_val,lstm1_label_one_val) )


# In[206]:


strong_label_train = [[0]]*len(mfcc_4_train)+[[1]]*len(mfcc_6_train)+[[2]]*len(mfcc_7_train)+[[3]]*len(mfcc_8_train)+[[4]]*len(mfcc_9_train)
one_hot = OneHotEncoder(n_values=5)
strong_label_one_train = one_hot.fit_transform(strong_label_train).toarray()   

strong_label_val = [[0]]*len(mfcc_4_val)+[[1]]*len(mfcc_6_val)+[[2]]*len(mfcc_7_val)+[[3]]*len(mfcc_8_val)+[[4]]*len(mfcc_9_val)
one_hot = OneHotEncoder(n_values=5)
strong_label_one_val = one_hot.fit_transform(strong_label_val).toarray()   


# In[216]:


input_shape = (1293, 13)
model_strong = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# Add a LSTM layer with 128 internal units.
model_strong.add(layers.LSTM(units = 128,return_sequences = True,input_shape= input_shape,name='d1'))
model_strong.add(layers.LSTM(units = 32,activation = 'tanh' ,return_sequences = False ,name='d2'))
# Add a Dense layer with 10 units and softmax activation.
model_strong.add(layers.Dense(strong_label_one_train.shape[1], activation='softmax',name='o1'))
model_strong.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(), 
                   metrics=['accuracy'])
model_strong.summary()


# In[217]:


num_epochs = 5
batch_size =35
history = model_strong.fit(strong_train,strong_label_one_train, epochs=num_epochs,
    batch_size=batch_size,validation_data =(strong_val,strong_label_one_val) )


# In[211]:


mild_label_train = [[0]]*len(mfcc_5_train)+[[1]]*len(mfcc_3_train)+[[2]]*len(mfcc_2_train)+[[3]]*len(mfcc_1_train)+[[4]]*len(mfcc_0_train)
one_hot = OneHotEncoder(n_values=5)
mild_label_one_train = one_hot.fit_transform(mild_label_train).toarray()   
mild_label_val = [[0]]*len(mfcc_5_val)+[[1]]*len(mfcc_3_val)+[[2]]*len(mfcc_2_val)+[[3]]*len(mfcc_1_val)+[[4]]*len(mfcc_0_val)
one_hot = OneHotEncoder(n_values=5)
mild_label_one_val = one_hot.fit_transform(mild_label_val).toarray()   


# In[218]:


input_shape = (1293, 13)
model_mild = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
# Add a LSTM layer with 128 internal units.
model_mild.add(layers.LSTM(units = 128,return_sequences = True,input_shape= input_shape,name='d1'))
model_mild.add(layers.LSTM(units = 32,activation = 'tanh' ,return_sequences = False ,name='d2'))
# Add a Dense layer with 10 units and softmax activation.
model_mild.add(layers.Dense(mild_label_one_train.shape[1], activation='softmax',name='o1'))
model_mild.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(), 
                   metrics=['accuracy'])
model_strong.summary()


# In[219]:


num_epochs = 5
batch_size =35
history = model_mild.fit(mild_train,mild_label_one_train, epochs=num_epochs,
    batch_size=batch_size,validation_data =(mild_val,mild_label_one_val) )


# In[220]:


strong_test =  np.concatenate((mfcc_4_test,mfcc_6_test,mfcc_7_test,mfcc_9_test,mfcc_8_test))
mild_test = np.concatenate((mfcc_5_test,mfcc_3_test,mfcc_2_test,mfcc_1_test,mfcc_0_test))
lstm1_label_test = [[0]]*len(strong_test)+[[1]]*len(mild_test)
one_hot = OneHotEncoder(n_values=2)
lstm1_label_one_test = one_hot.fit_transform(lstm1_label_test).toarray()   
lstm1_test = np.concatenate((strong_test,mild_test))


# In[221]:


pred_probs_lstm1 = model_lstm1.predict(lstm1_test, steps=2)
pred_lstm1 = np.argmax(pred_probs_lstm1, axis=-1)


# In[224]:


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(y_true=lstm1_label_test[:len(pred_lstm1)], y_pred=pred_lstm1)))
print('Test Set F-score =  {0:.2f}'.format(f1_score(y_true=lstm1_label_test[:len(pred_lstm1)], y_pred=pred_lstm1, average='macro')))


# In[259]:


#toknow ={"predlstm1":a,"real_lstm1":d,"real_labels":c}
#toknow = pd.DataFrame(toknow)
#toknow['real_s_m_labels']=[0]*15+[1]*15+[2]*15+[3]*15+[4]*15+[0]*15+[1]*15+[2]*15+[3]*15+[4]*15
for_strong_test = lstm1_test[toknow.predlstm1==0]
for_mild_test = lstm1_test[toknow.predlstm1==1]


# toknow.real_lstm1[toknow.predlstm1==0].index

# In[270]:


strong_label_test = [[0]]*len(mfcc_4_test)+[[1]]*len(mfcc_6_test)+[[2]]*len(mfcc_7_test)+[[3]]*len(mfcc_8_test)+[[4]]*len(mfcc_9_test)
one_hot = OneHotEncoder(n_values=5)
strong_label_one_test = one_hot.fit_transform(strong_label_test).toarray()   
mild_label_test = [[0]]*len(mfcc_5_test)+[[1]]*len(mfcc_3_test)+[[2]]*len(mfcc_2_test)+[[3]]*len(mfcc_1_test)+[[4]]*len(mfcc_0_test)
one_hot = OneHotEncoder(n_values=5)
mild_label_one_test = one_hot.fit_transform(mild_label_test).toarray()


# In[265]:


pred_probs_strong = model_strong.predict(for_strong_test, steps=1)
pred_strong = np.argmax(pred_probs_strong, axis=-1)


# In[271]:


pred_probs_mild = model_mild.predict(for_mild_test, steps=1)
pred_mild = np.argmax(pred_probs_mild, axis=-1)


# In[272]:


len(pred_strong),len(pred_mild)


# In[298]:


toknow['new_sm']=toknow.pred_s_m_labels-toknow.real_s_m_labels
toknow['right']=toknow.new_lstm-toknow.new_sm


# In[301]:


len(toknow[toknow.right==0])/150


# In[307]:


len(toknow[0:15][toknow[0:15]['right']==0]),
len(toknow[15:30][toknow[15:30]['right']==0])
len(toknow[30:45][toknow[30:45]['right']==0])
len(toknow[45:60][toknow[45:60]['right']==0])
len(toknow[60:75][toknow[60:75]['right']==0])
len(toknow[75:90][toknow[75:90]['right']==0])
len(toknow[90:105][toknow[90:105]['right']==0])
len(toknow[105:120][toknow[105:120]['right']==0])
len(toknow[120:135][toknow[120:135]['right']==0])
len(toknow[135:150][toknow[135:150]['right']==0])


# In[308]:


len(toknow[0:15][toknow[0:15]['right']==0])+len(toknow[15:30][toknow[15:30]['right']==0])+len(toknow[45:60][toknow[45:60]['right']==0])



# In[312]:


18/45


# In[309]:


len(toknow[30:45][toknow[30:45]['right']==0])+len(toknow[60:75][toknow[60:75]['right']==0])


# In[313]:


14/30


# In[310]:


len(toknow[90:105][toknow[90:105]['right']==0])+len(toknow[105:120][toknow[105:120]['right']==0])


# In[315]:


5/30


# In[311]:


len(toknow[75:90][toknow[75:90]['right']==0])+len(toknow[120:135][toknow[120:135]['right']==0])+len(toknow[135:150][toknow[135:150]['right']==0])


# In[316]:


9/45


# In[318]:


(0.4+0.4666666666666667+0.16666666666666666+0.2)/4

