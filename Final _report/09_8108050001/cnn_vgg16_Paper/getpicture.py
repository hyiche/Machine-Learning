#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import Audio
import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
from PIL import Image
from matplotlib.pyplot import imshow


# In[3]:


y, sr = librosa.load('D:/mgc/wav_files/0_Techno.wav', sr = 22050) # Use the default sampling rate of 22,050 Hz


# In[6]:


Audio(y, rate=sr)


# In[7]:


plt.figure(figsize=(15,2))
librosa.display.waveplot(y  = y,
                     sr     = sr, 
                     max_sr = 1000, 
                     alpha  = 0.25, 
                     color  = 'blue')


# In[8]:


pre_emphasis = 0.97
y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
plt.figure(figsize=(15,2))
librosa.display.waveplot(y  = y,
                     sr     = sr, 
                     max_sr = 1000, 
                     alpha  = 0.25, 
                     color  = 'red')


# In[9]:


import os
import librosa
from librosa import display
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pylab


# In[16]:


WAV_DIR = 'D:/mgc/wav_files/'
IMG_DIR = 'D:/mgc/spectrogram_images/'
wav_files = os.listdir(WAV_DIR)


# In[18]:


for f in wav_files:
    try:
        # Read wav-file
        y, sr = librosa.load(WAV_DIR+f, sr = 22050) # Use the default sampling rate of 22,050 Hz
        
        # Compute spectrogram
        M = librosa.feature.melspectrogram(y, sr, 
                                           fmax = sr/2, # Maximum frequency to be used on the on the MEL scale
                                           n_fft=2048, 
                                           hop_length=512, 
                                           n_mels = 96, # Set as per the Google Large-scale audio CNN paper
                                           power = 2) # Power = 2 refers to squared amplitude
        
        # Power in DB
        log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
        
        # Plotting the spectrogram
        pylab.figure(figsize=(5,5))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(log_power, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4]+'.jpg', bbox_inches=None, pad_inches=0)
        pylab.close()

    except Exception as e:
        print(f, e)
        pass


# In[14]:


wav_files[1][:-4]


# In[ ]:




