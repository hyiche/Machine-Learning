#!/usr/bin/env python
# coding: utf-8

# In[1]:


import youtube_dl
import re
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


# In[2]:


import ffmpeg


# In[3]:


genre_dict = {
            '/m/064t9': 'Pop_music',
            '/m/0glt670': 'Hip_hop_music',
            '/m/06by7': 'Rock_music',
            '/m/06j6l': 'Rhythm_blues',
            '/m/06cqb': 'Reggae',
            '/m/0y4f8': 'Vocal',
            '/m/07gxw': 'Techno',
            }

genre_set = set(genre_dict.keys())


# In[4]:


genre_set


# In[5]:


temp_str = []
with open('D:/mgc/data_files/csv_files/balanced_train_segments.csv', 'r') as f:
    temp_str = f.readlines()


# In[6]:


#Tqdm 是一個快速，可擴充套件的Python進度條，可以在 Python 長迴圈中新增一個進度提示資訊，使用者只需要封裝任意的迭代器 tqdm(iterator)。
data = np.ones(shape=(1,4)) 
for line in tqdm(temp_str):
    line = re.sub('\s?"', '', line.strip())
    elements = line.split(',')
    common_elements = list(genre_set.intersection(elements[3:]))
    if  common_elements != []:
        data = np.vstack([data, np.array(elements[:3]
                                         + [genre_dict[common_elements[0]]]).reshape(1, 4)])

df = pd.DataFrame(data[1:], columns=['url', 'start_time', 'end_time', 'class_label'])


# In[12]:


len(df.class_label)


# In[7]:


# Remove 10k Techno audio clips - to make the data more balanced 先用小樣本下去做 故先不用平衡資料
#np.random.seed(10)
#drop_indices = np.random.choice(df[df['class_label'] == 'Techno'].index, size=10000, replace=False)
#df.drop(labels=drop_indices, axis=0, inplace=True)
#df.reset_index(drop=True, inplace=False)

# Time to INT 
df['start_time'] = df['start_time'].map(lambda x: np.int32(np.float(x)))
df['end_time'] = df['end_time'].map(lambda x: np.int32(np.float(x)))


# In[8]:


df_1 = df.loc[:0,]


# In[138]:


for i, row in tqdm(df.iterrows()):
    url = '"https://www.youtube.com/embed/'+ row['url'] + '"'
    file_name = str(i)+"_"+row['class_label']    
    command_1 = 'youtube-dl -o "D:/mgc/wavf_files/' + file_name+'.wav" -f 140 ' + url+' -x --audio-format wav'    
    os.system(command_1)


# In[9]:


for i, row in tqdm(df.iterrows()):
    url = '"https://www.youtube.com/embed/'+ row['url'] + '"'
    file_name = str(i)+"_"+row['class_label']   
    command_2 = 'ffmpeg -ss 30 -i "D:/mgc/wavf_files/' + file_name+'.wav" -t 10 -vn -acodec pcm_s16le -ar 44100 -ac 1 '+                     '"D:/mgc/wav_files/'+file_name+'.wav"'    
    os.system(command_2)


# In[140]:


df


# In[ ]:




