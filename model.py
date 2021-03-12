# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:11:03 2021

@author: adelv
"""
import tensorflow as tf
from tensorflow.keras.layers import *

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np




data_path = r'C:\Users\adelv\Documents\Hackaton\hackathon_kpis_anonymised\hackathon_kpis_anonymised.csv'
df = pd.read_csv(data_path)
df = df.dropna()
df = df.values


df = df[:,2:]

#%%

def build_model(input_shape):
    
    #encoder
    conv1 = Conv1D(16,3,activation='relu',padding='same')((input_shape))
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = conv1 = Conv1D(32,3,activation='relu',padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(64,3,activation='relu',padding='same')(pool2)
    
    #decoder
    conv4 = Conv1D(64,3,activation='relu',padding='same')(conv3)
    up1 = UpSampling1D(2)(conv4)
    conv5 = Conv1D(32,3,activation='relu',padding='same')(up1)
    up2 = UpSampling1D(2)(conv5)
    out = Conv1D(14,3,activation='sigmoid',padding='same')(up2)
    
    return out



#%%



window_size = 64



a = df.shape[0] % window_size
xtrain = df[:-a,:]
xtrain = xtrain.reshape(xtrain.shape[0]//window_size,window_size,xtrain.shape[1])
xtrain = xtrain.astype(np.single)

#shuffler treningsdata
np.random.shuffle(xtrain)


#beholder en liten andel for testing
test_split = 0.1
xtest = xtrain[:int(xtrain.shape[0] * test_split)]

#resterende brukes til trening
xtrain = xtrain[int(xtrain.shape[0] * test_split):]



input_shape = tf.keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]))
model = Model(input_shape,build_model(input_shape))
model.compile(loss='mse',optimizer='adam')
model.summary()

EPOCHS = 100

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(xtrain,xtrain,validation_split=.1,epochs=EPOCHS,callbacks=[callback])


#%%
pred = model.predict(xtest[0].reshape(1,xtest[0].shape[0],xtest[0].shape[1]))

plt.plot(pred[0,:,0],label="pred")
plt.show()
plt.plot(xtest[0][:,0],label="ground truth")
plt.show()

plt.legend()





















