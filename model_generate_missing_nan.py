# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:11:03 2021

@author: adelv
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np




data_path = r'C:\Users\adelv\Documents\Hackaton\hackathon_kpis_anonymised\hackathon_kpis_anonymised.csv'
df = pd.read_csv(data_path)
df2 = df
df = df.dropna()
df2 = df2.drop(df.index)
df2 = df2.fillna(-1)

df2=df2.values[:,2:]


df = df.values
df = df[:,2:]

#%%

def build_model(input_shape,ksize=3):
    
    #encoder
    conv1 = Conv1D(16,ksize,activation='relu',padding='same')((input_shape))
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = conv1 = Conv1D(32,ksize,activation='relu',padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(64,ksize,activation='relu',padding='same')(pool2)
    
    #decoder
    conv4 = Conv1D(64,ksize,activation='relu',padding='same')(conv3)
    up1 = UpSampling1D(2)(conv4)
    conv5 = Conv1D(32,ksize,activation='relu',padding='same')(up1)
    up2 = UpSampling1D(2)(conv5)
    out = Conv1D(14,ksize,activation='sigmoid',padding='same')(up2)
    
    return out


def simple_autoencoder(input_shape):
    #encoder
    layer1 = Dense(14,activation='relu')(input_shape)
    layer2 = Dense(7,activation='relu')(layer1)
    layer3 = Dense(7,activation='relu')(layer2)
    output = Dense(14,activation='relu')(layer3)
    
    return output
    
    
    




#%%
df = df.astype(np.single)
xtrain=df
xtrain = xtrain.reshape(xtrain.shape[0],1,xtrain.shape[-1])
#%%
#shuffler treningsdata
np.random.shuffle(xtrain)


#beholder en liten andel for testing
test_split = 0.1
xtest = xtrain[:int(xtrain.shape[0] * test_split)]

#resterende brukes til trening
xtrain = xtrain[int(xtrain.shape[0] * test_split):]



input_shape = tf.keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]))


model = Model(input_shape,simple_autoencoder(input_shape))
model.compile(loss='mse',optimizer='adam')
model.summary()

EPOCHS = 100

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
model.fit(xtrain,xtrain,validation_split=.1,epochs=EPOCHS,callbacks=[callback])
#%%

df2 = df2.astype(np.single)
df2 = df2.reshape(df2.shape[0],1,df2.shape[-1])
preds = model.predict(df2)


preds = preds.reshape(preds.shape[0],preds.shape[-1])

np.savetxt('predicted_nans.csv', preds, delimiter=',')




#%%
df2 = df2.astype(np.single)
df2=df2.reshape(df2.shape[0],1,df2.shape[-1])

num_to_plot = 8

predictions = model.predict(df2[:num_to_plot])

fig,axs = plt.subplots(2,num_to_plot//2,sharey=True,figsize=(10,10))

for ind,ax in enumerate(axs.flat):
    ax.plot(predictions[ind,0,:],label="pred")
    ax.plot(df2[ind,0,:],label="gt")
    
    



plt.legend()





















