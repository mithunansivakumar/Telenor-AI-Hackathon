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




data_path = r'hackathon_kpis_anonymised.csv'
df = pd.read_csv(data_path,sep=",")

df.sort_values(['cell_name', 'timestamp'], ascending=[True, False],inplace=True)
print(df.columns)

df2 = df
df = df.dropna()
df2 = df2.drop(df.index)
df2 = df2.fillna(-1)

df2=df2.values[:,2:]


df = df.values
df = df[:,2:]



def build_model(input_shape,ksize=3):
    
    #encoder
    conv1 = Conv1D(14,ksize,activation='relu',padding='same')(input_shape)
    pool1 = MaxPooling1D(pool_size=1)(conv1)
    conv2 = conv1 = Conv1D(32,ksize,activation='relu',padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=1)(conv2)
    conv3 = Conv1D(64,ksize,activation='relu',padding='same')(pool2)
    
    #decoder
    conv4 = Conv1D(64,ksize,activation='relu',padding='same')(conv3)
    up1 = UpSampling1D(1)(conv4)
    conv5 = Conv1D(32,ksize,activation='relu',padding='same')(up1)
    up2 = UpSampling1D(1)(conv5)
    out = Conv1D(14,ksize,activation='sigmoid',padding='same')(up2)
    
    return out


def simple_autoencoder(input_shape):
    #encoder
    layer1 = Dense(14,activation='relu')(input_shape)
    layer2 = Dense(7,activation='relu')(layer1)
    layer3 = Dense(7,activation='relu')(layer2)
    output = Dense(14,activation='relu')(layer3)
    return output

def simple_conv_autoencoder(input_shape,ksize=3):
    #encoder
    layer1 = Conv1D(14,ksize=3,activation='relu')(input_shape)
    layer2 = Conv1D(28,ksize=3,activation='relu')(layer1)
    layer3 = Conv1D(28,ksize=3,activation='relu')(layer2)
    output = Dense(14,ksize=3,activation='relu')(layer3)
    
    
    return output
    
    
def unet(input_shape,pretrained_weights = None,ksize=3):

    conv1 = Conv1D(14, ksize, activation='relu', padding='same')(input_shape)
    conv1 = Conv1D(14, ksize, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(32, ksize, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(32, ksize, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(64, ksize, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(64, ksize, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(128, ksize, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(128, ksize, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling1D(size=2)(conv4), conv3], axis=-1)
    conv5 = Conv1D(64, ksize, activation='relu', padding='same')(up5)
    conv5 = Conv1D(64, ksize, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling1D(size=2)(conv5), conv2], axis=-1)
    conv6 = Conv1D(32, ksize, activation='relu', padding='same')(up6)
    conv6 = Conv1D(32, ksize, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling1D(size=2)(conv6), conv1], axis=-1)
    conv7 = Conv1D(14, ksize, activation='relu', padding='same')(up7)
    conv7 = Conv1D(14, ksize, activation='relu', padding='same')(conv7)

    conv8 = Conv1D(14,1, activation='sigmoid')(conv7)

    return conv8




window_size=32

#sjekk at antall rader går opp i row-context
#assert df.shape[0] % window_size == 0
a = df.shape[0] % window_size
xtrain = df[:-a]


xtrain = xtrain.astype(np.single)
xtrain = xtrain.reshape(xtrain.shape[0] // window_size, window_size, xtrain.shape[-1])


#shuffler treningsdata
np.random.shuffle(xtrain)


#beholder en liten andel for testing
test_split = 0.1
xtest = xtrain[:int(xtrain.shape[0] * test_split)]

#resterende brukes til trening
xtrain = xtrain[int(xtrain.shape[0] * test_split):]



input_shape = tf.keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]))


model = Model(input_shape, unet(input_shape,ksize=1) )


model.compile(loss='mse',optimizer='adam')
model.summary()

EPOCHS = 100

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
model.fit(xtrain,xtrain,validation_split=.1,epochs=EPOCHS,callbacks=[callback])



#%%
b = df2.shape[0] % window_size
df2 = df2.astype(np.single)
df2 = df2[:-b]

df2 = df2.reshape(df2.shape[0]//window_size,window_size,df2.shape[-1])
preds = model.predict(df2)


#preds = preds.reshape(preds.shape[0],preds.shape[-1])

test = preds[:,1,:]
x,y = np.where(df2[:,1,:] == - 1)
out = df2[:,1,:][x,y]
out = test[x,y]

np.savetxt('predicted_nans.csv', df2[:,1,:], delimiter=',')








#%%
num_to_plot = 8


predictions = model.predict(df2[:num_to_plot])

fig,axs = plt.subplots(2,num_to_plot//2,sharey=True,figsize=(10,10))

for ind,ax in enumerate(axs.flat):
    ax.plot(predictions[ind,:,2],label="pred")
    ax.plot(df2[ind,:,2],label="gt")
    
    



plt.legend()





















