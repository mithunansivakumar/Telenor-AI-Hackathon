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
import os




#leser inn predikerte nan-verdier
data_path_pred = r"predicted_nans.csv"
df = pd.read_csv(data_path_pred)
df = df.values



#%%

def build_model(input_shape,ksize=3):
    
    #encoder
    conv1 = Conv1D(input_shape.shape[-1],ksize,activation='relu',padding='same')((input_shape))
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = conv1 = Conv1D(64,ksize,activation='relu',padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(128,ksize,activation='relu',padding='same')(pool2)
    
    #decoder
    conv4 = Conv1D(128,ksize,activation='relu',padding='same')(conv3)
    up1 = UpSampling1D(2)(conv4)
    conv5 = Conv1D(64,ksize,activation='relu',padding='same')(up1)
    up2 = UpSampling1D(2)(conv5)
    out = Conv1D(input_shape.shape[-1],ksize,activation='sigmoid',padding='same')(up2)
    
    return out


def unet(input_shape,pretrained_weights = None,ksize=3):

    conv1 = Conv1D(input_shape.shape[-1], ksize, activation='relu', padding='same')(input_shape)
    conv1 = Conv1D(input_shape.shape[-1], ksize, activation='relu', padding='same')(conv1)
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
    conv7 = Conv1D(input_shape.shape[-1], ksize, activation='relu', padding='same')(up7)
    conv7 = Conv1D(input_shape.shape[-1], ksize, activation='relu', padding='same')(conv7)

    conv8 = Conv1D(input_shape.shape[-1],1, activation='sigmoid')(conv7)

    return conv8




#definerer st??rrelse p?? vinduet. 
#Valgt slik at resten blir null ved divisjon og vi beholder mest mulig data og f??r viss st??rrelse p??
#tids-contexten.
window_size = 32
a = df.shape[0] % window_size
xtrain = df[:-a]

#reshaper input til modellen
xtrain = xtrain.reshape(xtrain.shape[0]//window_size,window_size,xtrain.shape[1])
xtrain = xtrain.astype(np.single) #caster til konsistent datatype

#sparer orginalen
original_data = xtrain


#shuffler hele datasettet
np.random.shuffle(xtrain)


#beholder en liten andel for testing
test_split = 0.3
xtest = xtrain[:int(xtrain.shape[0] * test_split)]

#resterende brukes til trening
xtrain = xtrain[int(xtrain.shape[0] * test_split):]


#definerer input layer til modellen (None,27,14) i dette tilfellet
input_shape = tf.keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]))


model = Model(input_shape,unet(input_shape,ksize=9))
model.compile(loss='mse',optimizer='adam')
model.summary()

EPOCHS = 100
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(xtrain,xtrain,validation_split=.3,epochs=EPOCHS,callbacks=[callback])
model.save("autoencoder_model.h5")


#%%
#HERIFRA OG NED B??R INN I EGNE FUNKJSNOER

#plot training history
plt.plot(history.history['val_loss'],label="validation_loss")
plt.plot(history.history['loss'],label="training_loss")
plt.legend()

#%%



#plot a single feature

#henter labels fra dataframen
labels = df_orig.columns[2:]

#antallet du vil plotte (m?? v??re partall)
num_to_plot = 8


predictions = model.predict(xtest[:num_to_plot])

fig,axs = plt.subplots(2,num_to_plot//2,sharey=True)
feature = 0

for ind,ax in enumerate(axs.flat):
    ax.plot(predictions[ind,:,feature],label="pred")
    ax.plot(xtest[ind,:,feature],label="gt")
    ax.legend()
    
    
fig.canvas.set_window_title(labels[feature])

#%%
#plot SE over one feature
labels = df_orig.columns[2:]

num_to_plot = 8

predictions = model.predict(xtest[:num_to_plot])

fig,axs = plt.subplots(2,num_to_plot//2,sharey=True)
feature = 0

for ind,ax in enumerate(axs.flat):
    ax.plot((xtest[ind,:,feature]-predictions[ind,:,feature])**2,label="pred")
    ax.legend()
    
    
fig.canvas.set_window_title(labels[feature])

#%%
##plot features

#labels = df_orig.columns[2:]

predictions = model.predict(xtest[0].reshape(1,xtest.shape[1],xtest.shape[-1]))
num_to_plot = predictions.shape[-1]

fig,axs = plt.subplots(num_to_plot//2,2,figsize=(10,8))

for ind,ax in enumerate(axs.flat):
    ax.set_title(labels[ind])
    ax.plot(predictions[0,:,ind],label="pred")
    ax.plot(xtest[0,:,ind],label="gt")
    

fig.tight_layout()
plt.legend()

#%%
#plot MSE over all features

predictions = model.predict(xtest)
predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[-1])
xtest_unravel = xtest.reshape(xtest.shape[0]*xtest.shape[1],xtest.shape[-1])

n_features = predictions.shape[-1]

predictions = np.sum(predictions,axis=-1)
xtest_unravel = np.sum(xtest_unravel,axis=-1)

mse = 1/n_features * (xtest_unravel - predictions)**2

plt.scatter(np.arange(len(mse)),mse)

#%%
#plot MSE over all features, FULL DATASET


predictions = model.predict(original_data)
predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[-1])
xtest_unravel = original_data.reshape(original_data.shape[0]*original_data.shape[1],original_data.shape[-1])

n_features = predictions.shape[-1]

predictions = np.sum(predictions,axis=-1)
xtest_unravel = np.sum(xtest_unravel,axis=-1)

mse = 1/n_features * (xtest_unravel - predictions)**2

plt.scatter(np.arange(len(mse)),mse)

#%%


feature = 2

predictions = model.predict(original_data)
predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[-1])
xtest_unravel = original_data.reshape(original_data.shape[0]*original_data.shape[1],original_data.shape[-1])

n_features = predictions.shape[-1]

#predictions = np.sum(predictions,axis=-1)
#xtest_unravel = np.sum(xtest_unravel,axis=-1)

se =  (xtest_unravel[:,feature]-predictions[:,feature])**2

print(len(np.where(se > np.percentile(se,99))[0]))








