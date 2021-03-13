# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:04:31 2021

@author: adelv
"""
import tensorflow as tf
from tensorflow.keras import *
import pandas as pd

model = tf.keras.models.load_model('autoencoder_model.h5')
model.summary()

data_path = r'anomaly_dataset.csv'
df = pd.read_csv(data_path)
labels = df.columns[3:]

df.sort_values(['cell_name', 'timestamp'], ascending=[True, False],inplace=True)
df=df.values[:,3:]
df = df.astype(np.single)

window_size= model.layers[0].input_shape[0][-2]

predictions = model.predict(df[:32].reshape(1,32,14))

predictions = predictions.reshape(32,14)
gt = df[:32]

feature = 1
plt.title(labels[feature])
plt.plot(predictions[:,feature],label="prediction")
plt.plot(gt[:,feature],label="ground truth")
plt.legend()

fig,ax = plt.subplots(1)
mse = (gt - predictions)**2


hist,bins = np.histogram(mse[:,feature],10)
plt.hist(hist,bins='auto')


