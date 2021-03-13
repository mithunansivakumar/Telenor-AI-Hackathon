# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:39:44 2021

@author: adelv
"""
import tensorflow as tf
from tensorflow.keras import *
import pandas as pd

model = tf.keras.models.load_model('data_model.h5')
model.summary()


data_path = r'hackathon_kpis_anonymised.csv'

df = pd.read_csv(data_path,sep=",")
df.sort_values(['cell_name', 'timestamp'], ascending=[True, False],inplace=True)

df2 = df
df2 = df2.fillna(-1)
df2 = df2.values[:,2:]

window_size= model.layers[0].input_shape[0][-2]

df2 = df2.astype(np.single)

assert df2.shape[0] % window_size == 0


df2 = df2.reshape(df2.shape[0]//window_size,window_size,df2.shape[-1])


preds = model.predict(df2)

preds = preds.reshape(preds.shape[0]*window_size,preds.shape[-1])
df2 = df2.reshape(df2.shape[0]*window_size,df2.shape[-1])


assert df2.shape == preds.shape

out = df2

x,y = np.where(df2 == - 1)

out[x,y] = preds[x,y]


np.savetxt('predicted_nans.csv', out, delimiter=',')








#%%

plt.plot(out[0:200,0])
plt.plot(testing[0:200,0])

'''
predictions = model.predict(df2[:num_to_plot])


fig,axs = plt.subplots(2,num_to_plot//2,sharey=True,figsize=(10,10))
for ind,ax in enumerate(axs.flat):
    ax.plot(predictions[ind,:,2],label="pred")
    ax.plot(df2[ind,:,2],label="gt")
    
    



plt.legend()
'''



