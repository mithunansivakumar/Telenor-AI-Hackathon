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

print(os.getcwd())

data_path = r'hackathon_kpis_anonymised.csv'
df = pd.read_csv(data_path)
df_orig = df

df = df.dropna()
df = df.values
df = df[:, 2:]


data_path_pred = r"predicted_nans.csv"
df_pred = pd.read_csv(data_path_pred)
df_pred = df_pred.values

assert df_pred.shape[-1] == df.shape[-1]

df = np.vstack((df, df_pred))
print(df.shape)



def build_model(input_shape, ksize=3):
    # encoder
    conv1 = Conv1D(input_shape.shape[-1], ksize, activation='relu', padding='same')((input_shape))
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = conv1 = Conv1D(64, ksize, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(128, ksize, activation='relu', padding='same')(pool2)

    # decoder
    conv4 = Conv1D(128, ksize, activation='relu', padding='same')(conv3)
    up1 = UpSampling1D(2)(conv4)
    conv5 = Conv1D(64, ksize, activation='relu', padding='same')(up1)
    up2 = UpSampling1D(2)(conv5)
    out = Conv1D(input_shape.shape[-1], ksize, activation='sigmoid', padding='same')(up2)

    return out


# %%


window_size = 32

a = df.shape[0] % window_size
xtrain = df[:-a, :]
xtrain = xtrain.reshape(xtrain.shape[0] // window_size, window_size, xtrain.shape[1])
xtrain = xtrain.astype(np.single)

original_data = xtrain

# shuffler treningsdata
np.random.shuffle(xtrain)
# %%

# beholder en liten andel for testing
test_split = 0.3
xtest = xtrain[:int(xtrain.shape[0] * test_split)]

# resterende brukes til trening
xtrain = xtrain[int(xtrain.shape[0] * test_split):]

input_shape = tf.keras.layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))

# model = Model(input_shape, build_model(input_shape, ksize=9))
# model.compile(loss='mse', optimizer='adam')
# model.summary()

# EPOCHS = 100

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# history = model.fit(xtrain, xtrain, validation_split=.3, epochs=EPOCHS, callbacks=[callback])

# model.save("autoencoder_model.h5")
# pd.DataFrame.from_dict(history.history).to_csv('history.csv', index=False)
model = tf.keras.models.load_model("autoencoder_model.h5")
history = pd.read_csv("history.csv")


# model = tf.keras.models.load_model("autoencoder_model.h5")
# history = model.fit(xtrain, xtrain, validation_split=.3, epochs=EPOCHS, callbacks=[callback])

# HERIFRA OG NED BØR INN I EGNE FUNKJSNOER

# plot training history
def plotData(history):
    plt.plot(history['val_loss'], label="validation_loss")
    plt.plot(history['loss'], label="training_loss")
    plt.legend()
    plt.show()


plotData(history)




# plot a single feature
def plot_single_feature(df, model, num_plot=8, feature=0):
    # henter labels fra dataframen
    labels = df.columns[2:]

    # antallet du vil plotte (må være partall)

    predictions = model.predict(xtest[:num_plot])

    fig, axs = plt.subplots(2, num_plot // 2, sharey=True)

    for ind, ax in enumerate(axs.flat):
        ax.plot(predictions[ind, :, feature], label="pred")
        ax.plot(xtest[ind, :, feature], label="gt")
        ax.legend()

    fig.canvas.set_window_title(labels[feature])
    plt.show()


plot_single_feature(df_orig, model, 8, 0)


# plot SE over one feature
def plotSE_feature(df, model):
    labels = df.columns[2:]

    num_to_plot = 8

    predictions = model.predict(xtest[:num_to_plot])

    fig, axs = plt.subplots(2, num_to_plot // 2, sharey=True)
    feature = 3

    for ind, ax in enumerate(axs.flat):
        ax.plot((xtest[ind, :, feature] - predictions[ind, :, feature]) ** 2, label="pred")
        ax.legend()

    fig.canvas.set_window_title(labels[feature])
    plt.show()


plotSE_feature(df_orig, model)



##plot features
def plot_features(df, model):
    labels = df.columns[2:]

    predictions = model.predict(xtest[0].reshape(1, xtest.shape[1], xtest.shape[-1]))
    num_to_plot = predictions.shape[-1]

    fig, axs = plt.subplots(num_to_plot // 2, 2, figsize=(10, 8))

    for ind, ax in enumerate(axs.flat):
        ax.set_title(labels[ind])
        ax.plot(predictions[0, :, ind], label="pred")
        ax.plot(xtest[0, :, ind], label="gt")

    fig.tight_layout()
    plt.legend()
    plt.show()


plotSE_feature(df_orig, model)



# plot MSE over all features
def plot_MSE_features(df, model):
    predictions = model.predict(xtest)
    predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], predictions.shape[-1])
    xtest_unravel = xtest.reshape(xtest.shape[0] * xtest.shape[1], xtest.shape[-1])

    n_features = predictions.shape[-1]

    predictions = np.sum(predictions, axis=-1)
    xtest_unravel = np.sum(xtest_unravel, axis=-1)

    mse = 1 / n_features * (xtest_unravel - predictions) ** 2

    plt.scatter(np.arange(len(mse)), mse)
    plt.show()


plot_MSE_features(df, model)



# plot MSE over all features, FULL DATASET
def plot_MSE_feature_full(model):
    predictions = model.predict(original_data)
    predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1], predictions.shape[-1])
    xtest_unravel = original_data.reshape(original_data.shape[0] * original_data.shape[1], original_data.shape[-1])

    n_features = predictions.shape[-1]

    predictions = np.sum(predictions, axis=-1)
    xtest_unravel = np.sum(xtest_unravel, axis=-1)

    mse = 1 / n_features * (xtest_unravel - predictions) ** 2

    plt.scatter(np.arange(len(mse)), mse)
    plt.show()


plot_MSE_feature_full(model)
