import tensorflow as tf
from sklearn.ensemble import IsolationForest
from tensorflow.keras.layers import *
from tensorflow.keras import Model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def trainparam(df, colname):
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(df[[colname]])

    df['scores'] = model.decision_function(df[[colname]])
    df['anomaly'] = model.predict(df[[colname]])
    df.head(10)

    anomaly = df.loc[df['anomaly'] == -1]
    anomaly_index = list(anomaly.index)
    print("\n" + colname)
    print(anomaly)


data_path = r'hackathon_kpis_anonymised.csv'
df = pd.read_csv(data_path)
df = df.dropna()
df.drop(columns=['cell_name', 'timestamp'], inplace=True)
print(df.columns)
# for column in df.columns:
# trainparam(df, column)

model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.005, max_features=14, random_state=99)
model.fit(df)
df['scores'] = model.decision_function(df)
df['anomaly'] = model.predict(df[['ho_failure_rate', 'num_voice_attempts', 'voice_drop_rate', 'num_data_attempts',
                                  'voice_setup_failure_rate', 'voice_tot_failure_rate', 'avail_period_duration',
                                  'bandwidth', 'throughput_rate', 'data_setup_failure_rate', 'data_drop_rate',
                                  'data_tot_failure_rate', 'unavail_total_rate', 'unavail_unplan_rate']])
df.head(10)

anomaly = df.loc[df['anomaly'] == -1]
anomaly_index = list(anomaly.index)
anomaly.to_csv('anomalysus.csv')
print(anomaly)
