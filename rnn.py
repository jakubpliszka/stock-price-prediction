import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import prepare_dataset


def run_rnn() -> None:
    x_train, y_train, scaler, dataset_train, dataset_test = prepare_dataset()
    rnn = build_model(x_train, y_train)


def build_model(x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.models.Sequential():
    rnn = tf.keras.models.Sequential()
    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.LSTM(units=50))
    rnn.add(tf.keras.layers.Dropout(0.2))
    rnn.add(tf.keras.layers.Dense(units=1))

    rnn.compile(optimizer='adam', loss='mean_squared_error')

    rnn.fit(x_train, y_train, epochs=50, batch_size=32)
    return rnn


