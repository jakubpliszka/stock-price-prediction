import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import prepare_dataset


def run_rnn() -> None:
    x_train, y_train, scaler, dataset_train, dataset_test = prepare_dataset()
    rnn = build_model(x_train, y_train)
    predicted_stock_price = predict(scaler, rnn, dataset_train, dataset_test)
    visualize_results(dataset_test, predicted_stock_price)


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


def predict(scaler, rnn, dataset_train, dataset_test):
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    x_test: list = []
    for i in range(60, len(dataset_test)):
        x_test.append(inputs[i - 60:i, 0])

    x_test = np.array(x_test)
    input_shapes = (x_test.shape[0], x_test.shape[1], 1)
    x_test = np.reshape(x_test, input_shapes)

    predicted_stock_price = rnn.predict(x_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price


def visualize_results(real_stock_price, predicted_stock_price) -> None:
    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted  Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

