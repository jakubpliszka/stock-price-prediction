from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TRAIN_DATASET_FILE_NAME: str = "apple_stock_price_train.csv"
TEST_DATASET_FILE_NAME: str = "apple_stock_price_test.csv"


def prepare_dataset() -> (np.ndarray, np.ndarray):
    dataset_train = pd.read_csv(TRAIN_DATASET_FILE_NAME)
    dataset_train = dataset_train.iloc[:, 1:2].values

    scaler = MinMaxScaler()
    dataset_train = scaler.fit_transform(dataset_train)

    x_train: list = []
    y_train: list = []
    for i in range(60, len(dataset_train)):
        x_train.append(dataset_train[i-60:i, 0])
        y_train.append(dataset_train[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    input_shapes = (x_train.shape[0], x_train.shape[1], 1)
    x_train = np.reshape(x_train, input_shapes)

    dataset_test = pd.read_csv(TEST_DATASET_FILE_NAME)
    dataset_test = dataset_test.iloc[:, 1:2].values

    return x_train, y_train, scaler, dataset_train, dataset_test
