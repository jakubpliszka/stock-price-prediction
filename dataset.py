from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATASET_FILE_NAME: str = "apple_stock_price_history.csv"


def prepare_dataset() -> None:
    dataset = pd.read_csv(DATASET_FILE_NAME)
    dataset_train = dataset.iloc[:, 1:2].values

    # Parse string to float values
    for value in dataset_train:
        value[0] = float(value[0][2:])

    scaler = MinMaxScaler()

    dataset_train = scaler.fit_transform(dataset_train)

    x_train: list[Any] = []
    y_train: list[Any] = []
    for i in range(60, len(dataset_train)):
        x_train.append(dataset_train[i-60:i, 0])
        y_train.append(dataset_train[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

