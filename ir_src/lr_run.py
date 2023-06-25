# TODO Actualizar script


import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


PH = 180
FH = 30
TEST_SIZE = 0.2
SEED = 0
DATASET = "./data/Variable/Varennes_1000ms.csv"
DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


data = pd.read_csv(DATASET)
data.drop("Unnamed: 0", axis=1, inplace=True)  # remove date column

train_size = int(len(data)*(1-TEST_SIZE))

# TODO Averiguar qué sensor estuvo prediciendo Pedro
# Estoy utilizando el último. En parte da igual.
x_train = data.values[:train_size-FH, :-1]
x_train = np.hstack([
    x_train[i:len(x_train)-PH+i, :]
    for i in range(PH)
])
y_train = data.values[PH:train_size, -1]
y_train = np.hstack([
    y_train[i:len(y_train)-FH+i].reshape(-1, 1)
    for i in range(FH)
])
x_test = data.values[train_size:-FH, :-1]
x_test = np.hstack([
    x_test[i:len(x_test)-PH+i, :]
    for i in range(PH)
])
y_test = data.values[train_size+PH:, -1]
y_test = np.hstack([
    y_test[i:len(y_test)-FH+i].reshape(-1, 1)
    for i in range(FH)
])

model = LinearRegression(
    n_jobs=-1
)

t0 = time.time()
model.fit(
    x_train,
    y_train,
)
duration = time.time() - t0

# Test model  # TODO Utilizar las métrifas de Pedro
y_pred = model.predict(x_test)
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred)**2)

print(f"{mae=}, {mse=}, {duration=}")

with open(f"./models/{DATE}_lr.json", "w") as f:
    json.dump(
        dict(
            mae=mae,
            mse=mse,
            duration=duration,
            # params=params,
        ),
        f
    )
