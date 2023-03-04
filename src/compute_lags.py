
import numpy as np
import pandas as pd


DATASET_DAY = "Variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"
PH = 180
FH = 30
TEST_SIZE = 0.2

DATASET_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms.csv"


data = pd.read_csv(DATASET_PATH)
data.drop("Unnamed: 0", axis=1, inplace=True)  # remove date column

train_size = int(len(data)*(1-TEST_SIZE))

# TODO Averiguar qué sensor estuvo prediciendo Pedro
# Estoy utilizando el último. No debería ser muy determinante.
x_train = data.values[:train_size-FH, :-1]
x_train = np.hstack([
    x_train[i:len(x_train)-PH+i, :]
    for i in range(PH)
])
np.save(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy", x_train)

y_train = data.values[PH:train_size, -1]
y_train = np.hstack([
    y_train[i:len(y_train)-FH+i].reshape(-1, 1)
    for i in range(FH)
])
np.save(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy", y_train)

x_test = data.values[train_size:-FH, :-1]
x_test = np.hstack([
    x_test[i:len(x_test)-PH+i, :]
    for i in range(PH)
])
np.save(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_x_test_{TEST_SIZE}t.npy", x_test)

y_test = data.values[train_size+PH:, -1]
y_test = np.hstack([
    y_test[i:len(y_test)-FH+i].reshape(-1, 1)
    for i in range(FH)
])
np.save(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_y_test_{TEST_SIZE}t.npy", y_test)

