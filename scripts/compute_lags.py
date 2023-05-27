
import numpy as np
import pandas as pd


# DATASET_DAY = "Variable"
DATASET_DAY = "Very variable"
DATASET_PLACE = "Varennes"
# DATASET_FREC = "1000"  # For "Variable" (1 ins/s)
DATASET_FREC = "500"  # For "Very variable" (2 ins/s)
# PH = 180  # For "Variable"
PH = 360  # For "Very variable"
# FH = 30  # For "Variable"
FH = 60  # For "Very variable"
# TEST_SIZE = 0.2  # For batch learning
TEST_SIZE = 0.0  # For online learning

BASE_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms"


data = pd.read_csv(f"{BASE_PATH}.csv")
data.drop("Unnamed: 0", axis=1, inplace=True)  # remove date column

train_size = int(len(data)*(1-TEST_SIZE))

x_train = data.values[:train_size-FH, 1:]
x_train = np.hstack([
    x_train[i:len(x_train)-PH+i, :]
    for i in range(PH)
])
np.save(f"{BASE_PATH}_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy", x_train)

y_train = data.values[PH:train_size, 0]
y_train = np.hstack([
    y_train[i:len(y_train)-FH+i].reshape(-1, 1)
    for i in range(FH)
])
np.save(f"{BASE_PATH}_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy", y_train)

x_test = data.values[train_size:-FH, 1:]
x_test = np.hstack([
    x_test[i:len(x_test)-PH+i, :]
    for i in range(PH)
])
np.save(f"{BASE_PATH}_{PH}ph_{FH}fh_x_test_{TEST_SIZE}t.npy", x_test)

y_test = data.values[train_size+PH:, 0]
y_test = np.hstack([
    y_test[i:len(y_test)-FH+i].reshape(-1, 1)
    for i in range(FH)
])
np.save(f"{BASE_PATH}_{PH}ph_{FH}fh_y_test_{TEST_SIZE}t.npy", y_test)

