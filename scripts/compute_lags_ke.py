
import numpy as np
import pandas as pd


DATASET_FREC = "1"  # Time steps in mins
DATA_PATH = f"./data2/data2_{DATASET_FREC}min.csv"
DATA_PARAMS_PATH = f"./data2/data2_{DATASET_FREC}min_params.json"
FH = [15, 30, 60, 240]
PH_FACTORS = [2, 4, 6]
TEST_START = pd.Timestamp(
    year=2020,
    month=10,
    day=2,
    hour=0,
    minute=0,
    second=0,
)


data = pd.read_csv(DATA_PATH)

times = pd.to_datetime(
    data[data.columns[0]],
    format="%Y-%m-%d %H:%M:%S",
)
data.drop(data.columns[0], axis=1, inplace=True)  # remove timestamps column

data_train = data[times.map(lambda x: x < TEST_START)].values
data_test = data[times.map(lambda x: TEST_START <= x)].values

for fh in FH:
    for ph_factor in PH_FACTORS:
        ph = ph_factor * fh

        x_train = data_train[:-fh, :]
        x_train = np.hstack([
            x_train[i:len(x_train)-ph+i, :]
            for i in range(ph)
        ])
        np.save(
            (
                f"./data2/data2_{ph}ph_{fh}fh"
                f"_{TEST_START.year}-{TEST_START.month}-{TEST_START.day}"
                "_x_train.npy"
            ),
            x_train
        )
        del x_train

        y_train = data_train[ph:, :]
        y_train = np.hstack([
            y_train[i:len(y_train)-fh+i, :]
            for i in range(fh)
        ])
        np.save(
            (
                f"./data2/data2_{ph}ph_{fh}fh"
                f"_{TEST_START.year}-{TEST_START.month}-{TEST_START.day}"
                "_y_train.npy"
            ),
            y_train
        )
        del y_train

        x_test = data_test[:-fh, :]
        x_test = np.hstack([
            x_test[i:len(x_test)-ph+i, :]
            for i in range(ph)
        ])
        np.save(
            (
                f"./data2/data2_{ph}ph_{fh}fh"
                f"_{TEST_START.year}-{TEST_START.month}-{TEST_START.day}"
                "_x_test.npy"
            ),
            x_test
        )
        del x_test

        y_test = data_test[ph:, 0]
        y_test = np.hstack([
            y_test[i:len(y_test)-fh+i].reshape(-1, 1)
            for i in range(fh)
        ])
        np.save(
            (
                f"./data2/data2_{ph}ph_{fh}fh"
                f"_{TEST_START.year}-{TEST_START.month}-{TEST_START.day}"
                "_y_test.npy"
            ),
            y_test
        )
        del y_test

print("END")
