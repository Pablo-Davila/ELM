
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_DAY = "Clear sky"
# DATASET_DAY = "Overcast"
# DATASET_DAY = "Variable"
# DATASET_DAY = "Very variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"  # For "Variable" (1 ins/s)
# DATASET_FREC = "500"  # For "Very variable" (2 ins/s)

DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
DATA_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms.csv"
PARAM_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_params.json"
OUT_PATH = f"./results/{DATASET_DAY}/{DATE}_grafica.png"


# Read datasets
with open(PARAM_PATH) as fparams:
    params = json.load(fparams)
min_value = params["min"]
max_value = params["max"]

data = pd.read_csv(DATA_PATH)
data.drop(data.columns[0], axis="columns", inplace=True)

# Create image
for i, col in enumerate(data.columns):
    if i == 0: continue
    s = data[col].drop(data[data[col] == 0].index, axis=0)
    plt.plot(s.index, s)
    # break
    plt.savefig(f"{OUT_PATH}_{i}.png")
    plt.clf()
    plt.cla()
    plt.close()

# data = np.sin(np.linspace(0, 100, 10_000))
# plt.plot(data)

# Write results
if not os.path.exists(os.path.dirname(OUT_PATH)):
    os.makedirs(os.path.dirname(OUT_PATH))

plt.savefig(OUT_PATH)
