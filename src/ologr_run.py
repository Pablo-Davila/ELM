
import json
import os
import time

import numpy as np
from river import linear_model
from river import multioutput
from river import optim
from river import stream
import tqdm

from eval import mae, mape


DATASET_DAY = "Variable"
# DATASET_DAY = "Very variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"  # For "Variable" (1 ins/s)
# DATASET_FREC = "500"  # For "Very variable" (2 ins/s)
TEST_SIZE = 0.0
PH = 100  # For "Variable"
# PH = 180  # For "Variable"
# PH = 200  # For "Very variable"
# PH = 360  # For "Very variable"
FH = 10  # For "Variable"
# FH = 30  # For "Variable"
# FH = 20  # For "Very variable"
# FH = 60  # For "Very variable"
MODEL_CODE = "ologr"
INITIAL_TRAIN_SIZES = [1]  # [250, 450, 650]
L1S = [0.0010, 0.0100]
INTERCEPT_LR = [0.0001, 0.0010, 0.0100]
ALPHA = 0.99

DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
BASE_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms"
OUT_PATH = f"./results/{DATASET_DAY}/{DATE}_{MODEL_CODE}_ph{PH}_fh{FH}.json"


# Read datasets
with open(f"{BASE_PATH}_params.json") as fparams:
    params = json.load(fparams)
max_value = params["max"]

x_data = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy")
y_data = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy")

# Perform tests
count = 0
results = []
for initial_train_size in INITIAL_TRAIN_SIZES:
    for l1 in L1S:
        for intercept_lr in INTERCEPT_LR:
            print(count)

            # Set model parameters
            optimizer_log = "optim.SGD(0.01)"
            loss_log = "optim.losses.Squared()"
            params = dict(
                optimizer=optim.SGD(0.01),
                loss=optim.losses.Squared(),
                l2=0,
                l1=l1,
                intercept_init=0,
                intercept_lr=intercept_lr,
                clip_gradient=1e12,
                initializer=None,
            )

            # Build model
            model = multioutput.RegressorChain(
                linear_model.LogisticRegression(**params)
            )

            # Split initial and sequential data
            assert initial_train_size <= x_data.shape[0]
            x_initial = x_data[:initial_train_size, :]
            y_initial = y_data[:initial_train_size, :]
            data_initial = stream.iter_array(x_initial, y_initial)
            x_sequential = x_data[initial_train_size:, :]
            y_sequential = y_data[initial_train_size:, :]
            data_sequential = stream.iter_array(x_sequential, y_sequential)

            t0 = time.time()  # Timer start

            # Initial train
            pbar = tqdm.tqdm(
                total=x_data.shape[0],
                desc="Initial training phase"
            )
            for x, y in data_initial:
                model.learn_one(x, y)
                pbar.update(n=1)

            # Sequential train-test
            pbar.set_description("Sequential training phase")
            si_mae = 0
            si_mape = 0
            bi = 0
            pmae = []
            pmape = []

            for x, y in data_sequential:

                # Evaluate model
                yi = {k: max_value*e for k, e in y.items()}
                oi = {k: max_value*e for k, e in model.predict_one(x).items()}

                si_mae = mae(yi, oi) + ALPHA*si_mae
                si_mape = mape(yi, oi) + ALPHA*si_mape
                bi = 1 + ALPHA * bi
                pmae.append(si_mae / bi)
                pmape.append(si_mape / bi)

                # Train model
                model.learn_one(x, y)

                pbar.update(n=1)

            duration = time.time() - t0  # Timer end
            pbar.close()

            pmae_mean = np.array(pmae).mean()
            pmae_std = np.array(pmae).std()
            pmape_mean = np.array(pmape).mean()
            pmape_std = np.array(pmape).std()

            # Output results
            print(
                f"{pmae_mean=:.4f}",
                f"{pmae_std=:.4f}",
                f"{pmape_mean=:.4f}",
                f"{pmape_std=:.4f}",
                f"{duration=:.4f}",
            )

            params["optimizer"] = optimizer_log
            params["loss"] = loss_log
            results.append(dict(
                pmae_mean=pmae_mean,
                pmae_std=pmae_std,
                pmape_mean=pmape_mean,
                pmape_std=pmape_std,
                duration=duration,
                params=params,
            ))

            count += 1

# Write results
if not os.path.exists(os.path.dirname(OUT_PATH)):
    os.makedirs(os.path.dirname(OUT_PATH))
with open(OUT_PATH, "w") as f:
    json.dump(results, f)
