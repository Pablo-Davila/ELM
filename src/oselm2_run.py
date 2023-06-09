
import json
import os
import random
import time

import numpy as np
import tqdm

from eval import mae, mape
from pyoselm import OSELMRegressor


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
MODEL_CODE = "oselm2"
N_NEURONS = [20, 50, 100, 150, 200]
ACTIVATION_FUNCTIONS = [
    "tanh",
    "sine",
    "tribas",
    "inv_tribase",
    "sigmoid",
    "hardlim",
    "softlim",
    "gaussian",
    "multiquadric",
    "inv_multiquadric"
]
INITIAL_TRAIN_FACTORS = [1]
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
for n_neurons in N_NEURONS:
    for activation_function in ACTIVATION_FUNCTIONS:
        for initial_train_factor in INITIAL_TRAIN_FACTORS:
            print(count)

            # Set model parameters
            params = dict(
                n_hidden=n_neurons,
                activation_func=activation_function,
                activation_args=None,
                use_woodbury=False,
                random_state=random.randrange(1000),
            )

            # Build model
            model = OSELMRegressor(**params)

            # Split initial and sequential data
            initial_train_size = int(initial_train_factor * n_neurons)
            assert initial_train_size <= x_data.shape[0]
            x_initial = x_data[:initial_train_size, :]
            y_initial = y_data[:initial_train_size, :]
            x_sequential = x_data[initial_train_size:, :]
            y_sequential = y_data[initial_train_size:, :]

            t0 = time.time()  # Timer start

            # Initial train
            pbar = tqdm.tqdm(
                total=x_data.shape[0],
                desc="Initial training phase"
            )
            model.fit(x_initial, y_initial)
            pbar.update(n=x_initial.shape[0])

            # Sequential train-test
            pbar.set_description("Sequential training phase")
            si_mae = 0
            si_mape = 0
            bi = 0
            pmae = []
            pmape = []

            for i in range(0, x_sequential.shape[0]):

                # Get batch
                x_batch = x_sequential[i:i+1]
                y_batch = y_sequential[i:i+1]

                # Evaluate model
                o_batch = model.predict(x_batch)

                for yi, oi in zip(max_value*y_batch, max_value*o_batch):
                    si_mae = mae(yi, oi) + ALPHA*si_mae
                    si_mape = mape(yi, oi) + ALPHA*si_mape
                    bi = 1 + ALPHA * bi
                    pmae.append(si_mae / bi)
                    pmape.append(si_mape / bi)

                # Train model
                model.fit(x_batch, y_batch)

                pbar.update(n=len(x_batch))
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

            # Save results
            params = dict(
                **params,
                initial_train_factor=initial_train_factor,
            )
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
