
import json
import os
import random
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

from eval import mae, mape, rmse


TEST_START = dict(
    year=2020,
    month=10,
    day=2,
)
FH = [15, 30, 60, 240]
PH_FACTORS = [2, 4, 6]
MODEL_CODE = "mlp"

HIDDEN_LAYER_SIZES = [
    tuple(n_neurons for _ in range(n_layers))
    for n_layers in (1, 2, 3)
    for n_neurons in (64, 128)
]
LEARNING_RATE_INIT = []

# Create output dirs
if not os.path.exists("./results/KE/"):
    os.makedirs("./results/KE/")
if not os.path.exists("./models/"):
    os.makedirs("./models/")

# Read dataset params
with open("./data2/data2_1min_params.json") as fparams:
    params = json.load(fparams)
min_value = params["min"]
max_value = params["max"]

for fh in FH:
    for ph_factor in PH_FACTORS:
        ph = ph_factor * fh
        print(f"{ph=}, {fh=}")

        base_path = (
            f"./data2/data2_{ph}ph_{fh}fh"
            f"_{TEST_START['year']}-{TEST_START['month']}-{TEST_START['day']}"
        )
        timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

        # Read datasets
        x_train = np.load(f"{base_path}_x_train.npy")
        y_train = np.load(f"{base_path}_y_train.npy")
        x_test = np.load(f"{base_path}_x_test.npy")
        y_test = np.load(f"{base_path}_y_test.npy")

        # Grid search
        results = []
        i = 0
        for hidden_layer_sizes in HIDDEN_LAYER_SIZES:
            for learning_rate_init in LEARNING_RATE_INIT:

                # Set model parameters
                params = dict(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size=512,
                    learning_rate_init=learning_rate_init,  # TODO
                    max_iter=50,  # epochs
                    shuffle=True,
                    random_state=random.randrange(1000),
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    early_stopping=False,
                    validation_fraction=0.1,  # Only for early stopping
                    beta_1=0.96,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=5,
                )

                # Build model
                model = MLPRegressor(**params)

                # Train model
                t0 = time.time()
                model.fit(
                    x_train,
                    y_train,
                )
                duration = time.time() - t0

                # Test model
                y_pred = model.predict(x_test)

                y_test_orig = y_test*(max_value - min_value) + min_value
                y_pred = y_pred*(max_value - min_value) + min_value

                _mae = np.mean(mae(y_test_orig, y_pred))
                _mape = np.mean(mape(y_test_orig, y_pred))
                _rmse = np.mean(rmse(y_test_orig, y_pred))

                # Output results
                print(
                    f"{_mae=:.4f}",
                    f"{_mape=:.4f}",
                    f"{_rmse=:.4f}",
                    f"{duration=:.4f}",
                )

                # Save results
                results.append(dict(
                    mae=_mae,
                    mape=_mape,
                    rmse=_rmse,
                    duration=duration,
                    params=params,
                    id=i,
                ))

                i += 1

        # Write results
        with open(
            f"./results/KE/{timestamp}_{MODEL_CODE}_{ph}ph_{fh}fh.json",
            "w"
        ) as f:
            json.dump(results, f)
