
import json
import os
import random
import time

import numpy as np
from sklearn.linear_model import Ridge

from eval import mae, mape, rmse


TEST_START = dict(
    year=2020,
    month=10,
    day=2,
)
FH = [15, 30, 60, 240]
PH_FACTORS = [2, 4, 6]
MODEL_CODE = "lr"

ALPHAS = [0, 1, 2]

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
        for alpha in ALPHAS:

            # Set model parameters
            params = dict(
                alpha=alpha,
                fit_intercept=True,
                copy_X=True,
                max_iter=None,
                tol=0.0001,
                solver='auto',
                positive=False,
                random_state=random.randrange(1000)
            )

            # Build model
            model = Ridge(**params)

            # Train model
            t0 = time.time()
            model.fit(
                x_train,
                y_train,
            )
            duration = time.time() - t0

            # Test model
            y_pred = model.predict(x_test)

            y_test = y_test*(max_value - min_value) + min_value
            y_pred = y_pred*(max_value - min_value) + min_value

            _mae = np.mean(mae(y_test, y_pred))
            _mape = np.mean(mape(y_test, y_pred))
            _rmse = np.mean(rmse(y_test, y_pred))

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
