
import json
import os
import random
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from eval import mae, mape, rmse


TEST_START = dict(
    year=2020,
    month=10,
    day=2,
)
FH = [15, 30, 60, 240]
PH_FACTORS = [2, 4, 6]
MODEL_CODE = "rf"

N_ESTIMATORS = [100, 500]
MAX_DEPTHS = [10, 100, None]
MIN_SAMPLES_LEAF = [1, 4]

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
        for n_estimators in N_ESTIMATORS:
            for max_depth in MAX_DEPTHS:
                for min_samples_leaf in MIN_SAMPLES_LEAF:

                    # Set model parameters
                    params = dict(
                        n_estimators=n_estimators,
                        criterion='squared_error',
                        max_depth=max_depth,
                        min_samples_split=2,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=0.0,
                        max_features=1.0,
                        max_leaf_nodes=None,
                        min_impurity_decrease=0.0,
                        bootstrap=True,
                        oob_score=False,
                        n_jobs=None,
                        random_state=random.randrange(1000),
                        verbose=0,
                        warm_start=False,
                        ccp_alpha=0.0,
                        max_samples=None
                    )

                    # Build model
                    model = RandomForestRegressor(**params)

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
