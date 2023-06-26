
import json
import os
import time

from hpelm.elm import ELM
import numpy as np

from eval import mae, mape, rmse


TEST_START = dict(
    year=2020,
    month=10,
    day=2,
)
FH = [15, 30, 60, 240]
PH_FACTORS = [2, 4, 6]
MODEL_CODE = "elm"

N_NEURONS = [20, 50, 120, 300]
ACTIVATION_FUNCTIONS = ["lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf"]


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
        for n_neurons in N_NEURONS:
            for activation_fuction in ACTIVATION_FUNCTIONS:

                # Set model parameters
                params = dict(
                    inputs=x_train.shape[1],  # Input size
                    outputs=fh,  # Output size
                    # batch=...,  # Faster, but not compatible with model selection
                    accelerator="basic",  # Alternative: "GPU",
                    precision="double",  # Default: double
                    tprint=3,  # Log every 3 seconds
                )

                # Build model
                model = ELM(**params)
                extra_params = dict(
                    number=n_neurons,
                    func=activation_fuction,
                )
                model.add_neurons(**extra_params)

                params = {**params, **extra_params}

                # Train model
                t0 = time.time()
                model.train(
                    x_train,
                    y_train,
                    # "CV",  # Perform cross-validation
                    "OP",  # L1-regularization
                    "r",  # Regression
                    # k=3,  # Number of cross-validation splits
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

                # Save model
                model.save(
                    f"./models/{timestamp}_{i:03d}_{MODEL_CODE}"
                    f"_{ph}ph_{fh}fh.model"
                )

                i += 1

        # Write results
        with open(
            f"./results/KE/{timestamp}_{MODEL_CODE}_{ph}ph_{fh}fh.json",
            "w"
        ) as f:
            json.dump(results, f)
