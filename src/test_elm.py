
import json
import time

from hpelm.elm import ELM
import numpy as np

from eval import prequential_mae, prequential_mape


DATASET_DAY = "Variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"
TEST_SIZE = 0.2  # TODO Elegir test size
PH = 180
FH = 30
MODEL_CODE = "elm"

# N_NEURONS = [20]
N_NEURONS = [20, 50, 100, 150, 200]
# ACTIVATION_FUNCTIONS = ["lin"]
ACTIVATION_FUNCTIONS = ["lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf"]

BASE_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms"
DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


# Read datasets
with open(f"{BASE_PATH}_params.json") as fparams:
    params = json.load(fparams)
max_value = params["max"]

x_train = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy")
y_train = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy")
x_test = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_x_test_{TEST_SIZE}.npy")
y_test = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_y_test_{TEST_SIZE}.npy")

# Perform tests
results = []
i = 0
for n_neurons in N_NEURONS:
    for activation_fuction in ACTIVATION_FUNCTIONS:

        # Set model parameters
        params = dict(
            inputs=x_train.shape[1],  # Input size
            outputs=FH,  # Output size
            # batch=...,  # faster, but not compatible with model selection
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

        params |= extra_params

        # Train model
        t0 = time.time()
        model.train(
            x_train,
            y_train,
            # "CV",  # select optimal number of neurons using cross-validation
            "OP",  # L1-regularization  # TEMP
            "r",  # regression
            # k=3,  # cross-validation splits
        )
        duration = time.time() - t0

        # Test model
        y_pred = model.predict(x_test)

        pmae = prequential_mae(max_value*y_test, max_value*y_pred)
        pmae_mean = np.mean(pmae)
        pmae_std = np.std(pmae)

        pmape = prequential_mape(max_value*y_test, max_value*y_pred)
        pmape_mean = np.mean(pmape)
        pmape_std = np.std(pmape)

        # Output results
        print(
            f"{pmae_mean=:.4f}",
            f"{pmae_std=:.4f}",
            f"{pmape_mean=:.4f}",
            f"{pmape_std=:.4f}",
            f"{duration=:.4f}",
        )

        # Save results
        results.append(dict(
            pmae_mean=pmae_mean,
            pmae_std=pmae_std,
            pmape_mean=pmape_mean,
            pmape_std=pmape_std,
            duration=duration,
            params=params,
            id=i,
        ))

        # Save model
        model.save(f"./models/{DATE}_{i:2d}_{MODEL_CODE}.model")

        i += 1

# Write results
with open(f"./results/{DATE}_{MODEL_CODE}.json", "w") as f:
    json.dump(results, f)
