
import json
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

from eval import prequential_mae, prequential_mape


DATASET_DAY = "Variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"
TEST_SIZE = 0.2  # TODO Elegir test size
PH = 180
FH = 30
MODEL_CODE = "mlp"
LEARNING_RATES = [0.001, 0.005, 0.010]
LAYERS = [(64,)*6]
# LAYERS = [(150, 100, 80, 60, 40, 30)]
SEED = 0

BASE_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms"
DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


print(DATE)

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
for learning_rate in LEARNING_RATES:
    for hidden_layer_sizes in LAYERS:
        
        # Set model parameters
        params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # TODO param alpha
            batch_size=10000,  # TODO param batch_size
            learning_rate='constant',  # TODO param learning_rate
            # power_t=0.5,  # Only for sgd
            learning_rate_init=learning_rate,
            max_iter=50,  # 1000,
            shuffle=True,
            random_state=SEED,
            tol=0.001,
            verbose=True,
            warm_start=False,
            # momentum=0.9,  # Only for sgd
            # nesterovs_momentum=True,  # Only for sgd
            early_stopping=True,
            validation_fraction=0.1,  # TODO param validation_fraction
            beta_1=0.9,  # TODO param beta_1
            beta_2=0.999,  # TODO param beta_2
            epsilon=1e-08,  # TODO param epsilon
            n_iter_no_change=10,  # TODO param n_iter_no_change
            # max_fun=15000,  # Only for lbfgs
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
        ))

# Write results
with open(f"./results/{DATE}_{MODEL_CODE}.json", "w") as f:
    json.dump(
        results,
        f
    )
