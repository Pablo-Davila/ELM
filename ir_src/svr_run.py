
import json
import time

import numpy as np
from sklearn.svm import SVR


DATASET_DAY = "Variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"
TEST_SIZE = 0.2
PH = 180
FH = 30
KERNELS = ["rbf", "linear"]
EPSILONS = [0.05, 0.1, 0.2]

DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


# Read datasets
x_train = np.load(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy")
y_train = np.load(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy")
x_test = np.load(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_x_test_{TEST_SIZE}.npy")
y_test = np.load(f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms_{PH}ph_{FH}fh_y_test_{TEST_SIZE}.npy")

# Perform tests
results = []
for kernel in KERNELS:
    for epsilon in EPSILONS:
        
        # Set model parameters
        params = dict(
            kernel=kernel,
            gamma='scale',
            tol=0.001,
            C=1.0,
            epsilon=epsilon,
            shrinking=True,  # heuristic
            cache_size=1000,  # 200,
            verbose=True,
            max_iter=-1,  # No limit
        )

        # Build model
        model = SVR(**params)

        # Train model
        t0 = time.time()
        model.fit(
            x_train,
            y_train,
        )
        duration = time.time() - t0

        # Test model  # TODO Utilizar las métrifas de Pedro
        y_pred = model.predict(x_test)
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean(np.mean(y_test - y_pred, axis=1)**2)

        # Output results
        print(f"{mae=}, {mse=}, {duration=}")
        
        # Save results
        results.append(dict(
            mae=mae,
            mse=mse,
            duration=duration,
            params=params,
        ))

# Write results
with open(f"./models/{DATE}_svr.json", "w") as f:
    json.dump(
        results,
        f
    )
