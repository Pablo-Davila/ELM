
import json
import time

import numpy as np
import tqdm

from eval import mae, mape
from numpy_oselm import OS_ELM


DATASET_DAY = "Variable"
DATASET_PLACE = "Varennes"
DATASET_FREC = "1000"
TEST_SIZE = 0.0
PH = 180
FH = 30
MODEL_CODE = "oselm"
N_NEURONS = [20, 50, 100, 150, 200]  # Around the best: [15, 20, 25, 30,] (batch=64)
BATCH_SIZES = [1, 50, 100, 300, 3000, 10_000]
INITIAL_TRAIN_FACTORS = [2.5, 4.5, 6.5, 2.5, 4.5, 6.5]  # Size of initial training chunk
ALPHA = 0.99

BASE_PATH = f"./data/{DATASET_DAY}/{DATASET_PLACE}_{DATASET_FREC}ms"
DATE = time.strftime("%Y-%m-%d_%H-%M", time.localtime())


# Read datasets
with open(f"{BASE_PATH}_params.json") as fparams:
    params = json.load(fparams)
max_value = params["max"]

x_data = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_x_train_{TEST_SIZE}.npy")
y_data = np.load(f"{BASE_PATH}_{PH}ph_{FH}fh_y_train_{TEST_SIZE}.npy")

# Perform tests
i = 0
results = []
for n_neurons in N_NEURONS:
    for batch_size in BATCH_SIZES:
        for initial_train_factor in INITIAL_TRAIN_FACTORS:
            print(i)

            # Set model parameters
            params = dict(
                n_input_nodes=x_data.shape[1],
                n_hidden_nodes=n_neurons,
                n_output_nodes=FH,
                # TEMP activation options:
                # Default: sigmoid',
                # Could implement: "lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "rbf_linf"
                # TEMP loss options:
                # Default: mean_squared_error
                # Could implement: 'mean_absolute_error', 'categorical_crossentropy',
                # 'binary_crossentropy'
            )

            # Build model
            model = OS_ELM(**params)

            # Split initial and sequential data
            initial_train_size = int(initial_train_factor * n_neurons)
            assert initial_train_size <= x_data.shape[0]
            x_initial = x_data[:initial_train_size, :]
            y_initial = y_data[:initial_train_size, :]
            x_sequential = x_data[initial_train_size:, :]
            y_sequential = y_data[initial_train_size:, :]

            t0 = time.time()  # Timer start

            # Initial train
            pbar = tqdm.tqdm(total=x_data.shape[0], desc="Initial training phase")
            model.init_train(x_initial, y_initial)
            pbar.update(n=x_initial.shape[0])

            # Sequential train-test
            pbar.set_description("Sequential training phase")
            si_mae = 0
            si_mape = 0
            bi = 0
            pmae = []
            pmape = []

            for i in range(0, x_sequential.shape[0], batch_size):

                # Get batch
                x_batch = x_sequential[i:i+batch_size]
                y_batch = y_sequential[i:i+batch_size]

                # Evaluate model
                o_batch = model.predict(x_batch)

                for yi, oi in zip(max_value*y_batch, max_value*o_batch):
                    si_mae = mae(yi, oi) + ALPHA*si_mae
                    si_mape = mape(yi, oi) + ALPHA*si_mape
                    bi = 1 + ALPHA * bi
                    pmae.append(si_mae / bi)
                    pmape.append(si_mape / bi)

                # Train model
                model.seq_train(x_batch, y_batch)

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
            params |= dict(
                batch_size=batch_size,
                initial_train_factor=initial_train_factor,
                activation="sigm",
                loss="mean_squared_error",
            )
            results.append(dict(
                pmae_mean=pmae_mean,
                pmae_std=pmae_std,
                pmape_mean=pmape_mean,
                pmape_std=pmape_std,
                duration=duration,
                params=params,
            ))

            # # ===========================================
            # # Evaluation
            # # ===========================================
            # # we currently support 'loss' and 'accuracy' for 'metrics'.
            # # NOTE: 'accuracy' is valid only if the model assumes
            # # to deal with a classification problem, while 'loss' is always valid.
            # # loss = model.evaluate(x_test, t_test, metrics=['loss']
            # [loss, accuracy] = model.evaluate(
            #     x_test, t_test, metrics=['loss', 'accuracy'])
            # print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

            # # initialize weights of model
            # model.initialize_variables()

            # # ===========================================
            # # Load model
            # # ===========================================
            # model.restore('./checkpoint/model.ckpt')

            # # Save model
            # model.save(f"./models/{DATE}_{i:2d}_{MODEL_CODE}.model")

            i += 1

# Write results
with open(f"./results/{DATE}_{MODEL_CODE}.json", "w") as f:
    json.dump(results, f)
