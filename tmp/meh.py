
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def time_to_int(time: str):
    h, m, s_ms = time.split(":")
    s, ms = s_ms.split(".")
    return (
        3_600_000 * h
        + 60_000 * m
        + 1_000 * s
        + ms
    )


# for i in range(1, 18):
#     if i != 3: continue
#     data_path = f"./data/Clear sky/20141230_VAR{i:02d}.csv"

#     data = pd.read_csv(data_path)
#     plt.plot(data["G1 (W/m2)"])
#     # plt.plot(data["G2 (W/m2)"])

# plt.savefig("results/meh")


index = set()
data = []
for i in range(1, 18):
    data_path = f"./data/Clear sky/20141230_VAR{i:02d}.csv"
    _data = pd.read_csv(data_path)
    _data["Timestamp (hh:mm:ss.nnn)"] = _data["Timestamp (hh:mm:ss.nnn)"].map(time_to_int)
#     data.append(_data)
#     index |= set(_data.index)

# index = sorted(list(index))
# for _data in data:


    # plt.plot(_data["G1 (W/m2)"])
    plt.plot(_data["Timestamp (hh:mm:ss.nnn)"], _data["G2 (W/m2)"])

plt.savefig("results/meh")
