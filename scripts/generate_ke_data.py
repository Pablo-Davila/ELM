
import os
import json

import pandas as pd
import traces


DATA_PATH = "./data2/"
PERIODS = [1]  # in mins


def read_data(dataset_folder: str) -> pd.DataFrame:
    """Reads and concats all the csv files from a folder.

    Args:
        dataset_folder (str): Data folder's path

    Returns:
        pandas.DataFrame: kinetic energy dataset.
    """
    file_paths = sorted(
        [
            file.path
            for file in os.scandir(dataset_folder)
            if file.is_file() and file.name.endswith(".csv")
        ]
    )

    data = None
    for file_path in file_paths:
        file_df = pd.read_csv(file_path)
        file_df.drop(file_df.columns[1:4], axis="columns", inplace=True)
        file_df.columns = [
            "datetime",
            "KE",
        ]
        file_df.index = file_df.pop("datetime")

        if data is None:
            data = file_df
        else:
            data = pd.concat([data, file_df], axis=0)

    data.index = pd.to_datetime(
        data.index,
        format="%Y-%m-%d %H:%M:%S",
    )
    # data.fillna(method="ffill", inplace=True)
    return data


def separate_evenly(data: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Samples an uneven time series into an evenly separated time series.

    Args:
        data (pandas.DataFrame): uneven time series
        freq (int): sample frequency in min

    Returns:
        pandas.DataFrame: New evenly separated dataframe.
    """
    t0 = data.index.min()
    tf = data.index.max()

    even_index = pd.date_range(t0, tf, freq=f"{freq}min")
    even_data = pd.DataFrame(index=even_index)

    ts = traces.TimeSeries(data["KE"].to_dict())
    even_data["KE"] = [ts.get(t) for t in even_index]

    return even_data


def min_max_normalization(data, min_value=None, max_value=None):
    """Scales the input data.

    Args:
        data (pandas.DataFrame): Data
        max_value (int, optional): Parameter for scaling the data. If None, it is
            computed from the data. Defaults to None.

    Returns:
        pandas.DataFrame: scaled data
    """
    if min_value is None:
        min_value = data.values.min()
    if max_value is None:
        max_value = data.values.max()

    norm_params = {"min": min_value, "max": max_value}

    return (
        (data-min_value) / max_value,
        norm_params
    )


def write_data_csv(data, dataset_folder, freq):
    filename = os.path.join(
        dataset_folder,
        f"{os.path.basename(os.path.normpath(dataset_folder))}_{freq}min.csv",
    )
    data.to_csv(filename)


def write_norm_params(params, dataset_folder, freq):
    filename = os.path.join(
        dataset_folder,
        f"{os.path.basename(os.path.normpath(dataset_folder))}_{freq}min_params.json",
    )
    with open(filename, "w") as outfile:
        json.dump(params, outfile)


def generate_dataset(dataset_folder, periods=[], norm_max=None):
    uneven_data = read_data(dataset_folder)
    for freq in periods:
        data = separate_evenly(uneven_data, freq)
        data, norm_params = min_max_normalization(data, norm_max)
        write_data_csv(data, dataset_folder, freq)
        write_norm_params(norm_params, dataset_folder, freq)


if __name__ == "__main__":

    generate_dataset(DATA_PATH, PERIODS)
