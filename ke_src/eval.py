
from typing import Union, Dict
import numpy as np


def rmse(y: np.ndarray, o: np.ndarray) -> float:
    if (
        (y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1)
        and (o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1)
    ):
        return np.sqrt(np.sum((y-o)**2) / len(y))
    elif y.ndim == 2 and o.ndim == 2:
        n = len(y)
        return np.apply_along_axis(
            lambda r: np.sqrt(np.sum(r**2) / n),
            1,
            y - o
        )
    else:
        raise Exception("Input arrays must both have either 1 or 2 dimensions")


def mae(y: Union[np.ndarray, Dict], o: Union[np.ndarray, Dict]) -> float:
    if isinstance(y, dict):
        y = np.array(list(y.values()))
    if isinstance(o, dict):
        o = np.array(list(o.values()))

    if (
        (y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1)
        and (o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1)
    ):
        return np.mean(np.abs(y-o))
    elif y.ndim == 2 and o.ndim == 2:
        return np.apply_along_axis(
            lambda r: np.mean(np.abs(r)),
            1,
            y - o
        )
    else:
        raise Exception("Input arrays must both have either 1 or 2 dimensions")


def mape(y: Union[np.ndarray, Dict], o: Union[np.ndarray, Dict]) -> float:
    if isinstance(y, dict):
        y = np.array(list(y.values()))
    if isinstance(o, dict):
        o = np.array(list(o.values()))

    if (
        (y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1)
        and (o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1)
    ):
        return 100 * np.mean(np.abs((y-o) / y))
    elif y.ndim == 2 and o.ndim == 2:
        return np.apply_along_axis(
            lambda r: 100 * np.mean(np.abs(r)),
            1,
            (y-o) / y
        )
    else:
        raise Exception("Input arrays must both have either 1 or 2 dimensions")


def prequential_mae(y: np.ndarray, o: np.ndarray, alpha: int = 0.99) -> np.ndarray:
    assert y.ndim == 2 and 1 not in y.shape
    assert o.ndim == 2 and 1 not in o.shape

    si = 0
    bi = 0
    res = []
    for yi, oi in zip(y, o):
        si = mae(yi, oi) + alpha*si
        bi = 1 + alpha*bi
        res.append(si / bi)

    return np.array(res).reshape(-1, 1)


def prequential_mape(y: np.ndarray, o: np.ndarray, alpha: int = 0.99) -> np.ndarray:
    assert y.ndim == 2 and 1 not in y.shape
    assert o.ndim == 2 and 1 not in o.shape

    si = 0
    bi = 0
    res = []
    for yi, oi in zip(y, o):
        si = mape(yi, oi) + alpha*si
        bi = 1 + alpha*bi
        res.append(si / bi)

    return np.array(res).reshape(-1, 1)
