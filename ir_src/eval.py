
from typing import Union, Dict
import numpy as np


def mae(y: Union[np.ndarray, Dict], o: Union[np.ndarray, Dict]) -> float:
    if isinstance(y, dict):
        y = np.array(list(y.values()))
    if isinstance(o, dict):
        o = np.array(list(o.values()))

    assert y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1
    assert o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1

    return np.mean(np.abs(y-o))


def mape(y: Union[np.ndarray, Dict], o: Union[np.ndarray, Dict]) -> float:
    if isinstance(y, dict):
        y = np.array(list(y.values()))
    if isinstance(o, dict):
        o = np.array(list(o.values()))

    assert y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1
    assert o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1

    # TEMP Find a better solution for zero divisions
    res = np.abs((y-o) / y)
    res = np.where(
        res == np.Infinity,
        np.zeros_like(res),
        res,
    )
    return 100 * np.mean(res)


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
