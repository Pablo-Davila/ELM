
import numpy as np


def mae(y: np.ndarray, o: np.ndarray):
    assert y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1
    assert o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1

    return np.mean(np.abs(y-o))


def mape(y: np.ndarray, o: np.ndarray):
    assert y.ndim == 1 or y.ndim == 2 and y.shape[0] == 1
    assert o.ndim == 1 or o.ndim == 2 and o.shape[0] == 1

    return 100 * np.mean(np.abs((y-o) / y))


def prequential_mae(y: np.ndarray, o: np.ndarray, alpha: int = 0.99):
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


def prequential_mape(y: np.ndarray, o: np.ndarray, alpha: int = 0.99):
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
