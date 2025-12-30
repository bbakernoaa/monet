import numpy as np
import monet_stats as stats


def test_HSS():
    obs = np.array([0, 1, 1, 0, 1])
    mod = np.array([0, 1, 0, 0, 1])
    minval = 0
    maxval = 1
    result = stats.HSS(obs, mod, minval, maxval)
    assert isinstance(result, float)


def test_ETS():
    obs = np.array([0, 1, 1, 0, 1])
    mod = np.array([0, 1, 0, 0, 1])
    minval = 0
    maxval = 1
    result = stats.ETS(obs, mod, minval, maxval)
    assert isinstance(result, float)


def test_CSI():
    obs = np.array([0, 1, 1, 0, 1])
    mod = np.array([0, 1, 0, 0, 1])
    minval = 0
    maxval = 1
    result = stats.CSI(obs, mod, minval, maxval)
    assert isinstance(result, float)
