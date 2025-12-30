import numpy as np
import monet_stats as stats


def test_FB():
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 2.5])
    result = stats.FB(obs, mod)
    assert isinstance(result, float)


def test_ME():
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 2.5])
    result = stats.ME(obs, mod)
    assert isinstance(result, float)
