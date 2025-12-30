import numpy as np
import monet_stats as stats


def test_NSE():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1, 2, 2, 4, 5])
    result = stats.NSE(obs, mod)
    assert isinstance(result, float)
