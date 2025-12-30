import numpy as np
import monet_stats as stats


def test_FSS():
    obs = np.random.randint(0, 2, (5, 5))
    mod = np.random.randint(0, 2, (5, 5))
    result = stats.FSS(obs, mod, threshold=0.5, window_size=3)
    assert isinstance(result, float)


def test_EDS():
    obs = np.random.rand(5, 5)
    mod = np.random.rand(5, 5)
    result = stats.EDS(obs, mod, threshold=0.5)
    assert isinstance(result, float)


def test_CRPS():
    # CRPS expects ensemble axis last, so shape (n, m), axis=-1
    ensemble = np.random.rand(5, 10)
    obs = np.random.rand(5)
    result = stats.CRPS(ensemble, obs, axis=-1)
    assert np.isscalar(result) or isinstance(result, np.ndarray)
