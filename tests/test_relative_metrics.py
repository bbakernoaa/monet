import numpy as np
import pytest
import monet_stats as stats


def test_FB():
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 2.5])
    result = stats.FB(obs, mod)
    assert isinstance(result, float)


@pytest.mark.skipif(
    not hasattr(stats, 'ME'),
    reason="ME function not available in monet_stats"
)
def test_ME():
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 2.5])
    if hasattr(stats, 'ME'):
        result = stats.ME(obs, mod)
        assert isinstance(result, float)
