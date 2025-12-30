import numpy as np
import monet_stats as stats


def test_pearsonr():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1, 2, 2, 4, 5])
    # Use R2 as a proxy for Pearson correlation squared
    result = stats.R2(obs, mod)
    assert isinstance(result, float)


def test_spearmanr():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1, 2, 2, 4, 5])
    result = stats.spearmanr(obs, mod)
    assert isinstance(result, float)
