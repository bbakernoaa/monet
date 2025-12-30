from monet.util import stats


def test_stats_basic():
    import pandas as pd

    df = pd.DataFrame({"Obs": [1, 2, 3, 4, 5], "CMAQ": [1, 2, 2, 4, 5]})
    minval = 1
    maxval = 5
    result = stats.stats(df, minval, maxval)
    assert result is not None
    for key in ["N", "Obs", "Mod", "MB", "R", "IOA", "RMSE", "NMB", "POD", "FAR"]:
        assert key in result
