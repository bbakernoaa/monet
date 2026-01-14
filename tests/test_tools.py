import numpy as np
import pandas as pd
import pytest

from monet.util.tools import get_epa_region_df, get_giorgi_region_df, search_listinlist


def test_search_listinlist_correctness():
    """Tests the correctness of the search_listinlist function."""
    array1 = np.array([1, 2, 3, 4, 5])
    array2 = np.array([3, 5, 6, 7, 8])
    index1, index2 = search_listinlist(array1, array2)
    np.testing.assert_array_equal(index1, np.array([2, 4]))
    np.testing.assert_array_equal(index2, np.array([0, 1]))


@pytest.mark.skipif("benchmark" not in dir(), reason="pytest-benchmark not available")
def test_search_listinlist_benchmark(benchmark):
    """Benchmarks the search_listinlist function."""
    array1 = np.arange(1000)
    array2 = np.arange(500, 1500)
    benchmark(search_listinlist, array1, array2)


def test_get_giorgi_region_df():
    """Tests the vectorized get_giorgi_region_df function."""
    # Test data including points inside a region, outside any region,
    # and on a boundary.
    data = {
        "latitude": [40, 0, 30],
        "longitude": [-95, -150, -83],
    }
    df = pd.DataFrame(data)

    # Expected results:
    # 1. Central North America (CNA)
    # 2. No region (should be None/NaN)
    # 3. Central America (CAM) - on the boundary
    expected_indices = [7.0, np.nan, 5.0]
    expected_acros = ["CNA", None, "CAM"]

    result_df = get_giorgi_region_df(df)

    # Check that the columns were added
    assert "GIORGI_INDEX" in result_df.columns
    assert "GIORGI_ACRO" in result_df.columns

    # Check the values
    np.testing.assert_array_equal(
        result_df["GIORGI_INDEX"].values, np.array(expected_indices)
    )
    assert result_df["GIORGI_ACRO"].tolist() == expected_acros


def test_get_epa_region_df():
    """Tests the vectorized get_epa_region_df function."""
    # Test data including points inside a region, outside any region,
    # and on a boundary.
    data = {
        "latitude": [42.0, 30.0, 40.0],
        "longitude": [-90.0, -100.0, -125.0],
    }
    df = pd.DataFrame(data)

    # Expected results:
    # 1. Region 5 (R5)
    # 2. Region 6 (R6)
    # 3. No region (should be None/NaN)
    expected_indices = [5.0, 6.0, np.nan]
    expected_acros = ["R5", "R6", None]

    result_df = get_epa_region_df(df)

    # Check that the columns were added
    assert "EPA_INDEX" in result_df.columns
    assert "EPA_ACRO" in result_df.columns

    # Check the values
    np.testing.assert_array_equal(
        result_df["EPA_INDEX"].values, np.array(expected_indices)
    )
    assert result_df["EPA_ACRO"].tolist() == expected_acros
