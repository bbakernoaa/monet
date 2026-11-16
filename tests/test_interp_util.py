import numpy as np

from monet.util.interp_util import points_to_dataset


def test_scalar_coordinates_keep_fractional_values():
    locations = points_to_dataset(np.array([1, 2]), 12.5)
    np.testing.assert_array_equal(locations.lat.values.ravel(), [12.5, 12.5])
    locations = points_to_dataset(12.5, np.array([1, 2]))
    np.testing.assert_array_equal(locations.lon.values.ravel(), [12.5, 12.5])
