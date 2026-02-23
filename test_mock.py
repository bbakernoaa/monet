import sys
from unittest.mock import MagicMock
import types

def mock_if_missing(module_names):
    for name in module_names:
        try:
            __import__(name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            if name not in sys.modules:
                print(f"Mocking {name}")
                m = MagicMock()
                if name == "cartopy":
                    m.__version__ = "0.22.0"
                m.__path__ = []
                # Provide a dummy spec
                m.__spec__ = types.SimpleNamespace(loader=None, origin=None, submodule_search_locations=[])
                sys.modules[name] = m

mock_if_missing([
    "cartopy",
    "cartopy.mpl"
])

import cartopy
import cartopy.mpl.geoaxes as geoaxes
print(f"geoaxes: {geoaxes}")
