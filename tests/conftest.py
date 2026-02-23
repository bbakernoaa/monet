import sys
from unittest.mock import MagicMock

sys.modules["monet_regrid"] = MagicMock()
sys.modules["cartopy"] = MagicMock()
sys.modules["cartopy.crs"] = MagicMock()
sys.modules["cartopy.feature"] = MagicMock()
sys.modules["cartopy.mpl.gridliner"] = MagicMock()
sys.modules["cartopy.io.shapereader"] = MagicMock()
sys.modules["pydecorate"] = MagicMock()
sys.modules["xregrid"] = MagicMock()
sys.modules["monet-stats"] = MagicMock()
sys.modules["pytspack"] = MagicMock()
sys.modules["mpi4py"] = MagicMock()
