import sys
from unittest.mock import MagicMock

# Mock the monet_regrid module as it is a conda-only dependency
# and not required for the tests I am running.
sys.modules["monet_regrid"] = MagicMock()
