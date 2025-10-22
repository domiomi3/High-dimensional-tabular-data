# high_tab/__init__.py
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("high_tab")
except PackageNotFoundError:
    __version__ = "0.0.0"

from . import utils, preprocessors, models
__all__ = ["__version__", "utils", "preprocessors", "models"]
