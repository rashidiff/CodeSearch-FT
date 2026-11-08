"""CodeSearch-FT application package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("codesearch-ft")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
