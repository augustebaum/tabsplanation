"""This module contains the main namespace of tabsplanation."""
# Import the version from _version.py which is dynamically created by setuptoolsscm
# when the project is installed with ``pip install e .``. Do not put it into version
# control!
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


__all__ = ["__version__"]
