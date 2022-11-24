"""This module contains the general configuration of the project."""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent.resolve()
BLD = (ROOT / "bld").resolve()

__all__ = ["BLD", "SRC"]
