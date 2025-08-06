"""
Copyright (c) 2023 Matt Tankersley. All rights reserved.

invert4geom: Constrained gravity inversion to recover the geometry of a density
contrast.
"""

from __future__ import annotations

import logging

from ._version import version as __version__

__all__ = ["__version__"]

logger = logging.getLogger(__name__)
