"""
Copyright (c) 2023 Matt Tankersley. All rights reserved.

invert4geom: Constrained gravity inversion to recover the geometry of a density
contrast.
"""

from __future__ import annotations

__version__ = "0.10.0"

import logging

log = logging.getLogger(__name__)

log.addHandler(logging.NullHandler())
