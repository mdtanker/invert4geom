from __future__ import annotations

import importlib.metadata

import invert4geom


def test_version():
    assert importlib.metadata.version("invert4geom") == invert4geom.__version__
