from __future__ import annotations

import importlib.metadata

import invert4geom as m


def test_version():
    assert importlib.metadata.version("invert4geom") == m.__version__
