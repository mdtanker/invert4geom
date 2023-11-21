# invert4geom

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

**Invert4geom** is a Python library for performing 3D geometric gravity
inversions, where the aim is to recover the geometry of a density contrast.

Typical use cases include modeling the topography of the Moho, the
sediment-basement contact, or bathymetry. These density contrasts are
represented by a layer of vertical right-rectangular prisms. Since we use
vertical prisms, they don't take the curvature of the Earth into account. For
large-scale applications, such as continental studies, it would be better to use
tesseroids instead of prisms.

See the [overview](overview.md) for further description of this package and what
it can be used for.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/mdtanker/invert4geom/workflows/CI/badge.svg
[actions-link]:             https://github.com/mdtanker/invert4geom/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/invert4geom
[conda-link]:               https://github.com/conda-forge/invert4geom-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/mdtanker/invert4geom/discussions
[pypi-link]:                https://pypi.org/project/invert4geom/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/invert4geom
[pypi-version]:             https://img.shields.io/pypi/v/invert4geom
[rtd-badge]:                https://readthedocs.org/projects/invert4geom/badge/?version=latest
[rtd-link]:                 https://invert4geom.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
