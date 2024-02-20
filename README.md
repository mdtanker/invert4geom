<h1 align="center">Invert4geom</h1>
<h2 align="center">3D geometric gravity inversions
</h2>

<p align="center">
<a href="https://invert4geom.readthedocs.io"><strong>Documentation Link</strong></a>
</p>

<!-- SPHINX-START1 -->

<p align="center">
<a href="https://mybinder.org/v2/gh/mdtanker/invert4geom/main">
 <img src="https://mybinder.org/badge_logo.svg" alt="Binder link"></a>
 </p>

<p align="center">
<a href=https://pypi.org/project/invert4geom/>
<img src=https://img.shields.io/pypi/v/invert4geom
alt="Latest version on PyPI"
/>
</a>
<a href=https://github.com/conda-forge/invert4geom-feedstock>
<img src=https://img.shields.io/conda/vn/conda-forge/invert4geom
alt="Latest version on conda-forge"
/>
</a>
<a href=https://pypi.org/project/invert4geom/>
<img src=https://img.shields.io/pypi/pyversions/invert4geom
alt="Compatible Python versions"
/>

<p align="center">
<a href=https://app.codecov.io/github/mdtanker/invert4geom>
<img src=https://codecov.io/github/mdtanker/invert4geom/badge.svg?
alt="Test coverage status"
/>
</a>
<a href=https://invert4geom.readthedocs.io/en/latest/?badge=latest>
<img src=https://readthedocs.org/projects/invert4geom/badge/?version=latest&style=flat-square
alt='Documentation Status'
/>
<a href=https://github.com/mdtanker/invert4geom/actions>
<img src=https://github.com/mdtanker/invert4geom/workflows/CI/badge.svg
alt="Actions status"
/>
<p align="center">
<a href=https://github.com/mdtanker/invert4geom/discussions>
<img src=https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
alt="GitHub discussion"
/>
<!-- </a>
<a href="https://zenodo.org/badge/latestdoi/475677039">
<img src="https://zenodo.org/badge/475677039.svg?style=flat-square"
alt="Zenodo DOI"
/> -->
</a>
 </p>

<!-- <p align="center">
<img src="docs/figures/cover_fig.png"/>
</p> -->

<!-- SPHINX-END1 -->

![](docs/figures/cover_fig.png)

<!-- <p align="center">
<img src="docs/figures/cover_fig.png" width="400"/>
</p> -->

<!-- SPHINX-START2 -->

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
[codecov-badge]:            https://codecov.io/github/mdtanker/invert4geom/badge.svg?
[codecov-link]:             https://app.codecov.io/github/mdtanker/invert4geom
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/mdtanker/invert4geom/discussions
[pypi-link]:                https://pypi.org/project/invert4geom/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/invert4geom
[pypi-version]:             https://img.shields.io/pypi/v/invert4geom
[rtd-badge]:                https://readthedocs.org/projects/invert4geom/badge/?version=latest
[rtd-link]:                 https://invert4geom.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

<!-- SPHINX-END2 -->
