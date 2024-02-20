# 🚀 Install

## Online usage (Binder)

See below for the full installation instructions. If instead you'd like to use
this package online, without needing to install anything, check out our
[Binder link](https://mybinder.org/v2/gh/mdtanker/invert4geom/main), which
gives full access the the package in an online environment.

This Binder environment can also be accessed by clicking the Binder icon in any
of the `gallery` or `tutorial` examples.

## Install Python

Before installing *Invert4Geom*, ensure you have Python downloaded. If you don't, I recommend setting up Python
with Miniforge. See the install instructions
[here](https://github.com/conda-forge/miniforge).

## Install *Invert4Geom* Locally

There are 3 main ways to install `invert4geom`. We show them here in order of
simplest to hardest.

### Conda / Mamba

The easiest way to install this package and it's dependencies is with conda or
mamba into a new virtual environment:

    mamba create --name invert4geom --yes --force invert4geom

Activate the environment:

    conda activate invert4geom

### Pip

Instead, you can use pip to install `invert4geom`, but first you need to install
a few dependencies with conda. This is because a few dependencies rely on C packages, which can only be install with conda/mamba and
not with pip.

```{note}
`conda` and `mamba` are interchangeable
```

Create a new virtual environment:

```
mamba create --name invert4geom --yes --force polartoolkit
```

The package `polartoolkit` provides several useful functions used in
`invert4geom`. Since `polartoolkit` has several dependencies that can't be
install with `pip` (mostly `pygmt`), it is easiest to install with `conda` or `mamba`.

activate the environment and use `pip` to install `invert4geom`:

```
conda activate invert4geom
pip install invert4geom
```

```{note}
to install the optional dependencies, use this instead:
`pip install invert4geom[all]`
```

### Development version

You can use pip, with the above created environment, to install the latest
source from GitHub:

    pip install git+https://github.com/mdtanker/invert4tgeom.git

Or you can clone the git repository and install one of two ways:

#### with Make

Or, assuming you have `make` installed, as well as `mamba` (install
mamba with `pip install mamba`) installed within your Python environment, run
the following to install the package locally:

```
git clone https://github.com/mdtanker/invert4geom.git
cd invert4geom
make create
conda activate invert4geom
make install
```

#### manually

```
git clone https://github.com/mdtanker/invert4geom.git
cd invert4geom
conda create --name invert4geom --yes --force polartoolkit python=3.11
conda activate invert4geom
pip install -e .[all]
```
