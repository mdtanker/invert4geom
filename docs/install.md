# 🚀 Install

There are 3 main ways to install `invert4geom`. We show them here in order of
simplest to hardest.

## Conda / Mamba

```{warning}
Conda install instructions still to come ...
```

## Pip

You can install via pip, but some of the dependencies include packages which can
only be installed by other means, such as `conda` or `mamba`.

First create a `conda` environment and install the necessary packages into that:

```{note}
`conda` and `mamba` are interchangeable
```

```
mamba create --name invert4geom polartoolkit python=3.11
```

The package `polartoolkit` provides several useful functions used in
`invert4geom`. Since `polartoolkit` has several dependencies that can't be
install with `pip` (mostly `pygmt`), it is easiest to install with `conda`.

activate the environment and use `pip` to install `invert4geom`:

```
conda activate invert4geom
pip install invert4geom
```

```{note}
to install the optional dependencies, use this instead:
`pip install [all]`
```

## Locally

To get a local version of the package and include that in your environment
follow these instructions.

clone the GitHub repository and change directories into it:

```
git clone https://github.com/mdtanker/invert4geom.git
cd invert4geom
```

assuming you have `Python` and `make` installed, as well as `mamba` (install
mamba with `pip install mamba`) installed within your Python environment, run
the following to install the package locally:

```
make create
conda activate invert4geom
make install
```

If you don't have or want `make` or `mamba` installed, you can accomplish the
same with the following:

```
conda create --name invert4geom --yes --force polartoolkit python=3.11
conda activate invert4geom
pip install -e .[all]
```
