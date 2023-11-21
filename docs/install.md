# Install

```{warning}
Install instructions still to come ...
```

For now, clone the GitHub repository and change directories into it:

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
conda create --name invert4geom --yes --force antarctic-plots python=3.11
conda activate invert4geom
pip install -e .[viz,test,dev,docs]
```
