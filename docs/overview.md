# Overview

`Invert4geom` provides a series of tools for conducting a specific style of gravity inversion.
Many gravity inversions aim to model the density distribution of the subsurface.
These are commonly used to identify bodies of anomalous densities, such as igneous intrusions or ore deposits.
The typical way these are performed is to _discretize_ the subsurface into a series of finite volumes, such as cubes or prisms, where the shape of the volumes doesn't change.
The inversion then alters the density values of each of these volumes to match the observed gravity anomaly.
In these inversions the _density_ values changes, while the _geometry_ of the volumes remains unchanged.
These types of inversions may be referred to as _density inversions_.
Here, instead, we are performing _geometric inversions_.

Geometric inversions are essentially the opposite.
The density values of the volumes in the discretized model remain unchanged, while their geometry is altered.
Here we use a layer of either vertical right-rectangular prisms or tesseroids (spherical prisms) and alter their _tops_ and _bottoms_ during the inversion.
Typically use cases for these style of inversion are modeling the topography of the Moho, the contact between sediment and basement, or the shape of the seafloor in locations where it is not easily mapped.

Much of this software was developed as part of my Ph.D. thesis.
For detailed description of the theory and implementation of this inversion, as well as many synthetic tests and a real-world application to modelling bathymetry, see chapter 3 and 4 of my thesis, available [here](https://doi.org/10.26686/wgtn.24408304).
The code was originally included in [this GitHub repository](https://github.com/mdtanker/RIS_gravity_inversion), but much of it has been migrated here.

## Conventions

This package has a few conventions which need to be followed for the code to work.
1) Coordinates names for gravity data, topography, and _a priori_ constraints need to be projected units (meters) and named `easting`, `northing`, and  `upward`.
If you use names such as `x`, `y`, and `z`, please rename them.
2) Gravity data is expected to be gridded (interpolated), and in the form of an xarray Dataset with variable `gravity_anomaly`, defined the observed gravity data, whether its a Free Air anomaly, gravity disturbance, or some other form of anomaly, and variable `upward`, defined the elevation of the observation points. It should have coordinates `easting` and `northing`, in meters. If your data is in geographic coordinates (latitude/longitude), see python package `Verde` for reprojecting. If your data consist of point-observations (not interpolated), see the equivalent source interpolation tools of the Python package `Harmonica` for a geophysically-informed method of gridding the data.
3) Prior to inversion, the gravity dataset must also have variables `misfit`, `reg`, and `res`, which define the gravity misfit (difference between `gravity_anomaly` and the forward gravity of the starting model), and it's regional and residual components. If you use the regional separation functions in `regional.py`, these names will automatically be used.
