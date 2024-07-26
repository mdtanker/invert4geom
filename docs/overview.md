# 🔎 Overview

`Invert4geom` provides a series of tools for conducting a specific style of gravity inversion.
Many gravity inversions aim to model the density distribution of the subsurface.
These are commonly used to identify bodies of anomalous densities, such as igneous intrusions or ore deposits.
The typical way these are performed is to _discretize_ the subsurface into a series of finite volumes, such as cubes or prisms, where the shape of the volumes doesn't change.
The inversion then alters the density values of each of these volumes to match the observed gravity anomaly.
In these inversions the _density_ values changes, while the _geometry_ of the volumes remains unchanged.
These types of inversions may be referred to as _density inversions_.
Here, instead, we are performing _geometric inversions_.

Geometric inversions are essentially the opposite.
The density values of the volumes in the discretized model remain unchanged, while their geometry is
altered.
Here we use layers of vertical right-rectangular prisms and alter their _tops_ and _bottoms_ during the inversion.
Typically use cases for these style of inversion are modeling the topography of the Moho, the contact between sediment and basement, or the shape of the seafloor in locations where it is not easily mapped.

Currently, this package is only intended to perform inversions using right rectangular prisms.
Other types of volumes, such as tesseroids, are currently not implemented.
If you have interest in these, please raise an issue in GitHub.

Much of this software was developed as part of my Ph.D. thesis.
For detailed description of the theory and implementation of this inversion, as well as many synthetic tests and a real-world application to modelling bathymetry, see chapter 3 and 4 of my thesis, available [here](https://doi.org/10.26686/wgtn.24408304).
The code was originally included in [this GitHub repository](https://github.com/mdtanker/RIS_gravity_inversion), but much of it has been migrated here.

## Conventions

This package has a few conventions which need to be followed for the code to work.
1) Coordinates names for gravity data, topography, and _a priori_ constraints need to be projected units (meters) and named `easting`, `northing`, and  `upward`.
If you use names such as `x`, `y`, and `z`, please rename them.
2) The observed gravity data, whether its a Free Air anomaly, gravity disturbance, or some other form of anomaly, needs to be in a column called `gravity_anomly` in the gravity dataframe. The forward gravity of your starting model (set equal to 0 if you're starting model is flat) needs to be in a column called `starting_gravity`.
3) Similarly, the gravity misfit must be in a column called `misfit`, the regional component in a column named `reg`, and the residual component in a column named `res`.
If you use the regional separation functions in `regional.py`, these names will automatically be used.
4) Gravity data must be in projected coordinates and gridded!
If your data is in geographic coordinates (latitude/longitude) and consists of only the observations points, it must be gridded into an Xarray DataArray with projected coordinates. Please see the Python packages `Harmonica` for a geophysically-informed method of gridding discrete gravity data (the Equivalent Source technique), and `Verde` for projecting data.

## Modules

**Invert4Geom** consists of 7 modules:

### Inversion

This contains the core tools for performing the inversion.
These tools allow for creating the Jacobian matrix, performing the least squares solution, running the inversion, determining when to end the inversion, and updating the misfit and gravity values between each iteration.

## Cross Validation

This module contains the code necessary for performing the various cross-validation routines.
These routines are split into 2 categories; _gravity_ and _constraints_ cross validation.

### Gravity cross validations

The _gravity cross validations_ are those which split the gravity data into testing and training sets, perform the inversion with the training set, and compare the forward gravity of the inverted topography with the un-used testing data.
This is a Generalized Cross Validation, specifically a hold-out cross-validation, as described in [Uieda & Barbosa (2017)](https://academic.oup.com/gji/article-lookup/doi/10.1093/gji/ggw390).
This is used here for estimated the optimal regularization damping parameter during the inversion.

### Constraint cross validations

The _constraint cross validations_ are those which use _apriori_ points of known elevation of the surface of in interest to determine optimal inversion
parameters.
The inversion is performed without including these constraint points, and the inverted surface is compared with the elevations of these constraints points to give a score.
This style of validation is used here for estimating the optimal reference level and density contrast of the surface of interest.

## Regional

This module contains tools for estimating the regional gravity field.
In many styles of inversions, the regional component of the gravity misfit should be removed to help isolate the gravity effects of the layer of interest.
This is common in inversions for topography, bathymetry, or sedimentary basins.
We provide several methods of estimating the regional component.

1. Low pass filtering the data, 2) fitting a surface to the data with a user defined trend, 3) fitting a set of deep equivalent sources to the data and    predicting the gravity effect of those sources, and 4) using _apriori_ constraints to determine the regional field, by assuming the residual component is low at the those points.

## Optimization

Functions for performing optimizations.
This uses the package _Optuna_ which given a parameter space, an objective function, and a sampling routine, will perform optimization to minimize (or maximize) the object function.
This is used in the **Regional** module for finding the optimal regional separation methods.

## Plotting

Various function for plotting maps and graphs used throughout the other modules.

## Utils

Utilities used throughout the other modules.

## Synthetic

This module has tools for creating synthetic topography and contaminating data with random Gaussian noise.
This is mostly used in testing and in the User Guide notebooks.

## Uncertainty

This module contains the tools needed for performing a stochastic uncertainty analysis on the inversion.
This works be performing many inversions, each with slightly different chosen parameter values or input data values, and comparing the results.
Input data is sampled from a distributions of its estimated uncertainty, and input parameter values are sampled using a Latin Hypercube approach.
