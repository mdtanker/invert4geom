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
2) The observed gravity data, whether its a Free Air anomaly, gravity disturbance, or some other form of anomaly, needs to be in a column called `gravity_anomaly` in the gravity dataframe. The forward gravity of your starting model (set equal to 0 if you're starting model is flat) needs to be in a column called `starting_gravity`.
3) Similarly, the gravity misfit must be in a column called `misfit`, the regional component in a column named `reg`, and the residual component in a column named `res`.
If you use the regional separation functions in `regional.py`, these names will automatically be used.
4) Gravity data must be in projected coordinates and gridded!
If your data is in geographic coordinates (latitude/longitude) and consists of only the observations points, it must be gridded into an Xarray DataArray with projected coordinates. Please see the Python packages `Harmonica` for a geophysically-informed method of gridding discrete gravity data (the Equivalent Source technique), and `Verde` for projecting data.

## Modules

**Invert4Geom** consists of 8 modules:

### Inversion

The {mod}`.inversion` module contains the core tools for performing the inversion.
These tools allow for creating the Jacobian matrix, performing the least squares solution, running the inversion, determining when to end the inversion, and updating the misfit and gravity values between each iteration. The core function for running an inversion is {func}`~.inversion.run_inversion`. A function is also provided ({func}`~.inversion.run_inversion_workflow`) which allows the entire inversion workflow to be run from one function. This workflow includes creating the starting topography, creating the starting prism model, calculating for the forward gravity of the prism model, calculating the misfit and regional/residual components, running a standard inversion, or running cross-validation for determining the optimal damping, reference level, or density contrast values.

### Cross Validation

The {mod}`.cross_validation` module contains the code necessary for calculating cross-validation scores. This include scores for regional separation ({func}`~.cross_validation.regional_separation_score`), fitting equivalent sources ({func}`~.cross_validation.eq_sources_score`), inversion performance based on gravity data ({func}`~.cross_validation.grav_cv_score`) and based on constraint points ({func}`~.cross_validation.constraints_cv_score`). These scores are used in the {mod}`.optimization` module for hyperparameter optimization for regional separation parameters, damping, reference level, and density contrast values.
It also contains functions for separating data into testing and training sets ({func}`~.cross_validation.resample_with_test_points`, {func}`~.cross_validation.random_split_test_train`) and K-folds ({func}`~.cross_validation.split_test_train`).


### Regional

The {mod}`.regional` module contains tools for estimating the regional gravity field.
In many styles of inversions, the regional component of the gravity misfit should be removed to help isolate the gravity effects of the layer of interest.
This is common in inversions for topography, bathymetry, or sedimentary basins.
We provide several methods of estimating the regional component.

1) Using a constant value for the regional field ({func}`~.regional.regional_constant`)
2) Low pass filtering the data ({func}`~.regional.regional_filter`)
3) fitting a surface to the data with a user defined trend ({func}`~.regional.regional_trend`)
4) fitting a set of deep equivalent sources to the data and predicting the gravity effect of those sources ({func}`~.regional.regional_eq_sources`)
5) using _apriori_ constraints to determine the regional field, by assuming the residual component is low at the those points ({func}`~.regional.regional_constraints`)

The optimal parameter values associated with this functions; filter width, trend order, source depth, damping values, and various gridding parameters, can all be chosen via hyperparameter optimization routines within the {mod}`.optimization` module.

### Optimization

The {mod}`.optimization` module contains tools for performing hyperparameter optimizations using the Python package {mod}`optuna`. The main optimizations we provide are:

**Fitting equivalent sources**
* {func}`~.optimization.optimize_eq_source_params`

**Regional field estimation parameters**
* {func}`~.optimization.optimize_regional_filter`
* {func}`~.optimization.optimize_regional_trend`
* {func}`~.optimization.optimize_regional_eq_sources`
* {func}`~.optimization.optimize_regional_constraint_point_minimization`

**Inversion parameters**
* {func}`~.optimization.optimize_inversion_damping`
* {func}`~.optimization.optimize_inversion_zref_density_contrast`
* {func}`~.optimization.optimize_inversion_zref_density_contrast_kfolds`

### Uncertainty

The {mod}`.uncertainty` module contains the tools needed for performing stochastic uncertainty analyses of various portions Invert4Geom. This includes estimating the uncertainty of
1) creating a topography model from interpolation of *a priori* measurements of topography ({func}`~.uncertainty.starting_topography_uncertainty`)
2) the regional component of gravity misfit ({func}`~.uncertainty.regional_misfit_uncertainty`)
3) the final inverted topography model ({func}`~.uncertainty.full_workflow_uncertainty_loop`)

The stochastic approach works be performing the tasks (interpolation, regional estimation, or inversion) a range of time, each with slightly different chosen parameter values or input data values. This results in an ensemble of results. The cell-wise standard deviation (optionally weighted) is used as an estimate for uncertainty of the result.

### Plotting

The {mod}`.plotting` module contains various function for plotting maps and graphs used throughout the other modules.

### Utils

The {mod}`.utils` module contains functions which are primarily used throughout the other modules.

### Synthetic

The {mod}`.synthetic` module has tools for creating synthetic topography and contaminating data with random Gaussian noise. This is mostly used in testing and in the User Guide notebooks.
