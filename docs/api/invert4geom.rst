.. _api:

API Reference
=============

.. automodule:: invert4geom

.. currentmodule:: invert4geom


Model creation
--------------
Create a topography grid (or provide your own) and convert it into an initial prism or tesseroid model for the inversion.

.. autosummary::
    :toctree: generated/

    create_topography
    create_model
    normalized_mindist


Gravity data processing
-----------------------
Process gravity data and prepare it for the inversion.

.. autosummary::
    :toctree: generated/

    create_data
    DatasetAccessorInvert4Geom.forward_gravity

Regional gravity misfit estimation.

.. autosummary::
    :toctree: generated/

    DatasetAccessorInvert4Geom.regional_separation
    DatasetAccessorInvert4Geom.regional_constant
    DatasetAccessorInvert4Geom.regional_filter
    DatasetAccessorInvert4Geom.regional_trend
    DatasetAccessorInvert4Geom.regional_eq_sources
    DatasetAccessorInvert4Geom.regional_constraints
    DatasetAccessorInvert4Geom.regional_constraints_cv


:class:`xarray.Dataset` properties
----------------------------------
Properties of both the gravity and model :class:`xarray.Dataset` can be accessed via the :code:`inv` Accessor. For example, to return just the inner region of a dataset :code:`ds`, use :code:`ds.inv.inner`. These are intended to be used for both the the gravity and the model datasets.

.. autosummary::
    :toctree: generated/

     DatasetAccessorInvert4Geom.df
     DatasetAccessorInvert4Geom.inner_df
     DatasetAccessorInvert4Geom.masked_df
     DatasetAccessorInvert4Geom.masked


Inversion class
-------------------
Class for attributes and methods for performing a gravity inversion.

.. autosummary::
    :toctree: generated/

    Inversion

Function to run a full inversion workflow.

.. autosummary::
    :toctree: generated/

    run_inversion_workflow


Cross-Validation
----------------
Split data (gravity and constraints) into training and test sets.

.. autosummary::
    :toctree: generated/

    add_test_points
    remove_test_points
    split_test_train
    random_split_test_train
    kfold_df_to_lists

Calculate cross-validation scores.

.. autosummary::
    :toctree: generated/

    eq_sources_score
    regional_separation_score
    Inversion.grav_cv_score
    Inversion.constraints_cv_score


Optimization
------------
Run optimization routines with :mod:`optuna` to find optimal parameters.

Routines for optimal regional field estimation parameters.

.. autosummary::
    :toctree: generated/

    optimize_regional_filter
    optimize_regional_trend
    optimize_regional_eq_sources
    optimize_regional_constraint_point_minimization

Routines for optimal interpolation parameters.

.. autosummary::
    :toctree: generated/

    optimize_eq_source_params
    optimal_spline_damping

Routines for optimal inversion parameters.

.. autosummary::
    :toctree: generated/

    Inversion.optimize_inversion_damping
    Inversion.optimize_inversion_zref_density_contrast
    Inversion.optimize_inversion_zref_density_contrast_kfolds
    optimal_buffer


Uncertainty
-----------
Monte Carlo simulations to estimate the spatially variable uncertainties for the inversion and data processing.

.. autosummary::
    :toctree: generated/

    full_workflow_uncertainty_loop
    merged_stats
    randomly_sample_data
    regional_misfit_uncertainty

Plotting
--------
Plot gravity data.

.. autosummary::
    :toctree: generated/

    DatasetAccessorInvert4Geom.plot_observed
    DatasetAccessorInvert4Geom.plot_anomalies

Plot model.

.. autosummary::
    :toctree: generated/

    DatasetAccessorInvert4Geom.plot_model
    plot_prism_layers

Plot inversion results.

.. autosummary::
    :toctree: generated/

    Inversion.plot_convergence
    Inversion.plot_dynamic_convergence
    plot_inversion_topo_results
    plot_inversion_grav_results
    plot_inversion_iteration_results
    plot_inversion_results

Plot optimization and cross-validation results.

.. autosummary::
    :toctree: generated/

    plot_2_parameter_cv_scores
    plot_2_parameter_cv_scores_uneven
    plot_cv_scores
    plot_optimization_combined_slice
    plot_optuna_figures

Plot parameter sampling and uncertainty results.

.. autosummary::
    :toctree: generated/

    plot_stochastic_results
    plot_latin_hypercube
    plot_sampled_projection_2d


Other plotting functions.

.. autosummary::
    :toctree: generated/

    plot_edge_effects
