import logging

from ._version import version as __version__

__all__ = ["__version__"]

logger = logging.getLogger(__name__)

from .cross_validation import (  # noqa: E402
    add_test_points,
    eq_sources_score,
    kfold_df_to_lists,
    random_split_test_train,
    regional_separation_score,
    remove_test_points,
    split_test_train,
)
from .inversion import (  # noqa: E402
    DatasetAccessorInvert4Geom,
    Inversion,
    create_data,
    create_model,
    run_inversion_workflow,
)
from .optimization import (  # noqa: E402
    optimal_buffer,
    optimize_eq_source_params,
    optimize_regional_constraint_point_minimization,
    optimize_regional_eq_sources,
    optimize_regional_filter,
    optimize_regional_trend,
)
from .plotting import (  # noqa: E402
    plot_2_parameter_scores,
    plot_2_parameter_scores_uneven,
    plot_edge_effects,
    plot_inversion_grav_results,
    plot_inversion_iteration_results,
    plot_inversion_results,
    plot_inversion_topo_results,
    plot_latin_hypercube,
    plot_optimization_combined_slice,
    plot_optuna_figures,
    plot_prism_layers,
    plot_sampled_projection_2d,
    plot_scores,
    plot_stochastic_results,
)
from .synthetic import (  # noqa: E402
    contaminate,
    load_bishop_model,
    load_synthetic_model,
    synthetic_topography_regional,
    synthetic_topography_simple,
)
from .uncertainty import (  # noqa: E402
    full_workflow_uncertainty_loop,
    merged_stats,
    randomly_sample_data,
    regional_misfit_uncertainty,
)
from .utils import (  # noqa: E402
    create_topography,
    filter_grid,
    gravity_decay_buffer,
    normalized_mindist,
    optimal_spline_damping,
    rmse,
    sample_grids,
)
