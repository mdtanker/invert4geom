from __future__ import annotations

import deprecation

from invert4geom import cross_validation


@deprecation.fail_if_not_removed
def test_zref_density_optimal_parameter():
    cross_validation.zref_density_optimal_parameter()


@deprecation.fail_if_not_removed
def test_grav_optimal_parameter():
    cross_validation.grav_optimal_parameter()
