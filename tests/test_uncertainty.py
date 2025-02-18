from __future__ import annotations

import deepdiff
import numpy as np

from invert4geom import uncertainty


def test_create_lhc():
    """
    test the create_lhc function
    """

    # run the function
    lhc = uncertainty.create_lhc(
        n_samples=3,
        parameter_dict={
            "param1": {"distribution": "uniform", "loc": 0, "scale": 1},
        },
    )

    expected = {
        "param1": {
            "distribution": "uniform",
            "loc": 0,
            "scale": 1,
            "sampled_values": np.array([0.16666667, 0.83333333, 0.5]),
        }
    }

    assert not deepdiff.DeepDiff(
        lhc,
        expected,
        ignore_order=True,
        ignore_numeric_type_changes=True,
        significant_digits=6,
    )
