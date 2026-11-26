"""Results that two copies of the same built-in disagreed on, second round (see test_registry_unique.py).

Each test pins the behaviour that was kept when the copies were merged into one implementation.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def test_random_generators_keep_class_and_like(m):
    # the copy that was dropped ignored the class name and 'like'
    assert m.str("class(gamrnd(2, 3, 2, 2, 'single'))") == "single"
    assert m.str("class(betarnd(2, 3, 'like', single(1)))") == "single"
    assert m.num("size(wblrnd(1, 2, [2 3]))").tolist() == [[2, 3]]


def test_unidrnd_needs_whole_numbers(m):
    # N must be a positive integer: a fractional N gives NaN rather than a draw from 1:floor(N)
    v = m.num("unidrnd([0 1.5 -1 3])")
    assert np.isnan(v[0, :3]).all() and v[0, 3] in (1, 2, 3)
    assert np.isnan(m.num("unidrnd(2.5, 1, 4)")).all()


def test_lognrnd_zero_sigma_is_a_point_mass(m):
    # sigma must be nonnegative; sigma = 0 gives exp(mu), only a negative sigma is NaN
    v = m.num("lognrnd([2 0 0], [0 0 -1])")
    assert v[0, 0] == pytest.approx(np.exp(2)) and v[0, 1] == 1 and np.isnan(v[0, 2])


@pytest.mark.parametrize("fn", ["geomean", "harmmean"])
def test_geo_harm_mean_empty_shapes(m, fn):
    # like mean: [] gives NaN, other empty arrays reduce along the first non-singleton dimension
    assert np.isnan(m.num(f"{fn}([])"))
    assert m.num(f"size({fn}(zeros(0,3)))").tolist() == [[1, 3]]
    assert np.isnan(m.num(f"{fn}(zeros(0,3))")).all()
    assert m.num(f"size({fn}(zeros(3,0)))").tolist() == [[1, 0]]
    assert m.num(f"size({fn}(zeros(0,3), 2))").tolist() == [[0, 1]]


@pytest.mark.parametrize("fn, value", [("geomean", 2.0), ("harmmean", 1.6)])
def test_geo_harm_mean_options(m, fn, value):
    # the dropped copy rejected 'omitnan' and 'all'
    assert m.num(f"{fn}([1 NaN 4], 'omitnan')") == pytest.approx(value)
    assert np.isnan(m.num(f"{fn}([1 NaN 4])"))
    assert m.num(f"{fn}([1 2; 4 1], 'all')") == pytest.approx(m.num(f"{fn}([1 2 4 1])"))


def test_random_generators_check_sizes(m):
    with pytest.raises(MatlabError, match="Not enough input"):
        m.run("gamrnd(2)")
    with pytest.raises(MatlabError, match="size"):
        m.run("gamrnd([1 2], [1 2 3])")
