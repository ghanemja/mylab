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


def test_dec2bin_dec2hex_negative_and_fractional(m):
    # negative numbers are written in two's complement (fewest of 8/16/32/64 bits); fractions are dropped.
    # One copy wrote '-101', the other raised an error.
    assert m.str("dec2bin(-5)") == "11111011"
    assert m.str("dec2bin(int8(-3))") == "11111101"
    assert m.str("dec2bin(-129)") == "1111111101111111"
    assert m.str("dec2hex(-1)") == "FF"
    assert m.str("dec2hex(-200)") == "FF38"
    assert m.str("dec2bin(2.5)") == "10"
    assert m.num("size(dec2bin([-1 3]))").tolist() == [[2, 8]]
    m.run("b = dec2bin([1 2 3]);")
    assert m.str("b(3, :)") == "11"  # one copy only converted the first element


def test_bin2dec_hex2dec_arrays_literals_and_bad_text(m):
    # char matrices and cells give a column; invalid text is NaN (one copy summed the rows or raised)
    assert m.num("bin2dec(['10';'11'])").tolist() == [[2], [3]]
    assert m.num("hex2dec({'A','1F'})").tolist() == [[10], [31]]
    assert np.isnan(m.num("hex2dec('G')")) and np.isnan(m.num("bin2dec('12')"))
    # the 0x / 0b prefixes and the s8 ... u64 suffixes of the number literal syntax
    assert m.num("hex2dec('0xFF')") == 255 and m.num("bin2dec('0b101')") == 5
    assert m.num("hex2dec('FFs8')") == -1 and m.num("hex2dec('FFu8')") == 255
    assert m.num("bin2dec('11111011s8')") == -5
    assert np.isnan(m.num("hex2dec('1FFs8')"))


@pytest.mark.parametrize("call", ["contains(['ab';'cd'], 'a')", "extractAfter(['ab';'cd'], 1)"])
def test_text_functions_reject_char_matrices(m, call):
    # a char matrix is not text for these functions (one copy treated it row by row, or as one string)
    with pytest.raises(MatlabError, match="character matrix"):
        m.run(call)


def test_extract_from_string_without_match_is_missing(m):
    # string input without a match gives <missing>; char and cell input give ''
    assert m.num('ismissing(extractAfter("abc", "x"))') and m.num('ismissing(extractBefore("abc", "x"))')
    assert m.str("extractAfter('abc', 'x')") == ""


@pytest.mark.parametrize("name", ["contains", "startsWith", "endsWith"])
def test_text_predicates_options_are_strict(m, name):
    assert m.num(f"{name}('ABC', 'b', 'IgnoreCase', true)") == (name == "contains")
    assert m.num(f"{name}('ABC', {{'x', 'a', 'c'}}, 'ignorecase', 1)")
    with pytest.raises(MatlabError, match="Bogus"):
        m.run(f"{name}('abc', 'b', 'Bogus', 1)")


def test_anynan_allfinite_accept_char(m):
    # char input is allowed and never NaN/Inf (one copy raised an error for it)
    assert not m.num("anynan('abc')") and m.num("allfinite('abc')")
    assert m.num("anynan([1 NaN])") and not m.num("allfinite([1 Inf])")
    with pytest.raises(MatlabError):
        m.run("anynan({1})")


def test_e_takes_sizes(m):
    # e(n) and e(m, n) are arrays (one copy ignored its arguments and always returned the scalar)
    assert m.num("e") == pytest.approx(np.e)
    assert m.num("size(e(2))").tolist() == [[2, 2]] and m.num("size(e(1, 3))").tolist() == [[1, 3]]
    assert m.str("class(e(2, 'single'))") == "single"
