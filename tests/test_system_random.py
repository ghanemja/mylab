"""Random number streams: after rng(seed) the numbers are those of MATLAB's Mersenne Twister.

The expected values are the ones MATLAB prints (they appear in its published examples for the
default seed), and the standard reference output of the twister itself.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError

# randn(10, 1) after rng default, as MATLAB shows it in format long
RANDN_LONG = [0.537667139546100, 1.833885014595086, -2.258846861003648, 0.862173320368121, 0.318765239858981,
              -1.307688296305273, -0.433592022305684, 0.342624466538650, 3.578396939725760, 2.769437029884877]
# the next values, to four decimals; number 52 is the first that needs a second uniform number (a wedge)
RANDN_SHORT = [-1.3499, 3.0349, 0.7254, -0.0631, 0.7147, -0.2050, -0.1241, 1.4897, 1.4090, 1.4172,
               0.6715, -1.2075, 0.7172, 1.6302, 0.4889, 1.0347, 0.7269, -0.3034, 0.2939, -0.7873,
               0.8884, -1.1471, -1.0689, -0.8095, -2.9443, 1.4384, 0.3252, -0.7549, 1.3703, -1.7115,
               -0.1022, -0.2414, 0.3192, 0.3129, -0.8649, -0.0301, -0.1649, 0.6277, 1.0933, 1.1093,
               -0.8637, 0.0774, -1.2141, -1.1135, -0.0068, 1.5326, -0.7697, 0.3714, -0.2256, 1.1174]


def test_rand_matches_matlab(m):
    m.run("rng(0, 'twister'); a = rand; rng('default'); b = rand(1, 5); rng default; c = rand(2, 2);")
    assert abs(m.num("a").item() - 0.814723686393179) < 1e-15
    assert np.allclose(m.num("b"), [0.814723686393179, 0.905791937075619, 0.126986816293506,
                                    0.913375856139019, 0.632359246225410], atol=1e-15)
    # arrays are filled column by column
    assert np.allclose(m.num("c"), [[0.814723686393179, 0.126986816293506],
                                    [0.905791937075619, 0.913375856139019]], atol=1e-15)
    m.run("rng(1); d = rand;")
    assert abs(m.num("d").item() - 0.417022004702574) < 1e-15


def test_rand_is_the_reference_twister(m):
    # the generator's reference program prints 3499211612 as the first 32-bit output for its default seed;
    # a double uses the top 27 bits of one output and the top 26 of the next
    m.run("rng default; r = rand;")
    first, second = 3499211612, 581869302
    assert m.num("r").item() == ((first >> 5) * 67108864 + (second >> 6)) / 2 ** 53


def test_a_new_session_starts_like_rng_default(m):
    assert abs(m.num("rand").item() - 0.814723686393179) < 1e-15


def test_randn_matches_matlab(m):
    m.run("rng default; x = randn(10, 1); y = randn(1, 50);")
    assert np.allclose(m.num("x").ravel(), RANDN_LONG, atol=5e-15, rtol=0)
    assert np.allclose(m.num("y").ravel(), RANDN_SHORT, atol=5.1e-5, rtol=0)
    m.run("rng(0); z = randn(3);")
    assert np.allclose(m.num("z")[:, 0], RANDN_LONG[:3], atol=5e-15)       # column-major fill


def test_randn_stream_does_not_depend_on_block_sizes(m):
    m.run("rng(7); a = randn(1, 20000); rng(7); b = [randn(1, 51), randn, randn(1, 7000), randn(1, 12948)];"
          "rng(7); c = zeros(1, 500); for k = 1:500, c(k) = randn; end")
    assert m.num("isequal(a, b) && isequal(a(1:500), c)")
    a = m.num("a").ravel()
    assert abs(a.mean()) < 0.03 and abs(a.std() - 1) < 0.03
    assert abs(np.mean(a ** 3)) < 0.1 and abs(np.mean(a ** 4) - 3) < 0.2    # symmetric, normal tails
    assert (np.abs(a) > 3.6541528853610088).sum() <= 20                      # the tail beyond the last layer


def test_rand_and_randn_share_one_stream(m):
    m.run("rng default; u = rand; n = randn; rng default; rand; rand; rng default; v = rand(1, 2);")
    # the second pair of twister outputs gives 0.9058 as a uniform number and 1.8339 as a normal one
    assert abs(m.num("n").item() - 1.833885014595086) < 5e-15 and abs(m.num("v")[0, 1] - 0.905791937075619) < 1e-15


def test_randi_and_randperm_match_matlab(m):
    m.run("rng(0, 'twister'); r = randi(10, 1, 5); rng default; p = randperm(6); rng default; q = randperm(8, 3);"
          "rng default; s = randi([5 7], 1, 5); rng default; i8 = randi(100, 2, 2, 'int8');")
    assert m.num("r").tolist() == [[9, 10, 2, 10, 7]]
    assert m.num("p").tolist() == [[6, 3, 5, 1, 2, 4]]
    assert sorted(m.num("q").ravel().tolist()) == sorted(set(m.num("q").ravel().tolist())) and m.num("q").size == 3
    assert m.num("s").tolist() == [[7, 7, 5, 7, 6]]        # 5 + floor(3 * u) for the same uniform numbers
    assert m.num("i8").dtype == np.int8 and m.num("i8").tolist() == [[82, 13], [91, 92]]
    with pytest.raises(MatlabError, match="between 0 and N"):
        m.run("randperm(3, 5)")


def test_rng_state_save_and_restore(m):
    m.run("rng(42); s = rng; a = rand(1, 3); b = randn(1, 3); rng(s); a2 = rand(1, 3); b2 = randn(1, 3);")
    assert m.num("isequal(a, a2) && isequal(b, b2)")
    assert m.str("s.Type") == "twister" and m.num("s.Seed") == 42
    assert m.num("s.State").shape == (625, 1) and m.num("s.State").dtype == np.uint32
    m.run("old = rng(5); t = rng;")
    assert m.num("old.Seed") == 42 and m.num("t.Seed") == 5
    m.run("rng('shuffle'); u = rng; x1 = rand; rng(u); x2 = rand;")
    assert m.num("x1 == x2")


def test_randstream_objects_use_the_same_generator(m):
    m.run("st = RandStream('mt19937ar', 'Seed', 0); a = rand(st, 1, 2); b = randn(st); reset(st); c = rand(st);")
    assert np.allclose(m.num("a"), [0.814723686393179, 0.905791937075619], atol=1e-15)
    assert abs(m.num("b").item() - (-2.258846861003648)) < 5e-15      # the third normal number of the stream
    assert abs(m.num("c").item() - 0.814723686393179) < 1e-15


def test_like_and_class_arguments(m):
    m.run("a = zeros(2, 'like', single(1)); b = ones(1, 3, 'like', int8(5)); c = eye(2, 'like', 1 + 2i);"
          "d = zeros(2, 'like', true); e = rand(2, 'like', single(1)); f = ones(2, 2, 'like', uint16(1));"
          "g = randn(1, 2, 'single'); h = zeros(1, 2, 'like', 3);")
    assert m.num("a").dtype == np.float32 and m.num("b").dtype == np.int8 and m.num("b").tolist() == [[1, 1, 1]]
    assert m.num("c").dtype == np.complex128 and m.num("c").tolist() == [[1, 0], [0, 1]]
    assert m.num("d").dtype == bool and m.num("e").dtype == np.float32 and m.num("f").dtype == np.uint16
    assert m.num("g").dtype == np.float32 and m.num("h").dtype == np.float64
    with pytest.raises(MatlabError, match="'double' or 'single'"):
        m.run("rand(2, 'int8')")
    with pytest.raises(MatlabError, match="prototype"):
        m.run("zeros(2, 'like', 'abc')")
