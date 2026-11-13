"""Rotation and transform conversions, code-generation directives and simsizes."""

import numpy as np
import pytest

from mylab.api import Session


@pytest.fixture()
def s():
    return Session()


def val(s, code):
    r = s.run(f"ans__ = {code};")
    assert r.error is None, r.error
    return np.asarray(s.interp.get_var("ans__"), dtype=float)


def test_rotations_about_one_axis(s):
    assert np.allclose(val(s, "rotz(90)"), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(val(s, "rotx(90)"), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12)
    assert np.allclose(val(s, "roty(90)"), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
    # Euler angles are in the order of the axes: the default is Z, then Y, then X
    assert np.allclose(val(s, "eul2rotm([pi/2 0 0])"), val(s, "rotz(90)"), atol=1e-12)
    assert np.allclose(val(s, "eul2rotm([0 0 pi/2])"), val(s, "rotx(90)"), atol=1e-12)
    assert np.allclose(val(s, "eul2rotm([0.3 0 0], 'XYZ')"), val(s, "rotx(0.3*180/pi)"), atol=1e-12)
    assert np.allclose(val(s, "axang2rotm([0 0 1 pi/2])"), val(s, "rotz(90)"), atol=1e-12)


def test_conversions_round_trip(s):
    s.run("e = [0.4 -0.7 1.1]; R = eul2rotm(e); q = rotm2quat(R); T = eul2tform(e);")
    assert np.allclose(val(s, "rotm2eul(R)"), [0.4, -0.7, 1.1])
    assert np.allclose(val(s, "R*R'"), np.eye(3), atol=1e-12) and val(s, "det(R)") == pytest.approx(1)
    assert np.allclose(val(s, "quat2rotm(q)"), val(s, "R"))
    assert np.allclose(val(s, "norm(q)"), 1) and np.allclose(val(s, "quat2eul(eul2quat(e))"), [0.4, -0.7, 1.1])
    assert np.allclose(val(s, "tform2eul(T)"), [0.4, -0.7, 1.1]) and np.allclose(val(s, "tform2rotm(T)"), val(s, "R"))
    assert np.allclose(val(s, "rotm2tform(R)"), val(s, "T")) and np.allclose(val(s, "quat2tform(tform2quat(T))"), val(s, "T"))
    assert np.allclose(val(s, "rotm2eul(eul2rotm(e, 'XYZ'), 'XYZ')"), [0.4, -0.7, 1.1])
    assert np.allclose(val(s, "tform2trvec(trvec2tform([1 2 3]))"), [1, 2, 3])
    assert np.allclose(val(s, "trvec2tform([1 2 3])")[:, 3], [1, 2, 3, 1])
    a = val(s, "rotm2axang(axang2rotm([0 1 0 0.6]))")
    assert np.allclose(a, [0, 1, 0, 0.6])
    # the identity quaternion is [1 0 0 0]: the scalar part comes first
    assert np.allclose(val(s, "eul2quat([0 0 0])"), [1, 0, 0, 0]) and np.allclose(val(s, "quat2rotm([1 0 0 0])"), np.eye(3))


def test_angdiff_wraps_to_plus_minus_pi(s):
    assert val(s, "angdiff(0.1, 0.3)") == pytest.approx(0.2)
    assert val(s, "angdiff(pi - 0.1, -pi + 0.1)") == pytest.approx(0.2)  # the short way round
    assert np.allclose(val(s, "angdiff([0 1 3])"), [1, 2])
    assert val(s, "angdiff(3*pi)") == pytest.approx(np.pi) or val(s, "angdiff(3*pi)") == pytest.approx(-np.pi)


def test_code_generation_directives_do_nothing_when_code_is_run(s):
    r = s.run(
        "coder.extrinsic('disp'); coder.inline('always'); coder.varsize('x', [1 10]);\n"
        "x = coder.const(3) + coder.nullcopy(4); t = coder.target('MATLAB'); f = coder.target('Rtw');"
    )
    assert r.error is None, r.error
    assert val(s, "x") == 7 and val(s, "t") == 1 and val(s, "f") == 0
    r = s.run("coder.ceval('some_c_function', 1);")
    assert r.error and "compiled" in r.error.lower()


def test_simsizes_makes_the_vector_a_level_1_s_function_returns(s):
    r = s.run("z = simsizes; z.NumContStates = 2; z.NumOutputs = 3; z.NumInputs = 1; z.DirFeedthrough = 1; z.NumSampleTimes = 1; v = simsizes(z);")
    assert r.error is None, r.error
    assert list(val(s, "v").ravel()) == [2, 0, 3, 1, 0, 1, 1]
    assert val(s, "numel(fieldnames(simsizes))") >= 6


def test_bad_inputs_are_refused(s):
    assert "3-by-3" in s.run("rotm2eul([1 2; 3 4])").error or "3" in s.run("rotm2eul([1 2; 3 4])").error
    assert s.run("eul2rotm([1 2 3], 'ABC')").error
    assert s.run("quat2rotm([1 2 3])").error
