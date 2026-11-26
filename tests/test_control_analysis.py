"""Control-system analysis: poles and zeros, time responses, frequency responses, margins, norms.

Expected values are closed-form results of the underlying mathematics.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), rtol=tol, atol=tol)


def col(m, expr):
    return np.asarray(m.num(expr)).ravel()


# -- poles, zeros, gain ---------------------------------------------------------------------------

def test_pole_zero_dcgain(m):
    assert sorted(col(m, "pole(tf(1, [1 3 2]))")) == [-2, -1]
    assert close(col(m, "zero(tf([1 1], [1 3 2]))"), [-1])
    assert m.num("pole(tf(1, [1 3 2]))").shape == (2, 1)
    assert close(m.num("dcgain(tf(1, [1 2 1]))"), 1)
    assert close(m.num("dcgain(zpk([], [-1 -2], 2))"), 1)
    assert close(m.num("dcgain(ss([0 1; -2 -3], [0; 1], [1 0], 0))"), 0.5)
    assert np.isinf(m.num("dcgain(tf(1, [1 0]))"))
    assert np.isinf(m.num("dcgain(ss(0, 1, 1, 0))"))
    assert close(m.num("dcgain(tf(1, [1 -0.5], 0.1))"), 2)          # discrete: value at z = 1
    assert close(m.num("dcgain([tf(1, [1 2]), tf([1 0], [1 1])])"), [[0.5, 0]])
    m.run("[z, k] = zero(tf([3 6], [1 4 3]));")
    assert close(m.num("z"), -2) and m.num("k") == 3


def test_transmission_zeros(m):
    # G(s) = [1/(s+1), 0; 0, (s-2)/(s+3)] has one transmission zero, at s = 2
    m.run("G = [tf(1, [1 1]), 0; 0, tf([1 -2], [1 3])];")
    assert close(col(m, "tzero(G)"), [2], 1e-7)
    assert close(col(m, "tzero([0 1; -2 -3], [0; 1], [1 1], 0)"), [-1], 1e-9)
    m.run("[z, nrank] = tzero(ss(G));")
    assert m.num("nrank") == 2
    # a tall system: both outputs see the same zero
    m.run("T = [tf([1 5], [1 3 2]); tf([2 10], [1 4 3])];")
    assert close(col(m, "tzero(T)"), [-5], 1e-6)


def test_damp(m):
    m.run("[wn, zeta, p] = damp(tf(4, [1 2 4]));")
    assert close(col(m, "wn"), [2, 2]) and close(col(m, "zeta"), [0.5, 0.5])
    m.run("[wn, zeta] = damp(tf(1, [1 11 10]));")          # sorted by natural frequency
    assert close(col(m, "wn"), [1, 10]) and close(col(m, "zeta"), [1, 1])
    # discrete poles are mapped through log(z)/Ts
    m.run("[wn, zeta] = damp(tf(1, [1 -exp(-0.3)], 0.1));")
    assert close(m.num("wn"), 3) and close(m.num("zeta"), 1)
    assert "Damping" in m.run("damp(tf(4, [1 2 4]))")


# -- time responses -------------------------------------------------------------------------------

def test_step_first_and_second_order(m):
    m.run("[y, t] = step(tf(1, [1 1]), 0:0.1:5);")
    t = col(m, "t")
    assert m.num("y").shape == (51, 1) and m.num("t").shape == (51, 1)
    assert close(col(m, "y"), 1 - np.exp(-t), 1e-12)
    zeta, wn = 0.5, 2.0
    wd = wn * np.sqrt(1 - zeta ** 2)
    m.run("[y, t] = step(tf(4, [1 2 4]), 0:0.05:6);")
    t = col(m, "t")
    ref = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * t))
    assert close(col(m, "y"), ref, 1e-11)


def test_step_default_time_shows_settled_response(m):
    for sys, final in (("tf(1, [1 1])", 1.0), ("tf(1, [1 3 3 1])", 1.0), ("tf(5, [100 1])", 5.0),
                       ("tf(4, [1 0.4 4])", 1.0), ("tf(1, [1 -0.5], 0.1)", 2.0)):
        m.run(f"[y, t] = step({sys});")
        t, y = col(m, "t"), col(m, "y")
        assert t[0] == 0 and np.allclose(np.diff(t), t[1] - t[0])
        assert abs(y[-1] - final) < 5e-3 * final, sys
        assert 10 < t.size < 6000


def test_impulse_and_initial(m):
    m.run("[y, t] = impulse(tf(1, [1 1]), 0:0.1:2);")
    assert close(col(m, "y"), np.exp(-col(m, "t")), 1e-12)
    m.run("[y, t] = impulse(tf(1, [1 0 1]), 0:0.1:3);")
    assert close(col(m, "y"), np.sin(col(m, "t")), 1e-11)
    m.run("[y, t, x] = initial(ss([0 1; -1 0], [0; 0], [1 0], 0), [1; 0], 0:0.1:3);")
    assert close(col(m, "y"), np.cos(col(m, "t")), 1e-11)
    assert close(m.num("x(:, 2)").ravel(), -np.sin(col(m, "t")), 1e-11)
    with pytest.raises(MatlabError, match="state-space"):
        m.run("initial(tf(1, [1 1]), 1)")


def test_discrete_time_responses(m):
    m.run("[y, t] = step(tf(1, [1 -0.5], 0.1), 0:0.1:1);")
    k = np.arange(11)
    assert close(col(m, "t"), 0.1 * k) and close(col(m, "y"), 2 * (1 - 0.5 ** k))
    m.run("y = impulse(tf(1, [1 -0.5], 0.1), 0.5);")       # the pulse has height 1/Ts
    assert close(col(m, "y"), [0, 10, 5, 2.5, 1.25, 0.625])
    with pytest.raises(MatlabError, match="sample time"):
        m.run("step(tf(1, [1 -0.5], 0.1), 0:0.05:1)")


def test_time_response_shapes_and_states(m):
    m.run("M = [tf(1, [1 1]), tf(2, [1 2])]; [y, t] = step(M, 0:1:3);")
    assert m.num("size(y)").tolist() == [[4, 1, 2]]
    assert close(m.num("y(end, 1, 2)"), 1 - np.exp(-6))
    m.run("sys = ss([0 1; -2 -3], [0; 1], [1 0; 0 1], 0); [y, t, x] = step(sys, 0:0.5:2);")
    assert m.num("size(y)").tolist() == [[5, 2]] and close(m.num("x"), m.num("y"))
    m.run("[y, t, x] = step(tf(1, [1 1]), 1);")
    assert m.num("x").size == 0                       # states only for state-space models


def test_step_with_delay_and_options(m):
    m.run("[y, t] = step(tf(1, [1 1], 'InputDelay', 0.55), 0:0.1:2);")
    t = col(m, "t")
    ref = np.where(t >= 0.55, 1 - np.exp(-(t - 0.55)), 0.0)
    assert close(col(m, "y"), ref, 1e-12)
    m.run("y = step(tf(1, [1 -0.5], 0.1, 'OutputDelay', 2), 0:0.1:0.5);")
    assert close(col(m, "y"), [0, 0, 0, 1, 1.5, 1.75])
    m.run("y = step(tf(1, [1 1]), 0:0.5:2, stepDataOptions('StepAmplitude', 3, 'InputOffset', 1));")
    assert close(col(m, "y"), 1 + 3 * (1 - np.exp(-np.arange(0, 2.1, 0.5))))
    with pytest.raises(MatlabError, match="improper"):
        m.run("step(tf([1 0 0], [1 1]))")


def test_lsim(m):
    m.run("t = 0:0.05:4; G = tf(1, [1 1]);")
    assert close(col(m, "lsim(G, t, t)"), np.arange(0, 4.01, 0.05) - 1 + np.exp(-np.arange(0, 4.01, 0.05)), 1e-12)
    assert close(col(m, "lsim(G, ones(1, 81), t)"), 1 - np.exp(-np.arange(0, 4.01, 0.05)), 1e-12)
    m.run("[y, tt, x] = lsim(ss(-2, 1, 3, 0), zeros(81, 1), t, 5);")       # free response from x0
    assert close(col(m, "y"), 15 * np.exp(-2 * col(m, "tt")), 1e-12)
    assert close(col(m, "x"), 5 * np.exp(-2 * col(m, "tt")), 1e-12)
    # a sine: zero-order hold differs from the exact response, first-order hold is close to it
    m.run("u = sin(3 * t); yf = lsim(G, u, t); yz = lsim(G, u, t, 'zoh');")
    tt = np.arange(0, 4.01, 0.05)
    exact = (3 * np.exp(-tt) + np.sin(3 * tt) - 3 * np.cos(3 * tt)) / 10
    assert np.max(np.abs(col(m, "yf") - exact)) < 1e-3
    assert np.max(np.abs(col(m, "yz") - exact)) > 5e-3
    m.run("yd = lsim(tf(1, [1 -0.5], 1), [1 0 0 0 0]);")       # discrete: no time vector needed
    assert close(col(m, "yd"), [0, 1, 0.5, 0.25, 0.125])
    m.run("[u, tg] = gensig('square', 2, 4, 0.5);")
    assert col(m, "u").tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 0]
    with pytest.raises(MatlabError, match="one column per input"):
        m.run("lsim(G, ones(81, 2), t)")
    with pytest.raises(MatlabError, match="evenly spaced"):
        m.run("lsim(G, [1 1 1], [0 1 3])")


def test_stepinfo_closed_forms(m):
    m.run("S = stepinfo(tf(1, [1 1]));")
    assert close(m.num("S.RiseTime"), np.log(9), 1e-8)
    assert close(m.num("S.SettlingTime"), np.log(50), 1e-8)
    assert m.num("S.Overshoot") == 0 and m.num("S.Undershoot") == 0
    zeta, wn = 0.5, 2.0
    m.run("S = stepinfo(tf(4, [1 2 4]));")
    assert close(m.num("S.Overshoot"), 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)), 1e-8)
    assert close(m.num("S.PeakTime"), np.pi / (wn * np.sqrt(1 - zeta ** 2)), 1e-7)
    assert close(m.num("S.Peak"), 1 + np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)), 1e-9)
    m.run("S = stepinfo(tf(1, [1 2 1]));")           # (1 + t) exp(-t) = 0.02 at the settling time
    ts = m.num("S.SettlingTime").item()
    assert abs((1 + ts) * np.exp(-ts) - 0.02) < 1e-9
    m.run("S = stepinfo(tf(1, [1 1]), 'SettlingTimeThreshold', 0.05, 'RiseTimeThreshold', [0.2 0.8]);")
    assert close(m.num("S.SettlingTime"), np.log(20), 1e-8) and close(m.num("S.RiseTime"), np.log(4), 1e-8)
    m.run("S = stepinfo(tf([-1 1], [1 2 1]));")       # a right-half-plane zero: the response dips first
    assert m.num("S.Undershoot") > 5
    m.run("S = stepinfo(tf(1, [1 -1]));")
    assert np.isnan(m.num("S.RiseTime")) and np.isinf(m.num("S.Peak"))
    m.run("S = stepinfo([tf(1, [1 1]), tf(2, [1 2])]);")
    assert m.num("size(S)").tolist() == [[1, 2]] and close(m.num("S(2).RiseTime"), np.log(9) / 2, 1e-8)


def test_stepinfo_and_lsiminfo_from_data(m):
    m.run("t = (0:0.001:10)'; y = 1 - exp(-t); S = stepinfo(y, t, 1);")
    assert abs(m.num("S.RiseTime") - np.log(9)) < 1e-5
    assert abs(m.num("S.SettlingTime") - np.log(50)) < 1e-5
    m.run("S2 = stepinfo(3 - 2 * exp(-t), t, 3, 1);")      # from 1 to 3
    assert abs(m.num("S2.RiseTime") - np.log(9)) < 1e-5
    m.run("L = lsiminfo(exp(-t) .* cos(5 * t), t, 0);")
    assert close(m.num("L.Max"), 1) and m.num("L.MaxTime") == 0
    assert 3.5 < m.num("L.TransientTime") < np.log(50) + 1e-6


# -- frequency responses --------------------------------------------------------------------------

def test_bode_values_and_shapes(m):
    m.run("[mag, ph, w] = bode(tf(1, [1 1]), [0.1 1 10]);")
    assert m.num("size(mag)").tolist() == [[1, 1, 3]] and m.num("w").shape == (3, 1)
    assert close(col(m, "mag"), 1 / np.sqrt(1 + np.array([0.01, 1, 100])))
    assert close(col(m, "ph"), -np.degrees(np.arctan([0.1, 1, 10])))
    m.run("[mag, ph, w] = bode(tf(1, [1 1]));")
    w = col(m, "w")
    assert w[0] <= 0.011 and w[-1] >= 99 and np.all(np.diff(w) > 0)
    m.run("[mag, ph] = bode([tf(1, [1 1]); tf(2, [1 2])], 1);")
    assert m.num("size(mag)").tolist() == [[2, 1]]
    m.run("[mag, ph] = bode(tf(1, [1 -0.5], 0.1), [1 pi/0.1]);")
    z = np.exp(1j * np.array([0.1, np.pi]))
    assert close(col(m, "mag"), np.abs(1 / (z - 0.5))) and close(col(m, "ph")[1], -180)


def test_bode_phase_is_continuous(m):
    w = np.array([0.01, 1, 100])
    m.run("[~, ph] = bode(tf(1, [1 1 0 0 0]), [0.01 1 100]);")           # three integrators
    assert close(col(m, "ph"), -270 - np.degrees(np.arctan(w)))
    m.run("[~, ph] = bode(tf(-1, [1 1]), [0.01 1 100]);")                # negative gain starts at 180
    assert close(col(m, "ph"), 180 - np.degrees(np.arctan(w)))
    m.run("[~, ph] = bode(tf([1 -1], [1 1]), [0.01 1 100]);")            # right-half-plane zero
    assert close(col(m, "ph"), 180 - 2 * np.degrees(np.arctan(w)))
    m.run("[~, ph] = bode(tf(1, [1 1], 'InputDelay', 2), [0.01 1 100]);")    # delay: no wrapping
    assert close(col(m, "ph"), -np.degrees(np.arctan(w)) - np.degrees(2 * w))
    m.run("[~, ph] = bode(tf(1, [1 0.02 1]), [0.5 2]);")                 # across a sharp resonance
    assert close(col(m, "ph"), [-np.degrees(np.arctan2(0.01, 0.75)), -180 + np.degrees(np.arctan2(0.04, 3))])
    m.run("[~, ph] = bode(ss(tf(1, [1 1 0 0 0])), [0.01 1 100]);")       # same from a state-space model
    assert close(col(m, "ph"), -270 - np.degrees(np.arctan(w)), 1e-6)


def test_nyquist_freqresp_evalfr_sigma(m):
    m.run("[re, im, w] = nyquist(tf(1, [1 1]), [1 2]);")
    assert close(col(m, "re"), [0.5, 0.2]) and close(col(m, "im"), [-0.5, -0.4])
    assert close(m.num("evalfr(tf(1, [1 1]), 1i)"), 0.5 - 0.5j)
    m.run("H = freqresp([tf(1, [1 1]), tf(1, [1 2])], [1 3]);")
    assert m.num("size(H)").tolist() == [[1, 2, 2]]
    assert close(m.num("H(1, 2, 2)"), 1 / (2 + 3j))
    assert close(m.num("freqresp(tf(1, [1 1]), 1/(2*pi), 'Hz')"), 0.5 - 0.5j)
    m.run("[sv, w] = sigma(ss([], [], [], [3 0; 0 4]), [1 10]);")
    assert close(m.num("sv"), [[4, 4], [3, 3]])
    m.run("[mag, ph] = nichols(tf(1, [1 1]), 1);")
    assert close(m.num("mag"), 1 / np.sqrt(2)) and close(m.num("ph"), -45)
    with pytest.raises(MatlabError, match="single complex number"):
        m.run("evalfr(tf(1, [1 1]), [1 2])")


def test_margins_closed_forms(m):
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(1, [1 3 2 0]));")       # 1/(s(s+1)(s+2))
    assert close(m.num("Gm"), 6) and close(m.num("Wcg"), np.sqrt(2))
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(1, [1 1 0]));")         # 1/(s(s+1))
    wc = np.sqrt((np.sqrt(5) - 1) / 2)
    assert np.isinf(m.num("Gm")) and close(m.num("Wcp"), wc)
    assert close(m.num("Pm"), 90 - np.degrees(np.arctan(wc)))
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(1, [1 3 3 1]));")       # 1/(s+1)^3
    assert close(m.num("Gm"), 8) and close(m.num("Wcg"), np.sqrt(3))
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(0.5, [1 1]));")         # never reaches 0 dB or -180 degrees
    assert np.isinf(m.num("Gm")) and np.isinf(m.num("Pm")) and np.isnan(m.num("Wcp"))
    m.run("[Gm, Pm] = margin(ss(tf(1, [1 3 2 0])));")
    assert close(m.num("Gm"), 6, 1e-7)


def test_margins_with_delay_and_discrete_time(m):
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(2, [1 1], 'InputDelay', 0.5));")
    assert close(m.num("Wcp"), np.sqrt(3))
    assert close(m.num("Pm"), 180 - 60 - np.degrees(0.5 * np.sqrt(3)))
    wg = m.num("Wcg").item()
    assert abs(np.arctan(wg) + 0.5 * wg - np.pi) < 1e-9
    assert close(m.num("Gm"), np.sqrt(1 + wg ** 2) / 2)
    m.run("[Gm, Pm, Wcg, Wcp] = margin(tf(1, [1 -0.5], 0.1));")
    th = np.arccos(0.25)
    assert close(m.num("Gm"), 1.5) and close(m.num("Wcg"), np.pi / 0.1)
    assert close(m.num("Wcp"), th / 0.1)
    assert close(m.num("Pm"), 180 - np.degrees(np.arctan2(np.sin(th), np.cos(th) - 0.5)))


def test_allmargin_and_data_margins(m):
    m.run("S = allmargin(tf(8, [1 6 11 6]));")
    assert close(m.num("S.GainMargin"), 7.5) and close(m.num("S.GMFrequency"), np.sqrt(11))
    assert m.num("S.Stable") and m.num("S.PhaseMargin").size == 1
    wp = m.num("S.PMFrequency").item()
    assert close(m.num("S.DelayMargin"), np.radians(m.num("S.PhaseMargin").item()) / wp)
    assert not m.num("getfield(allmargin(tf(80, [1 6 11 6])), 'Stable')")
    m.run("w = logspace(-2, 2, 4000); [mag, ph] = bode(tf(1, [1 3 2 0]), w);")
    m.run("[Gm, Pm, Wcg, Wcp] = margin(squeeze(mag), squeeze(ph), w);")
    assert abs(m.num("Gm") - 6) < 1e-3 and abs(m.num("Wcg") - np.sqrt(2)) < 1e-4
    m.run("[Gm2, Pm2] = margin(frd(tf(1, [1 3 2 0]), w));")
    assert abs(m.num("Gm2") - 6) < 1e-3 and abs(m.num("Pm2") - m.num("Pm")) < 1e-6


def test_bandwidth_crossover_and_norms(m):
    assert close(m.num("bandwidth(tf(1, [1 1]))"), np.sqrt(10 ** 0.3 - 1))
    assert close(m.num("bandwidth(tf(1, [1 1]), -6)"), np.sqrt(10 ** 0.6 - 1))
    assert np.isnan(m.num("bandwidth(tf(1, [1 0]))")) and np.isinf(m.num("bandwidth(tf([1 1], [1 2]))"))
    assert close(col(m, "getGainCrossover(tf(10, [1 1]), 1)"), [np.sqrt(99)])
    assert close(m.num("norm(tf(1, [1 1]))"), np.sqrt(0.5))
    assert close(m.num("norm(tf(1, [1 1]), 2)"), np.sqrt(0.5))
    assert np.isinf(m.num("norm(tf([1 2], [1 1]))")) and np.isinf(m.num("norm(tf(1, [1 -1]))"))
    zeta = 0.1
    m.run("[g, f] = norm(tf(1, [1 0.2 1]), inf); [gp, fp] = getPeakGain(tf(1, [1 0.2 1]));")
    assert close(m.num("g"), 1 / (2 * zeta * np.sqrt(1 - zeta ** 2)))
    assert close(m.num("f"), np.sqrt(1 - 2 * zeta ** 2)) and close(m.num("gp"), m.num("g"))
    assert np.isinf(m.num("norm(tf(1, [1 0 1]), inf)"))
    # discrete: H2 norm of 1/(z - a) is 1/sqrt(1 - a^2), peak gain 1/(1 - a) at w = 0
    assert close(m.num("norm(tf(1, [1 -0.5], 0.1))"), 1 / np.sqrt(0.75))
    assert close(m.num("norm(tf(1, [1 -0.5], 0.1), inf)"), 2)
    assert close(m.num("norm(ss([], [], [], [3 0; 0 4]), inf)"), 4)
    m.run("[g, f] = norm([tf(1, [1 0.2 1]), tf(1, [1 1])], inf);")
    w = m.num("f").item()
    ref = np.hypot(1 / abs(1 - w * w + 0.2j * w), 1 / abs(1 + 1j * w))
    assert close(m.num("g"), ref, 1e-7) and 0.95 < w < 1


def test_rlocus_and_pzmap(m):
    m.run("[r, k] = rlocus(tf(1, [1 3 2 0]), [0 1 6]);")
    assert m.num("size(r)").tolist() == [[3, 3]]
    assert close(np.sort_complex(m.num("r(:, 2)").ravel()), np.sort_complex(np.roots([1, 3, 2, 1])))
    assert np.max(np.abs(m.num("real(r(:, 3))"))) > 2.9 and np.min(np.abs(m.num("real(r(:, 3))"))) < 1e-9
    m.run("[r, k] = rlocus(tf(1, [1 2 0]));")             # branches meet at s = -1 for k = 1
    k = col(m, "k")
    assert k[0] == 0 and np.isinf(k[-1]) and np.all(np.diff(k[:-1]) > 0)
    assert np.any(np.abs(k - 1) < 1e-9)
    r = m.num("r")
    assert r.shape[0] == 2
    finite = r[:, 1:-1]
    assert np.allclose(np.sum(finite, axis=0), -2)        # the sum of the closed-loop poles stays -2
    m.run("[p, z] = pzmap(tf([1 2], [1 2 5]));")
    assert close(np.sort_complex(col(m, "p")), [-1 - 2j, -1 + 2j]) and close(col(m, "z"), [-2])
    with pytest.raises(MatlabError, match="SISO"):
        m.run("rlocus([tf(1, [1 1]), tf(1, [1 2])])")
