"""Filter design, filter representations and filter analysis.

Expected values come from closed forms, from SciPy where it implements the same definition, or from
hand calculation.
"""

import numpy as np
import pytest
import scipy.signal as ss

from mylab.errors import MatlabError


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), rtol=tol, atol=tol)


def response(b, f):
    """Magnitude of an FIR filter at normalised frequencies f (1 = Nyquist)."""
    n = np.arange(len(b))
    return np.abs(np.exp(-1j * np.pi * np.outer(np.atleast_1d(f), n)) @ np.asarray(b, dtype=float))


# -- windows ------------------------------------------------------------------------------------------

def test_window_closed_forms(m):
    close(m.num("triang(5)").ravel(), [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])
    close(m.num("triang(4)").ravel(), [0.25, 0.75, 0.75, 0.25])
    close(m.num("bartlett(5)").ravel(), [0, 0.5, 1, 0.5, 0])
    close(m.num("tukeywin(5, 0)").ravel(), np.ones(5))
    close(m.num("tukeywin(9, 1)").ravel(), 0.5 * (1 - np.cos(2 * np.pi * np.arange(9) / 8)))
    k = np.arange(-3, 4)
    close(m.num("gausswin(7)").ravel(), np.exp(-0.5 * (2.5 * k / 3) ** 2))
    close(m.num("gausswin(7, 1.5)").ravel(), np.exp(-0.5 * (1.5 * k / 3) ** 2))
    x = 2 * np.pi * np.arange(8) / 7
    close(m.num("blackmanharris(8)").ravel(),
          0.35875 - 0.48829 * np.cos(x) + 0.14128 * np.cos(2 * x) - 0.01168 * np.cos(3 * x))
    xp = 2 * np.pi * np.arange(8) / 8
    close(m.num("blackmanharris(8, 'periodic')").ravel(),
          0.35875 - 0.48829 * np.cos(xp) + 0.14128 * np.cos(2 * xp) - 0.01168 * np.cos(3 * xp))
    # a flat-top window dips slightly below zero at its ends and peaks at one
    ft = m.num("flattopwin(9)").ravel()
    assert abs(ft[0] + 0.000421051) < 1e-9 and abs(ft[4] - 1) < 1e-8
    assert abs(m.num("nuttallwin(7)").ravel()[0] - 0.0003628) < 1e-9
    for name in ("barthannwin", "bohmanwin", "parzenwin", "chebwin", "taylorwin", "nuttallwin"):
        w = m.num(f"{name}(16)")
        assert w.shape == (16, 1) and np.allclose(w, w[::-1])
        assert m.num(f"{name}(1)").ravel()[0] == 1 or name == "taylorwin"
    close(m.num("parzenwin(7)").ravel()[[0, 3]], [2 * (1 - 3 / 3.5) ** 3, 1])
    close(m.num("bohmanwin(5)").ravel(), [0, (0.5 * 0 + 1 / np.pi), 1, 1 / np.pi, 0])
    close(m.num("chebwin(8, 60)").ravel(), ss.windows.chebwin(8, 60))
    close(m.num("taylorwin(12, 4, -30)").ravel(), ss.windows.taylor(12, 4, 30, norm=False))
    close(m.num("window(@hann, 6)"), m.num("hann(6)"))
    close(m.num("window(@kaiser, 6, 3)"), m.num("kaiser(6, 3)"))
    close(m.num("enbw(ones(8, 1))"), 1.0)
    close(m.num("enbw(hann(1024))"), 1.5 * 1024 / 1023, 1e-6)
    with pytest.raises(MatlabError):
        m.run("blackmanharris(8, 'sideways')")


# -- FIR design ---------------------------------------------------------------------------------------

def test_fir1_forms(m):
    close(m.num("fir1(10, 0.4)").ravel(), ss.firwin(11, 0.4))
    close(m.num("fir1(10, 0.4, 'high')").ravel(), ss.firwin(11, 0.4, pass_zero=False))
    close(m.num("fir1(12, [0.2 0.5])").ravel(), ss.firwin(13, [0.2, 0.5], pass_zero=False))
    close(m.num("fir1(12, [0.2 0.5], 'stop')").ravel(), ss.firwin(13, [0.2, 0.5]))
    close(m.num("fir1(8, 0.3, kaiser(9, 4))").ravel(), ss.firwin(9, 0.3, window=("kaiser", 4)))
    close(m.num("fir1(8, 0.3, 'noscale')").ravel(), ss.firwin(9, 0.3, scale=False))
    close(m.num("fir1(20, [0.1 0.3 0.6], 'DC-1')").ravel(), ss.firwin(21, [0.1, 0.3, 0.6]))
    close(m.num("fir1(20, [0.1 0.3 0.6])").ravel(), ss.firwin(21, [0.1, 0.3, 0.6], pass_zero=False))
    # a highpass filter needs an even order: an odd order is raised by one
    assert m.num("fir1(9, 0.4, 'high')").shape == (1, 11)
    assert abs(m.num("sum(fir1(10, 0.4))") - 1) < 1e-12
    with pytest.raises(MatlabError):
        m.run("fir1(10, 1.2)")


def test_fir2_firls_firpm(m):
    b = m.num("fir2(30, [0 0.4 0.4 1], [1 1 0 0])").ravel()
    assert b.size == 31 and np.allclose(b, b[::-1])
    assert abs(response(b, 0.1)[0] - 1) < 0.02 and response(b, 0.8)[0] < 0.01
    # least squares: SciPy's firls solves the same problem for odd lengths
    close(m.num("firls(10, [0 0.3 0.5 1], [1 1 0 0])").ravel(), ss.firls(11, [0, 0.3, 0.5, 1], [1, 1, 0, 0], fs=2),
          1e-8)
    close(m.num("firls(12, [0 0.3 0.5 1], [1 1 0 0], [1 10])").ravel(),
          ss.firls(13, [0, 0.3, 0.5, 1], [1, 1, 0, 0], weight=[1, 10], fs=2), 1e-8)
    # an even length is zero at the Nyquist frequency
    b = m.num("firls(5, [0 0.4 0.6 1], [1 1 0 0])").ravel()
    assert b.size == 6 and np.allclose(b, b[::-1]) and response(b, 1.0)[0] < 1e-12
    # equiripple: SciPy's remez is the same Parks-McClellan design
    close(m.num("firpm(10, [0 0.3 0.5 1], [1 1 0 0])").ravel(), ss.remez(11, [0, 0.3, 0.5, 1], [1, 0], fs=2), 1e-8)
    close(m.num("firpm(8, [0 0.2 0.4 0.6 0.8 1], [0 0 1 1 0 0], [1 5 1])").ravel(),
          ss.remez(9, [0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 0], weight=[1, 5, 1], fs=2), 1e-8)
    # a Hilbert transformer: odd symmetry, gain one, and -90 degrees of phase beyond the delay
    h = m.num("firpm(20, [0.1 0.9], [1 1], 'hilbert')").ravel()
    assert np.allclose(h, -h[::-1])
    H = np.sum(h * np.exp(-1j * np.pi * 0.5 * np.arange(21)))
    close(H * np.exp(1j * np.pi * 0.5 * 10), -1j, 0.02)
    # a differentiator approximates j*w
    d = m.num("firpm(20, [0 0.8], [0 0.8*pi], 'differentiator')").ravel()
    D = np.sum(d * np.exp(-1j * np.pi * 0.4 * np.arange(21))) * np.exp(1j * np.pi * 0.4 * 10)
    close(D, 1j * 0.4 * np.pi, 0.01)
    d2 = m.num("firls(20, [0 0.8], [0 0.8*pi], 'differentiator')").ravel()
    D2 = np.sum(d2 * np.exp(-1j * np.pi * 0.4 * np.arange(21))) * np.exp(1j * np.pi * 0.4 * 10)
    close(D2, 1j * 0.4 * np.pi, 0.01)
    m.run("[b, err] = firpm(12, [0 0.3 0.5 1], [1 1 0 0]);")
    assert 0 < m.num("err") < 0.1


def test_kaiserord(m):
    m.run("[n, Wn, beta, ftype] = kaiserord([1000 1500], [1 0], [0.05 0.01], 8000);")
    assert m.num("n") == 36 and abs(m.num("Wn") - 0.3125) < 1e-12
    att = 40.0
    close(m.num("beta"), 0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21))
    assert m.str("ftype") == "low"
    m.run("[n, Wn, beta, ftype] = kaiserord([0.2 0.3 0.6 0.7], [0 1 0], [0.01 0.05 0.01]);")
    close(m.num("Wn").ravel(), [0.25, 0.65])
    assert m.str("ftype") == "DC-0"
    m.run("c = kaiserord([0.3 0.4], [1 0], [0.01 0.01], 2, 'cell'); b = fir1(c{:});")
    assert m.num("numel(b)") == m.num("c{1}") + 1


# -- IIR design ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("code,ref", [
    ("butter(4, 0.3)", lambda: ss.butter(4, 0.3)),
    ("butter(3, 0.3, 'high')", lambda: ss.butter(3, 0.3, "high")),
    ("butter(2, [0.2 0.5])", lambda: ss.butter(2, [0.2, 0.5], "bandpass")),
    ("butter(2, [0.2 0.5], 'stop')", lambda: ss.butter(2, [0.2, 0.5], "bandstop")),
    ("butter(3, 5, 's')", lambda: ss.butter(3, 5, analog=True)),
    ("cheby1(3, 1, 0.4)", lambda: ss.cheby1(3, 1, 0.4)),
    ("cheby2(3, 30, 0.4)", lambda: ss.cheby2(3, 30, 0.4)),
    ("cheby2(4, 30, [0.2 0.6], 'stop')", lambda: ss.cheby2(4, 30, [0.2, 0.6], "bandstop")),
    ("ellip(4, 1, 40, 0.3)", lambda: ss.ellip(4, 1, 40, 0.3)),
    ("ellip(3, 0.5, 30, [0.2 0.5])", lambda: ss.ellip(3, 0.5, 30, [0.2, 0.5], "bandpass")),
    ("ellip(3, 1, 40, 2, 'high', 's')", lambda: ss.ellip(3, 1, 40, 2, "highpass", analog=True)),
    ("besself(4, 10)", lambda: ss.bessel(4, 10, analog=True, norm="phase")),
])
def test_iir_designs_match_scipy(m, code, ref):
    m.run(f"[b, a] = {code};")
    b, a = ref()
    b = np.concatenate([np.zeros(len(a) - len(b)), b])      # the numerator is padded to the denominator's length
    close(m.num("b").ravel(), b, 1e-8)
    close(m.num("a").ravel(), a, 1e-8)


def test_iir_output_forms(m):
    m.run("[z, p, k] = butter(3, 0.4);")
    z, p, k = ss.butter(3, 0.4, output="zpk")
    assert m.num("z").shape == (3, 1) and m.num("p").shape == (3, 1)
    close(np.sort_complex(m.num("p").ravel()), np.sort_complex(p))
    close(m.num("k"), k)
    # the state-space form describes the same filter
    m.run("[A, B, C, D] = cheby2(3, 30, 0.4); [b, a] = cheby2(3, 30, 0.4);")
    b2, a2 = ss.ss2tf(m.num("A"), m.num("B"), m.num("C"), m.num("D"))
    close(b2.ravel(), m.num("b").ravel(), 1e-8)
    close(a2, m.num("a").ravel(), 1e-8)
    m.run("[z, p, k] = butter(2, 1, 's');")
    assert m.num("z").size == 0
    close(np.sort_complex(m.num("p").ravel()), np.sort_complex(np.exp(1j * np.pi * np.array([0.75, 1.25]))))
    for bad in ("butter(2, 1.5)", "butter(0, 0.5)", "cheby1(2, -1, 0.5)", "butter(2, [0.5 0.2])",
                "butter(2, 0.5, 'sideways')"):
        with pytest.raises(MatlabError):
            m.run(bad)


def test_prototypes_and_transformations(m):
    m.run("[z, p, k] = buttap(4);")
    p = m.num("p").ravel()
    assert m.num("z").size == 0 and abs(m.num("k") - 1) < 1e-12
    assert np.allclose(np.abs(p), 1) and np.all(p.real < 0)
    m.run("[z, p, k] = cheb1ap(3, 1);")
    b, a = ss.zpk2tf([], m.num("p").ravel(), m.num("k"))
    close(abs(np.polyval(b, 1j) / np.polyval(a, 1j)), 10 ** (-1 / 20), 1e-9)   # 1 dB down at the edge
    m.run("[z, p, k] = cheb2ap(4, 40);")
    b, a = ss.zpk2tf(m.num("z").ravel(), m.num("p").ravel(), m.num("k"))
    close(abs(np.polyval(b, 1j) / np.polyval(a, 1j)), 10 ** (-40 / 20), 1e-9)  # 40 dB down at the stopband edge
    m.run("[z, p, k] = ellipap(3, 1, 40);")
    assert m.num("z").shape == (2, 1) and m.num("p").shape == (3, 1)
    m.run("[z, p, k] = besselap(3);")
    assert np.all(m.num("p").real < 0)
    # frequency transformations of 1/(s+1)
    m.run("[b, a] = lp2lp(1, [1 1], 5);")
    close(m.num("b").ravel() / m.num("a").ravel()[0], [5 / 1])
    close(m.num("a").ravel() / m.num("a").ravel()[0], [1, 5])
    m.run("[b, a] = lp2hp(1, [1 1], 5);")
    close(np.polyval(m.num("b").ravel(), 5j) / np.polyval(m.num("a").ravel(), 5j), 1j * 5 / (1j * 5 + 5))
    m.run("[b, a] = lp2bp(1, [1 1], 2, 1);")
    close(m.num("b").ravel(), [1, 0])
    close(m.num("a").ravel(), [1, 1, 4])
    m.run("[b, a] = lp2bs(1, [1 1], 2, 1);")
    close(abs(np.polyval(m.num("b").ravel(), 2j) / np.polyval(m.num("a").ravel(), 2j)), 0, 1e-12)
    # the state-space forms give the same transfer function
    m.run("[A, B, C, D] = lp2bp(-1, 1, 1, 0, 2, 1);")
    b2, a2 = ss.ss2tf(m.num("A"), m.num("B"), m.num("C"), m.num("D"))
    close(a2, [1, 1, 4])
    close(b2.ravel(), [0, 1, 0], 1e-12)


def test_order_estimation(m):
    # Butterworth: the order from the closed form, the cutoff from meeting the stopband exactly
    m.run("[n, Wn] = buttord(40/500, 150/500, 3, 60);")
    wp, ws = np.tan(np.pi * 0.08 / 2), np.tan(np.pi * 0.3 / 2)
    n = int(np.ceil(np.log10((10 ** 6 - 1) / (10 ** 0.3 - 1)) / (2 * np.log10(ws / wp))))
    assert m.num("n") == n == 5
    close(m.num("Wn"), 2 / np.pi * np.arctan(ws / (10 ** 6 - 1) ** (1 / (2 * n))))
    assert abs(m.num("Wn") - 0.0810) < 5e-5
    m.run("[b, a] = butter(n, Wn); h = abs(freqz(b, a, [0.08 0.3]*pi));")
    h = m.num("h").ravel()
    assert -20 * np.log10(h[0]) <= 3 and abs(-20 * np.log10(h[1]) - 60) < 1e-6
    for fn, design, pars in (("cheb1ord", "cheby1(n, 1, Wn)", (1, 40)), ("cheb2ord", "cheby2(n, 40, Wn)", (1, 40)),
                             ("ellipord", "ellip(n, 1, 40, Wn)", (1, 40))):
        m.run(f"[n, Wn] = {fn}(0.2, 0.4, 1, 40); [b, a] = {design}; h = abs(freqz(b, a, [0.2 0.4]*pi));")
        nref, _ = getattr(ss, fn)(0.2, 0.4, 1, 40)
        assert m.num("n") == nref
        h = m.num("h").ravel()
        assert -20 * np.log10(h[0]) <= pars[0] + 1e-6 and -20 * np.log10(h[1]) >= pars[1] - 1e-6
    m.run("[n, Wn] = buttord([0.2 0.5], [0.1 0.6], 1, 40);")
    assert m.num("Wn").shape == (1, 2)
    m.run("[b, a] = butter(n, Wn); h = abs(freqz(b, a, [0.1 0.2 0.5 0.6]*pi));")
    h = -20 * np.log10(m.num("h").ravel())
    assert h[1] <= 1 + 1e-6 and h[2] <= 1 + 1e-6 and h[0] >= 40 - 1e-6 and h[3] >= 40 - 1e-6
    m.run("[n, Wn] = buttord(2, 6, 1, 30, 's');")
    assert m.num("n") == int(np.ceil(np.log10((10 ** 3 - 1) / (10 ** 0.1 - 1)) / (2 * np.log10(3))))


def test_bilinear_and_impinvar(m):
    # H(s) = 1/(s+1) with fs = 2: s -> 4 (z-1)/(z+1) gives (1 + z^-1) / (5 - 3 z^-1)
    m.run("[bd, ad] = bilinear(1, [1 1], 2);")
    close(m.num("bd").ravel(), [0.2, 0.2])
    close(m.num("ad").ravel(), [1, -0.6])
    m.run("[zd, pd, kd] = bilinear([], -1, 1, 2);")
    close(m.num("pd"), 0.6)
    close(m.num("zd"), -1)
    close(m.num("kd"), 0.2)
    m.run("[Ad, Bd, Cd, Dd] = bilinear(-1, 1, 1, 0, 2);")
    b2, a2 = ss.ss2tf(m.num("Ad"), m.num("Bd"), m.num("Cd"), m.num("Dd"))
    close(b2.ravel(), [0.2, 0.2])
    close(a2, [1, -0.6])
    # pre-warping makes the digital response equal the analog one at the chosen frequency
    m.run("[b, a] = butter(2, 2*pi*10, 's'); [bd, ad] = bilinear(b, a, 100, 10); h = freqz(bd, ad, [10 10], 100);")
    close(abs(m.num("h").ravel()[0]), 1 / np.sqrt(2), 1e-9)
    # impulse invariance of 1/((s+1)(s+2)) = 1/(s+1) - 1/(s+2), sampled at 10 Hz
    m.run("[bz, az] = impinvar(1, [1 3 2], 10);")
    e1, e2 = np.exp(-0.1), np.exp(-0.2)
    close(m.num("bz").ravel(), [0, 0.1 * (e1 - e2)])
    close(m.num("az").ravel(), [1, -(e1 + e2), e1 * e2])
    # a double pole: t e^-t sampled gives T^2 x z^-1 / (1 - x z^-1)^2
    m.run("[bz, az] = impinvar(1, [1 2 1], 10);")
    close(m.num("bz").ravel(), [0, 0.01 * e1])
    close(m.num("az").ravel(), [1, -2 * e1, e1 * e1])


# -- representations ----------------------------------------------------------------------------------

def test_sos_conversions(m):
    m.run("[b, a] = ellip(5, 1, 40, 0.3); [sos, g] = tf2sos(b, a); [b2, a2] = sos2tf(sos, g);")
    close(m.num("b2").ravel(), m.num("b").ravel(), 1e-8)
    close(m.num("a2").ravel(), m.num("a").ravel(), 1e-8)
    sos = m.num("sos")
    assert sos.shape == (3, 6) and np.allclose(sos[:, 3], 1)
    # the first-order section comes first, and sections go towards the unit circle
    assert sos[0, 2] == 0 and sos[0, 5] == 0
    radius = [max(abs(np.roots(r[3:]))) for r in sos]
    assert radius == sorted(radius)
    m.run("sos1 = tf2sos(b, a); x = [1 zeros(1, 19)]; y1 = sosfilt(sos1, x); y2 = filter(b, a, x);")
    close(m.num("y1"), m.num("y2"), 1e-9)
    m.run("sd = tf2sos(b, a, 'down');")
    close(m.num("sd")[::-1, 3:], sos[:, 3:], 1e-9)
    m.run("[z, p, k] = sos2zp(sos, g); [z0, p0, k0] = ellip(5, 1, 40, 0.3);")
    close(np.sort_complex(m.num("p").ravel()), np.sort_complex(m.num("p0").ravel()), 1e-8)
    close(m.num("k"), m.num("k0"), 1e-9)
    m.run("[z, p, k] = butter(2, 0.5); s2 = zp2sos(z, p, k); [bb, aa] = butter(2, 0.5);")
    close(m.num("s2").ravel(), np.concatenate([m.num("bb").ravel(), m.num("aa").ravel()]), 1e-9)
    m.run("[A, B, C, D] = butter(4, 0.3); s3 = ss2sos(A, B, C, D); [b3, a3] = sos2tf(s3); [b4, a4] = butter(4, 0.3);")
    close(m.num("b3").ravel(), m.num("b4").ravel(), 1e-7)
    close(m.num("sosfilt([1 1 0 1 0 0; 1 0 0 1 -0.5 0], [1 0 0 0]')").ravel(), [1, 1.5, 0.75, 0.375])
    with pytest.raises(MatlabError):
        m.run("sosfilt([1 2 3], 1:5)")


def test_zero_pole_conversions(m):
    m.run("[z, p, k] = tf2zpk([1 0], [1 -0.5]);")
    close(m.num("z"), 0)
    close(m.num("p"), 0.5)
    close(m.num("k"), 1)
    m.run("[z, p, k] = tf2zpk([2 1], [1 -0.3 0.02]);")          # in powers of z^-1: a zero at the origin too
    close(np.sort(m.num("z").ravel()), [-0.5, 0])
    close(np.sort(m.num("p").ravel()), [0.1, 0.2])
    m.run("[z, p, k] = tf2zp([2 4], [1 3 2]);")                  # in powers of s
    close(m.num("z"), -2)
    close(np.sort(m.num("p").ravel()), [-2, -1])
    close(m.num("k"), 2)
    m.run("[b, a] = zp2tf([-1; -3], [-2; -4; -5], 3);")
    close(m.num("b").ravel(), [0, 3, 12, 9])
    close(m.num("a").ravel(), np.poly([-2, -4, -5]))
    m.run("[A, B, C, D] = tf2ss([1 2], [1 3 2]); [b, a] = ss2tf(A, B, C, D);")
    close(m.num("A"), [[-3, -2], [1, 0]])
    close(m.num("b").ravel(), [0, 1, 2])
    close(m.num("a").ravel(), [1, 3, 2])
    m.run("[z, p, k] = ss2zp(A, B, C, D);")
    close(m.num("z"), -2)


def test_reflection_coefficients_and_lattices(m):
    # step-down by hand for [1 0.5 0.2]: k2 = 0.2, k1 = (0.5 - 0.2*0.5)/(1 - 0.04)
    close(m.num("poly2rc([1 0.5 0.2])").ravel(), [0.4 / 0.96, 0.2])
    close(m.num("rc2poly(poly2rc([1 -0.9 0.6 -0.2]))").ravel(), [1, -0.9, 0.6, -0.2])
    m.run("[k, v] = tf2latc([1 2 3], [1 0.5 0.2]); [b, a] = latc2tf(k, v);")
    close(m.num("b").ravel(), [1, 2, 3])
    close(m.num("a").ravel(), [1, 0.5, 0.2])
    m.run("k = tf2latc([1 0.5 0.2]); [f, g] = latcfilt(k, [1 0 0 0]);")
    close(m.num("f").ravel(), [1, 0.5, 0.2, 0])
    close(m.num("g").ravel(), [0.2, 0.5, 1, 0])
    m.run("x = [1 -1 2 0.5 0 3]; [k, v] = tf2latc([1 2 3], [1 0.5 0.2]); f = latcfilt(k, v, x);")
    close(m.num("f").ravel(), ss.lfilter([1, 2, 3], [1, 0.5, 0.2], [1, -1, 2, 0.5, 0, 3]))
    m.run("[b, a] = latc2tf([0.5 0.2], 'allpole');")
    close(m.num("b"), 1)
    close(m.num("a").ravel(), m.num("rc2poly([0.5 0.2])").ravel())


def test_linear_prediction(m):
    # Levinson-Durbin on r = [1 0.5 0.2]: the 2-by-2 normal equations solved by hand
    m.run("[a, e, k] = levinson([1 0.5 0.2], 2);")
    R = np.array([[1, 0.5], [0.5, 1]])
    sol = np.linalg.solve(R, -np.array([0.5, 0.2]))
    close(m.num("a").ravel(), np.concatenate([[1], sol]))
    close(m.num("e"), 1 + np.dot(sol, [0.5, 0.2]))
    close(m.num("k").ravel()[-1], sol[-1])
    x = np.array([1.0, 2, 3, 4, 3, 2, 1, 0, -1, -2])
    m.interp.set_var("x", x.reshape(1, -1))
    r = np.array([np.dot(x[:x.size - j], x[j:]) for j in range(3)]) / x.size
    sol = np.linalg.solve([[r[0], r[1]], [r[1], r[0]]], -r[1:])
    m.run("[a, e] = aryule(x, 2); [a2, g] = lpc(x, 2);")
    close(m.num("a").ravel(), np.concatenate([[1], sol]))
    close(m.num("a2").ravel(), m.num("a").ravel())
    close(m.num("e"), r[0] + np.dot(sol, r[1:]))
    close(m.num("g"), m.num("e"))
    # Burg and the covariance methods recover a noise-free AR(2) signal exactly
    y = ss.lfilter([1], [1, -1.2, 0.5], np.r_[1.0, np.zeros(39)])
    m.interp.set_var("y", y.reshape(-1, 1))
    close(m.num("arcov(y(2:end), 2)").ravel(), [1, -1.2, 0.5], 1e-8)
    close(m.num("armcov(y(3:end), 2)").ravel(), [1, -1.2, 0.5], 1e-8)
    m.run("[ab, eb, kb] = arburg(y, 2);")
    assert np.all(np.abs(m.num("kb")) < 1) and m.num("ab").shape == (1, 3)
    m.run("A = aryule([x' flipud(x')], 2);")
    assert m.num("A").shape == (2, 3)


# -- analysis -----------------------------------------------------------------------------------------

def test_freqz_forms(m):
    m.run("[h, w] = freqz([1 1]/2, 1, 8);")
    w = np.arange(8) * np.pi / 8
    close(m.num("w").ravel(), w)
    close(m.num("h").ravel(), (1 + np.exp(-1j * w)) / 2)
    assert m.num("h").shape == (8, 1)
    assert m.num("freqz([1 2 3], [1 -0.5])").shape == (512, 1)
    m.run("[h, w] = freqz([1 2 3], [1 -0.5], 4, 'whole');")
    close(m.num("w").ravel(), np.arange(4) * np.pi / 2)
    z = np.exp(-1j * np.arange(4) * np.pi / 2)
    close(m.num("h").ravel(), (1 + 2 * z + 3 * z * z) / (1 - 0.5 * z))
    m.run("[h, f] = freqz([1 2 3], 1, 4, 1000);")
    close(m.num("f").ravel(), [0, 125, 250, 375])
    # a vector of frequencies keeps its shape; with a sample rate it is in Hz
    assert m.num("freqz([1 1], 1, [0 pi/2 pi])").shape == (1, 3)
    close(m.num("freqz([1 1], 1, [0 pi/2 pi])").ravel(), [2, 1 - 1j, 0])
    close(m.num("freqz([1 1], 1, [0 250 500], 1000)").ravel(), [2, 1 - 1j, 0])
    m.run("sos = [1 2 1 1 -0.5 0.2; 1 1 0 1 0.3 0]; [b, a] = sos2tf(sos); h1 = freqz(sos, 16); h2 = freqz(b, a, 16);")
    close(m.num("h1"), m.num("h2"))


def test_grpdelay_phasez_impz_freqs(m):
    # a symmetric FIR filter delays every frequency by half its order
    close(m.num("grpdelay([1 2 3 2 1], 1, 16)").ravel(), 2 * np.ones(16))
    w, gd = ss.group_delay(([1, 0.5], [1, -0.4]), w=np.arange(8) * np.pi / 8)
    m.run("[gd, w] = grpdelay([1 0.5], [1 -0.4], 8);")
    close(m.num("gd").ravel(), gd, 1e-8)
    m.run("[phi, w] = phasez([0 0 1], 1, 8);")                   # a pure two-sample delay
    close(m.num("phi").ravel(), -2 * np.arange(8) * np.pi / 8)
    m.run("[h, t] = impz(1, [1 -0.5], 4);")
    close(m.num("h").ravel(), [1, 0.5, 0.25, 0.125])
    close(m.num("t").ravel(), [0, 1, 2, 3])
    assert m.num("impz([1 2 3])").shape == (3, 1)
    m.run("[h, t] = impz(1, [1 -0.5], 4, 100);")
    close(m.num("t").ravel(), [0, 0.01, 0.02, 0.03])
    close(m.num("impz(1, [1 -0.5], [0 3])").ravel(), [1, 0.125])
    assert m.num("numel(impz(1, [1 -0.9]))") > 50                 # long enough for the response to die away
    close(m.num("stepz(1, [1 -0.5], 4)").ravel(), [1, 1.5, 1.75, 1.875])
    close(m.num("freqs(1, [1 1], [0 1 10])").ravel(), 1 / (1 + 1j * np.array([0, 1, 10])))
    m.run("[h, w] = freqs([1 0], [1 2 5]);")
    assert m.num("h").shape == (1, 200) and np.all(np.diff(m.num("w").ravel()) > 0)
    assert m.num("filtord([1 2 3], [1 0.5])") == 2 and m.num("filtord([1 2 1 1 0 0; 1 1 0 1 0.5 0])") == 3
    close(m.num("fftfilt([1 1]/2, [2 4 6 8])").ravel(), [1, 3, 5, 7])
    close(m.num("fftfilt([1 -1], [1; 4; 9])").ravel(), [1, 3, 5])


def test_designfilt_objects(m):
    m.run("d = designfilt('lowpassfir', 'FilterOrder', 10, 'CutoffFrequency', 0.4);")
    assert m.str("class(d)") == "digitalFilter"
    close(m.num("d.Coefficients").ravel(), ss.firwin(11, 0.4))
    assert m.num("d.FilterOrder") == 10 and m.str("d.DesignMethod") == "window"
    assert m.str("d.FrequencyResponse") == "lowpass" and m.str("d.ImpulseResponse") == "fir"
    x = np.cos(0.2 * np.arange(40))
    m.interp.set_var("x", x.reshape(1, -1))
    close(m.num("filter(d, x)").ravel(), ss.lfilter(ss.firwin(11, 0.4), 1, x))
    close(m.num("freqz(d, 8)"), m.num("freqz(d.Coefficients, 1, 8)"))
    close(m.num("grpdelay(d, 4)").ravel(), 5 * np.ones(4))
    assert m.num("filtord(d)") == 10 and m.num("isstable(d)") and m.num("islinphase(d)")
    assert "digitalFilter" in m.run("d")
    # sample rate, IIR designs as second-order sections, and the tf/zpk forms
    m.run("d2 = designfilt('lowpassiir', 'FilterOrder', 4, 'HalfPowerFrequency', 150, 'SampleRate', 1000);")
    b, a = ss.butter(4, 0.3)
    assert m.num("d2.Coefficients").shape == (2, 6)
    m.run("[b, a] = tf(d2); [z, p, k] = zpk(d2);")
    close(m.num("b").ravel(), b, 1e-8)
    close(m.num("a").ravel(), a, 1e-8)
    close(m.num("k"), ss.butter(4, 0.3, output="zpk")[2], 1e-9)
    close(m.num("filter(d2, x)").ravel(), ss.lfilter(b, a, x), 1e-8)
    close(m.num("filtfilt(d2, x)").ravel(), ss.filtfilt(b, a, x, padtype="odd", padlen=12), 1e-6)
    m.run("d3 = designfilt('highpassiir', 'FilterOrder', 3, 'PassbandFrequency', 0.4, 'PassbandRipple', 1);")
    b, a = ss.cheby1(3, 1, 0.4, "high")
    m.run("[b, a] = tf(d3);")
    close(m.num("b").ravel(), b, 1e-8)
    m.run("d4 = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', 0.2, 'HalfPowerFrequency2', 0.5);")
    m.run("[b, a] = tf(d4);")
    close(m.num("a").ravel(), ss.butter(2, [0.2, 0.5], "bandpass")[1], 1e-8)
    # minimum-order designs meet their specification
    m.run("d5 = designfilt('lowpassfir', 'PassbandFrequency', 0.2, 'StopbandFrequency', 0.3, "
          "'PassbandRipple', 1, 'StopbandAttenuation', 40); b5 = d5.Coefficients;")
    b5 = m.num("b5").ravel()
    assert np.all(np.abs(20 * np.log10(response(b5, np.linspace(0, 0.2, 50)))) <= 0.5 + 1e-6)
    assert np.all(20 * np.log10(response(b5, np.linspace(0.3, 1, 100)) + 1e-12) <= -40 + 1e-6)
    m.run("d6 = designfilt('lowpassiir', 'PassbandFrequency', 0.2, 'StopbandFrequency', 0.3, "
          "'PassbandRipple', 1, 'StopbandAttenuation', 40, 'DesignMethod', 'ellip'); h = abs(freqz(d6, [0.2 0.3]*pi));")
    h = -20 * np.log10(m.num("h").ravel())
    assert h[0] <= 1 + 1e-6 and h[1] >= 40 - 1e-6 and m.num("filtord(d6)") == ss.ellipord(0.2, 0.3, 1, 40)[0]
    m.run("d7 = designfilt('lowpassfir', 'FilterOrder', 12, 'PassbandFrequency', 0.3, 'StopbandFrequency', 0.5);")
    close(m.num("d7.Coefficients").ravel(), ss.remez(13, [0, 0.3, 0.5, 1], [1, 0], fs=2), 1e-8)
    m.run("d8 = designfilt('hilbertfir', 'FilterOrder', 20, 'TransitionWidth', 0.1);")
    h = m.num("d8.Coefficients").ravel()
    assert np.allclose(h, -h[::-1])
    for bad in ("designfilt('lowpassfir')", "designfilt('nosuch', 'FilterOrder', 3)",
                "designfilt('lowpassfir', 'FilterOrder', 10, 'Cutoff', 0.4)"):
        with pytest.raises(MatlabError):
            m.run(bad)


def test_lowpass_family(m):
    m.run("t = (0:999)/1000; x = sin(2*pi*20*t) + sin(2*pi*200*t);")
    m.run("[y, d] = lowpass(x, 50, 1000); e = y - sin(2*pi*20*t);")
    assert m.num("y").shape == (1, 1000) and m.str("class(d)") == "digitalFilter"
    assert np.max(np.abs(m.num("e").ravel()[150:850])) < 0.02       # the fast tone is gone, with no delay left
    m.run("y = highpass(x, 100, 1000); e = y - sin(2*pi*200*t);")
    assert np.max(np.abs(m.num("e").ravel()[150:850])) < 0.02
    m.run("y = bandpass(x, [150 250], 1000); e = y - sin(2*pi*200*t);")
    assert np.max(np.abs(m.num("e").ravel()[200:800])) < 0.02
    m.run("y = bandstop(x, [150 250], 1000); e = y - sin(2*pi*20*t);")
    assert np.max(np.abs(m.num("e").ravel()[200:800])) < 0.02
    assert m.num("lowpass(ones(1, 100), 0.2)").shape == (1, 100)   # short signals fall back to an IIR filter
    assert m.num("lowpass(ones(300, 2), 0.2)").shape == (300, 2)
    m.run("[y, d] = lowpass(x, 50, 1000, 'ImpulseResponse', 'iir', 'Steepness', 0.95);")
    assert m.str("d.ImpulseResponse") == "iir"
    with pytest.raises(MatlabError):
        m.run("lowpass(x, 600, 1000)")
