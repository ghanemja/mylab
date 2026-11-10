"""Tests for the signal, stats, optim and extras built-in modules.

Expected values are computed independently with NumPy/SciPy or by hand.
"""

import numpy as np
import pytest
import scipy.signal as ss
import scipy.special as sp
import scipy.stats as st

from mylab.errors import MatlabError
from mylab.values import Cell, MString


def close(a, b, tol=1e-8):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=tol)


# =========================================================================================
# Signal processing
# =========================================================================================

class TestFFT:
    def test_fft_row(self, m):
        x = np.array([1.0, 2, 3, 4, 7])
        r = m.num("fft([1 2 3 4 7])")
        assert r.shape == (1, 5)
        close(r.ravel(), np.fft.fft(x))

    def test_fft_column_and_n(self, m):
        r = m.num("fft([1;2;3], 8)")
        assert r.shape == (8, 1)
        close(r.ravel(), np.fft.fft([1, 2, 3], 8))
        r = m.num("fft([1 2 3 4], 2)")
        close(r.ravel(), np.fft.fft([1, 2]))

    def test_fft_matrix_dims(self, m):
        a = np.arange(1.0, 7).reshape(2, 3)
        close(m.num("fft([1 2 3; 4 5 6])"), np.fft.fft(a, axis=0))
        close(m.num("fft([1 2 3; 4 5 6], [], 2)"), np.fft.fft(a, axis=1))

    def test_ifft_roundtrip_real(self, m):
        r = m.num("ifft(fft([1 2 3 4]))")
        assert not np.iscomplexobj(r)
        close(r, [[1, 2, 3, 4]])
        r = m.num("ifft([1 1i 0])")
        close(r.ravel(), np.fft.ifft([1, 1j, 0]))
        r = m.num("ifft([1 2 3], 'symmetric')")
        assert not np.iscomplexobj(r)

    def test_fft2_ifft2(self, m):
        a = np.array([[8.0, 1, 6], [3, 5, 7], [4, 9, 2]])
        close(m.num("fft2(magic(3))"), np.fft.fft2(a))
        close(m.num("ifft2(fft2(magic(3)))"), a)
        assert m.num("fft2(magic(3), 4, 5)").shape == (4, 5)

    def test_shifts(self, m):
        close(m.num("fftshift(1:5)"), [[4, 5, 1, 2, 3]])
        close(m.num("ifftshift(fftshift(1:6))"), [[1, 2, 3, 4, 5, 6]])
        close(m.num("fftshift([1 2; 3 4])"), np.fft.fftshift([[1, 2], [3, 4]]))
        close(m.num("fftshift([1 2 3; 4 5 6], 2)"), [[3, 1, 2], [6, 4, 5]])

    def test_nextpow2(self, m):
        close(m.num("nextpow2([0 1 5 1024 1025 0.3])"), [[0, 0, 3, 10, 11, -1]])


class TestFilters:
    def test_butter_low(self, m):
        m.run("[b, a] = butter(4, 0.25);")
        b, a = ss.butter(4, 0.25)
        assert m.val("b").shape == (1, 5)
        close(m.val("b"), [b])
        close(m.val("a"), [a])

    def test_butter_types(self, m):
        m.run("[b, a] = butter(2, 0.6, 'high');")
        b, a = ss.butter(2, 0.6, "highpass")
        close(m.val("b"), [b])
        close(m.val("a"), [a])
        m.run("[b, a] = butter(2, [0.2 0.5]);")
        b, a = ss.butter(2, [0.2, 0.5], "bandpass")
        close(m.val("b"), [b])
        m.run("[b, a] = butter(2, [0.2 0.5], 'stop');")
        b, a = ss.butter(2, [0.2, 0.5], "bandstop")
        close(m.val("a"), [a])
        close(m.val("b"), [b])

    def test_butter_zpk_and_errors(self, m):
        m.run("[z, p, k] = butter(3, 0.5);")
        z, p, k = ss.butter(3, 0.5, output="zpk")
        assert m.val("p").shape == (3, 1)
        close(m.val("k"), k)
        with pytest.raises(MatlabError):
            m.run("butter(2, 1.5)")

    def test_cheby1(self, m):
        m.run("[b, a] = cheby1(3, 0.5, 0.4);")
        b, a = ss.cheby1(3, 0.5, 0.4)
        close(m.val("b"), [b])
        close(m.val("a"), [a])
        m.run("[b, a] = cheby1(2, 1, 0.3, 'high');")
        b, a = ss.cheby1(2, 1, 0.3, "highpass")
        close(m.val("b"), [b])

    def test_freqz(self, m):
        m.run("[h, w] = freqz([1 2 1], [1 -0.5], 64);")
        w, h = ss.freqz([1, 2, 1], [1, -0.5], worN=64)
        assert m.val("h").shape == (64, 1)
        close(m.val("w").ravel(), w)
        close(m.val("h").ravel(), h)
        m.run("[h, f] = freqz([1 1], 1, 8, 100);")
        close(m.val("f").ravel(), np.arange(8) * 50 / 8)

    def test_filtfilt(self, m):
        x = np.sin(np.arange(1, 41) / 3.0) + 0.1 * np.cos(np.arange(1, 41) * 2.0)
        m.interp.set_var("x", x.reshape(1, -1))
        m.run("[b, a] = butter(2, 0.3); y = filtfilt(b, a, x);")
        b, a = ss.butter(2, 0.3)
        expected = ss.filtfilt(b, a, x, padtype="odd", padlen=3 * (3 - 1))
        assert m.val("y").shape == (1, 40)
        close(m.val("y").ravel(), expected)
        m.run("yc = filtfilt(b, a, x');")
        assert m.val("yc").shape == (40, 1)


class TestWindows:
    def test_symmetric_windows(self, m):
        close(m.num("hamming(6)").ravel(), np.hamming(6))
        close(m.num("hann(7)").ravel(), np.hanning(7))
        close(m.num("blackman(8)").ravel(), np.clip(np.blackman(8), 0, None), 1e-12)
        close(m.num("kaiser(5, 3)").ravel(), np.kaiser(5, 3))
        close(m.num("kaiser(4)").ravel(), np.kaiser(4, 0.5))
        assert m.num("hamming(6)").shape == (6, 1)

    def test_hanning_rectwin_periodic(self, m):
        close(m.num("hanning(3)").ravel(), [0.5, 1, 0.5])
        close(m.num("rectwin(4)"), np.ones((4, 1)))
        close(m.num("hann(4, 'periodic')").ravel(), ss.windows.hann(4, sym=False))
        close(m.num("hamming(1)"), [[1]])


class TestSpectra:
    def test_xcorr(self, m):
        x, y = np.array([1.0, 2, 3]), np.array([0.0, 1, 0.5])
        m.run("[r, lags] = xcorr([1 2 3], [0 1 0.5]);")
        close(m.val("r").ravel(), np.correlate(x, y, "full"))
        close(m.val("lags"), [[-2, -1, 0, 1, 2]])
        r = np.correlate(x, x, "full")
        close(m.num("xcorr([1 2 3], 'biased')").ravel(), r / 3)
        close(m.num("xcorr([1 2 3], 'unbiased')").ravel(), r / (3 - np.abs(np.arange(-2, 3))))
        close(m.num("xcorr([1 2 3], 'coeff')").ravel(), r / r.max())
        assert m.num("xcorr([1;2;3])").shape == (5, 1)
        close(m.num("xcorr([1 2 3 4], 1)").ravel(), np.correlate([1, 2, 3, 4], [1, 2, 3, 4], "full")[2:5])

    def test_periodogram(self, m):
        x = np.array([1.0, 3, 2, 5, 4, 0, -1, 2])
        m.interp.set_var("x", x.reshape(1, -1))
        m.run("[p, f] = periodogram(x, [], 8, 1);")
        X = np.fft.fft(x)
        p = np.abs(X) ** 2 / 8
        p = p[:5]
        p[1:4] *= 2
        close(m.val("p").ravel(), p)
        close(m.val("f").ravel(), np.arange(5) / 8)
        f2, p2 = ss.periodogram(x, fs=1, window="boxcar", nfft=8, detrend=False)
        close(m.val("p").ravel(), p2)

    def test_pwelch(self, m):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(200)
        m.interp.set_var("x", x.reshape(-1, 1))
        m.run("[p, f] = pwelch(x, 64, 32, 64, 10);")
        f, p = ss.welch(x, fs=10, window=ss.windows.hamming(64, sym=True), nperseg=64, noverlap=32,
                        nfft=64, detrend=False)
        close(m.val("f").ravel(), f)
        close(m.val("p").ravel(), p)
        m.run("p2 = pwelch(x);")
        assert m.val("p2").shape == (129, 1)

    def test_findpeaks(self, m):
        m.run("[pk, lc] = findpeaks([1 3 2 5 4 4 6 1]);")
        close(m.val("pk"), [[3, 5, 6]])
        close(m.val("lc"), [[2, 4, 7]])
        m.run("[pk, lc] = findpeaks([1 3 2 5 4 4 6 1], 'MinPeakDistance', 3);")
        close(m.val("lc"), [[4, 7]])
        m.run("[pk, lc] = findpeaks([1 3 2 5 4 4 6 1]', 'MinPeakHeight', 4);")
        close(m.val("pk"), [[5], [6]])
        m.run("pk = findpeaks([0 2 0 3 0 1 0], 'NPeaks', 2);")
        close(m.val("pk"), [[2, 3]])
        m.run("[pk, lc] = findpeaks([0 2 2 0 1 0]);")  # plateau reported at left edge
        close(m.val("lc"), [[2, 5]])
        m.run("[pk, lc] = findpeaks([0 2 0 3 0], 'SortStr', 'descend');")
        close(m.val("pk"), [[3, 2]])


class TestResampling:
    def test_resample(self, m):
        x = np.sin(np.arange(30) / 4.0)
        m.interp.set_var("x", x.reshape(1, -1))
        m.run("y = resample(x, 3, 2);")
        h = ss.firwin(2 * 10 * 3 + 1, 1 / 3, window=("kaiser", 5.0))
        expected = ss.resample_poly(x, 3, 2, window=h)
        assert m.val("y").shape == (1, 45)
        close(m.val("y").ravel(), expected)

    def test_decimate(self, m):
        x = np.cos(np.arange(60) / 10.0)
        m.interp.set_var("x", x.reshape(-1, 1))
        m.run("y = decimate(x, 3);")
        b, a = ss.cheby1(8, 0.05, 0.8 / 3)
        expected = ss.filtfilt(b, a, x, padtype="odd", padlen=24)[::3]
        assert m.val("y").shape == (20, 1)
        close(m.val("y").ravel(), expected)

    def test_interpft(self, m):
        t = np.arange(5) / 5
        x = np.sin(2 * np.pi * t)
        m.interp.set_var("x", x.reshape(1, -1))
        y = m.num("interpft(x, 20)")
        assert y.shape == (1, 20)
        close(y.ravel(), np.sin(2 * np.pi * np.arange(20) / 20), 1e-10)
        assert m.num("interpft([1;2;3], 6)").shape == (6, 1)


class TestWaveforms:
    def test_sinc_sawtooth_square(self, m):
        close(m.num("sinc([0 0.5 1.5])"), [np.sinc([0, 0.5, 1.5])])
        t = np.linspace(0, 10, 23)
        m.interp.set_var("t", t.reshape(1, -1))
        close(m.num("sawtooth(t)").ravel(), ss.sawtooth(t))
        close(m.num("sawtooth(t, 0.5)").ravel(), ss.sawtooth(t, 0.5))
        close(m.num("square([0 1 pi+0.1 5])"), [[1, 1, -1, -1]])
        close(m.num("square([0 1 2], 25)"), [[1, 1, -1]])

    def test_chirp(self, m):
        t = np.linspace(0, 1, 11)
        m.interp.set_var("t", t.reshape(1, -1))
        close(m.num("chirp(t, 1, 1, 20)").ravel(), np.cos(2 * np.pi * (t + 19 / 2 * t ** 2)))
        close(m.num("chirp(t, 2, 1, 8, 'quadratic')").ravel(), ss.chirp(t, 2, 1, 8, method="quadratic"))
        close(m.num("chirp(t, 2, 1, 8, 'logarithmic')").ravel(), ss.chirp(t, 2, 1, 8, method="logarithmic"))


class TestSmoothing:
    def test_medfilt1(self, m):
        close(m.num("medfilt1([1 9 2 3 8 4])"), [[1, 2, 3, 3, 4, 4]])
        close(m.num("medfilt1([4 9 2 3 8 4], 3, [], 2, 'truncate')"), [[6.5, 4, 3, 3, 4, 6]])
        close(m.num("medfilt1([1;5;2;8;3], 4)"), [[0.5], [1.5], [3.5], [4], [2.5]])

    def test_sgolayfilt(self, m):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(25)
        m.interp.set_var("x", x.reshape(1, -1))
        close(m.num("sgolayfilt(x, 3, 7)").ravel(), ss.savgol_filter(x, 7, 3, mode="interp"))
        with pytest.raises(MatlabError):
            m.run("sgolayfilt(x, 2, 6)")

    def test_detrend(self, m):
        x = np.array([1.0, 2, 4, 3, 5])
        close(m.num("detrend([1 2 4 3 5])").ravel(), ss.detrend(x))
        close(m.num("detrend([1 2 4 3 5], 'constant')").ravel(), x - x.mean())
        t = np.arange(6.0)
        q = 2 * t ** 2 - t + 1
        m.interp.set_var("q", q.reshape(-1, 1))
        close(m.num("detrend(q, 2)"), np.zeros((6, 1)), 1e-9)

    def test_unwrap(self, m):
        p = np.array([0, 3, 6.5, -3, 1])
        close(m.num("unwrap([0 3 6.5 -3 1])").ravel(), np.unwrap(p))
        close(m.num("unwrap([0; 3; 6.5])").ravel(), np.unwrap([0, 3, 6.5]))

    def test_db_conversions(self, m):
        close(m.num("db2mag(6)"), 10 ** (6 / 20))
        close(m.num("mag2db([1 10 100])"), [[0, 20, 40]])
        close(m.num("pow2db([1 10 100])"), [[0, 10, 20]])
        close(m.num("db2pow(-3)"), 10 ** -0.3)

    def test_hilbert_envelope(self, m):
        x = np.cos(np.arange(16) * 2 * np.pi / 8)
        m.interp.set_var("x", x.reshape(1, -1))
        h = m.num("hilbert(x)")
        close(h.ravel(), ss.hilbert(x))
        close(np.imag(h).ravel(), np.sin(np.arange(16) * 2 * np.pi / 8), 1e-12)
        m.run("[up, lo] = envelope(x);")
        close(m.val("up"), np.ones((1, 16)), 1e-12)
        close(m.val("lo"), -np.ones((1, 16)), 1e-12)


# =========================================================================================
# Statistics
# =========================================================================================

class TestDistributions:
    def test_normal(self, m):
        close(m.num("normpdf([-1 0 2])"), [st.norm.pdf([-1, 0, 2])])
        close(m.num("normpdf(1, 2, 3)"), st.norm.pdf(1, 2, 3))
        close(m.num("normcdf([0 1.96])"), [st.norm.cdf([0, 1.96])])
        close(m.num("normcdf(1, 0, 1, 'upper')"), st.norm.sf(1))
        close(m.num("norminv([0.025 0.5])"), [st.norm.ppf([0.025, 0.5])])
        close(m.num("norminv(0.9, 10, 2)"), st.norm.ppf(0.9, 10, 2))
        assert np.isnan(m.num("normpdf(0, 0, -1)"))

    def test_uniform_exponential(self, m):
        close(m.num("unifpdf([0.5 2], 0, 1)"), [[1, 0]])
        close(m.num("unifcdf([0.3 3], 0, 2)"), [[0.15, 1]])
        close(m.num("exppdf([0 1 2], 2)"), [st.expon.pdf([0, 1, 2], scale=2)])
        close(m.num("expcdf(1.5)"), st.expon.cdf(1.5))

    def test_discrete(self, m):
        close(m.num("binopdf(0:5, 5, 0.3)"), [st.binom.pmf(np.arange(6), 5, 0.3)])
        close(m.num("binocdf(2, 10, 0.4)"), st.binom.cdf(2, 10, 0.4))
        close(m.num("binopdf(1.5, 5, 0.3)"), 0.0)
        close(m.num("poisspdf([0 1 4], 2.5)"), [st.poisson.pmf([0, 1, 4], 2.5)])
        close(m.num("poisscdf(3, 2)"), st.poisson.cdf(3, 2))

    def test_t_chi2_f(self, m):
        close(m.num("tpdf([0 1], 4)"), [st.t.pdf([0, 1], 4)])
        close(m.num("tcdf(2, 10)"), st.t.cdf(2, 10))
        close(m.num("tinv([0.9 0.975], 7)"), [st.t.ppf([0.9, 0.975], 7)])
        close(m.num("chi2pdf(3, 4)"), st.chi2.pdf(3, 4))
        close(m.num("chi2cdf(3, 2)"), st.chi2.cdf(3, 2))
        close(m.num("chi2inv(0.95, [1 2])"), [st.chi2.ppf(0.95, [1, 2])])
        close(m.num("fpdf(1.2, 3, 7)"), st.f.pdf(1.2, 3, 7))
        close(m.num("fcdf(2, 3, 7)"), st.f.cdf(2, 3, 7))
        close(m.num("finv(0.95, 2, 10)"), st.f.ppf(0.95, 2, 10))

    def test_gamma_beta(self, m):
        close(m.num("gampdf(2, 3, 1.5)"), st.gamma.pdf(2, 3, scale=1.5))
        close(m.num("gamcdf([1 2], 2)"), [st.gamma.cdf([1, 2], 2)])
        close(m.num("betapdf(0.3, 2, 5)"), st.beta.pdf(0.3, 2, 5))
        close(m.num("betacdf(0.3, 2, 5)"), st.beta.cdf(0.3, 2, 5))

    def test_scalar_expansion_and_errors(self, m):
        assert m.num("normpdf(zeros(2,3), 0, 1)").shape == (2, 3)
        close(m.num("tcdf(1, [2 5 10])"), [st.t.cdf(1, [2, 5, 10])])
        with pytest.raises(MatlabError):
            m.run("normpdf([1 2], 0, [1 2 3])")
        with pytest.raises(MatlabError):
            m.run("tpdf(1)")


class TestRandom:
    def test_shapes_and_reproducibility(self, m):
        m.run("rng(5); a = normrnd(10, 2, 3, 4); rng(5); b = normrnd(10, 2, 3, 4);")
        assert m.val("a").shape == (3, 4)
        close(m.val("a"), m.val("b"))
        m.run("rng(1); u = unifrnd(2, 5, [1000 1]);")
        u = m.val("u")
        assert u.shape == (1000, 1) and u.min() >= 2 and u.max() <= 5
        m.run("rng(1); e = exprnd(3, 1, 5000);")
        assert abs(m.val("e").mean() - 3) < 0.25
        m.run("rng(1); k = binornd(10, 0.5, 2, 2); p = poissrnd(4, 1, 3);")
        k = m.val("k")
        assert k.shape == (2, 2) and np.all(k == np.round(k)) and np.all((k >= 0) & (k <= 10))
        assert m.val("p").shape == (1, 3)
        assert m.num("normrnd([0 100], 1)").shape == (1, 2)

    def test_randsample(self, m):
        m.run("rng(2); s = randsample(10, 4);")
        s = m.val("s")
        assert s.shape == (4, 1) and len(set(s.ravel())) == 4 and s.min() >= 1 and s.max() <= 10
        m.run("rng(2); s = randsample([5 6 7], 2);")
        assert m.val("s").shape == (1, 2) and set(m.val("s").ravel()) <= {5, 6, 7}
        m.run("rng(2); s = randsample(3, 50, true, [0 0 1]);")
        assert np.all(m.val("s") == 3)
        with pytest.raises(MatlabError):
            m.run("randsample(3, 5)")


class TestDescriptive:
    def test_prctile_quantile(self, m):
        x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6])
        m.interp.set_var("x", x.reshape(1, -1))
        close(m.num("prctile(x, [25 50 90])").ravel(), np.percentile(x, [25, 50, 90], method="hazen"))
        close(m.num("quantile(x, 0.3)"), np.quantile(x, 0.3, method="hazen"))
        assert m.num("prctile(x, [25 50 90])").shape == (1, 3)
        A = np.arange(12.0).reshape(4, 3) ** 1.5
        m.interp.set_var("A", A)
        r = m.num("prctile(A, [10 50])")
        assert r.shape == (2, 3)
        close(r, np.percentile(A, [10, 50], axis=0, method="hazen"))
        close(m.num("prctile(A, 50, 2)"), np.percentile(A, 50, axis=1, method="hazen").reshape(-1, 1))
        close(m.num("prctile([1 NaN 3], 50)"), 2)

    def test_iqr_mad(self, m):
        x = np.array([2.0, 4, 4, 5, 7, 9, 10, 30])
        m.interp.set_var("x", x.reshape(-1, 1))
        q = np.percentile(x, [25, 75], method="hazen")
        close(m.num("iqr(x)"), q[1] - q[0])
        close(m.num("mad(x)"), np.mean(np.abs(x - x.mean())))
        close(m.num("mad(x, 1)"), np.median(np.abs(x - np.median(x))))
        close(m.num("mad([1 2; 3 8])"), [[1, 3]])

    def test_zscore(self, m):
        A = np.array([[1.0, 2, 3], [4, 5, 9], [2, 2, 2]])
        m.interp.set_var("A", A)
        m.run("[z, mu, s] = zscore(A);")
        close(m.val("z"), st.zscore(A, axis=0, ddof=1))
        close(m.val("mu"), [A.mean(axis=0)])
        close(m.val("s"), [A.std(axis=0, ddof=1)])
        close(m.num("zscore([1 2 3], 1)"), [st.zscore([1, 2, 3])])
        close(m.num("zscore([5 5 5])"), [[0, 0, 0]])

    def test_skewness_kurtosis(self, m):
        x = np.array([1.0, 2, 3, 10, 4, 4.5])
        m.interp.set_var("x", x.reshape(1, -1))
        close(m.num("skewness(x)"), st.skew(x))
        close(m.num("skewness(x, 0)"), st.skew(x, bias=False))
        close(m.num("kurtosis(x)"), st.kurtosis(x, fisher=False))
        close(m.num("kurtosis(x, 0)"), st.kurtosis(x, fisher=False, bias=False))
        A = np.array([[1.0, 5], [2, 1], [7, 2], [3, 3]])
        m.interp.set_var("A", A)
        close(m.num("skewness(A)"), [st.skew(A, axis=0)])

    def test_cov_corrcoef(self, m):
        X = np.array([[1.0, 2, 0], [3, 5, 1], [4, 4, 7], [6, 1, 2]])
        m.interp.set_var("X", X)
        close(m.num("cov(X)"), np.cov(X, rowvar=False))
        close(m.num("cov(X, 1)"), np.cov(X, rowvar=False, ddof=0))
        close(m.num("cov([1 2 3 4])"), np.var([1, 2, 3, 4], ddof=1))
        close(m.num("cov([1 2 3], [1 3 2])"), np.cov([1, 2, 3], [1, 3, 2]))
        close(m.num("corrcoef(X)"), np.corrcoef(X, rowvar=False))
        m.run("[R, P] = corrcoef([1 2 3 4 5], [2 4 5 4 5]);")
        r, p = st.pearsonr([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        close(m.val("R")[0, 1], r)
        close(m.val("P")[0, 1], p)

    def test_corr(self, m):
        x = np.array([1.0, 2, 3, 4, 5, 6])
        y = np.array([2.0, 1, 4, 3, 7, 5])
        m.interp.set_var("x", x.reshape(-1, 1))
        m.interp.set_var("y", y.reshape(-1, 1))
        close(m.num("corr(x, y)"), st.pearsonr(x, y)[0])
        close(m.num("corr(x, y, 'Type', 'Spearman')"), st.spearmanr(x, y)[0])
        m.run("[r, p] = corr(x, y);")
        close(m.val("p"), st.pearsonr(x, y)[1])
        assert m.num("corr([x y])").shape == (2, 2)


class TestHypothesisTests:
    def test_ttest_one_sample(self, m):
        x = np.array([1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.7])
        m.interp.set_var("x", x.reshape(1, -1))
        m.run("[h, p, ci, stats] = ttest(x, 1);")
        res = st.ttest_1samp(x, 1)
        close(m.val("p"), res.pvalue)
        close(m.val("h"), float(res.pvalue <= 0.05))
        se = x.std(ddof=1) / np.sqrt(x.size)
        c = st.t.ppf(0.975, x.size - 1)
        close(m.val("ci"), [[x.mean() - c * se, x.mean() + c * se]])
        s = m.val("stats")
        close(s.get_field(0, "tstat"), res.statistic)
        close(s.get_field(0, "df"), 6)
        m.run("[h, p] = ttest(x, 1, 'Tail', 'right');")
        close(m.val("p"), st.ttest_1samp(x, 1, alternative="greater").pvalue)

    def test_ttest_paired_and_matrix(self, m):
        a = np.array([5.0, 6, 7, 8, 9])
        b = np.array([4.5, 6.2, 6.1, 7.0, 8.4])
        m.interp.set_var("a", a.reshape(-1, 1))
        m.interp.set_var("b", b.reshape(-1, 1))
        m.run("[h, p, ci] = ttest(a, b);")
        close(m.val("p"), st.ttest_rel(a, b).pvalue)
        assert m.val("ci").shape == (2, 1)
        m.run("[h, p] = ttest([a b]);")
        assert m.val("p").shape == (1, 2)

    def test_ttest2(self, m):
        x = np.array([1.0, 2, 3, 4, 5])
        y = np.array([3.0, 4, 5, 6, 7, 8])
        m.interp.set_var("x", x.reshape(1, -1))
        m.interp.set_var("y", y.reshape(1, -1))
        m.run("[h, p, ci, stats] = ttest2(x, y);")
        res = st.ttest_ind(x, y)
        close(m.val("p"), res.pvalue)
        close(m.val("h"), 1)
        close(m.val("stats").get_field(0, "tstat"), res.statistic)
        assert m.val("ci").shape == (1, 2)
        m.run("[h, p] = ttest2(x, y, 'Vartype', 'unequal');")
        close(m.val("p"), st.ttest_ind(x, y, equal_var=False).pvalue)

    def test_ranksum(self, m):
        m.run("[p, h, s] = ranksum([1 2 3 4 5], [6 7 8 9 10 11]);")
        close(m.val("p"), st.mannwhitneyu([1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], method="exact").pvalue)
        close(m.val("h"), 1)
        close(m.val("s").get_field(0, "ranksum"), 15)
        x, y = np.arange(1.0, 16), np.arange(5.0, 25)
        m.interp.set_var("x", x.reshape(1, -1))
        m.interp.set_var("y", y.reshape(1, -1))
        p = m.num("ranksum(x, y)")
        close(p, st.mannwhitneyu(x, y, method="asymptotic", use_continuity=True).pvalue, 1e-10)

    def test_anova1(self, m):
        A = np.array([[1.0, 2, 3], [2, 3, 4], [3, 4, 6], [1, 2, 5]])
        m.interp.set_var("A", A)
        m.run("[p, tbl, stats] = anova1(A, [], 'off');")
        res = st.f_oneway(A[:, 0], A[:, 1], A[:, 2])
        close(m.val("p"), res.pvalue)
        tbl = m.val("tbl")
        assert isinstance(tbl, Cell) and tbl.shape == (4, 6)
        close(tbl.data[1, 4], res.statistic)
        close(tbl.data[2, 2], 9)
        close(m.val("stats").get_field(0, "means"), [A.mean(axis=0)])
        m.run("p2 = anova1([1 2 3 2 3 4 3 4 6], {'a','a','a','b','b','b','c','c','c'}, 'off');")
        close(m.val("p2"), st.f_oneway([1, 2, 3], [2, 3, 4], [3, 4, 6]).pvalue)

    def test_regress(self, m):
        X = np.column_stack([np.ones(6), np.arange(1.0, 7), np.array([2.0, 1, 4, 3, 6, 5])])
        y = np.array([1.1, 2.3, 2.9, 4.2, 5.1, 5.8])
        m.interp.set_var("X", X)
        m.interp.set_var("y", y.reshape(-1, 1))
        m.run("[b, bint, r, rint, stats] = regress(y, X);")
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        close(m.val("b").ravel(), b)
        r = y - X @ b
        close(m.val("r").ravel(), r)
        s2 = r @ r / 3
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * s2)
        t = st.t.ppf(0.975, 3)
        close(m.val("bint"), np.column_stack([b - t * se, b + t * se]))
        r2 = 1 - (r @ r) / np.sum((y - y.mean()) ** 2)
        close(m.val("stats")[0, 0], r2)
        close(m.val("stats")[0, 3], s2)
        assert m.val("rint").shape == (6, 2)


class TestHistTab:
    def test_histc(self, m):
        m.run("[n, bin] = histc([1 2 2 3 5 6 7], [0 2 4 6]);")
        close(m.val("n"), [[1, 3, 1, 1]])
        close(m.val("bin"), [[1, 2, 2, 2, 3, 4, 0]])
        assert m.num("histc([1;2;3], [0 2 4])").shape == (3, 1)
        close(m.num("histc([1 2; 3 1], [1 2 3])"), [[1, 1], [0, 1], [1, 0]])

    def test_tabulate(self, m):
        t = m.num("tabulate([1 2 2 5])")
        close(t, [[1, 1, 25], [2, 2, 50], [3, 0, 0], [4, 0, 0], [5, 1, 25]])
        t = m.num("tabulate([0.5 1.5 1.5])")
        close(t, [[0.5, 1, 100 / 3], [1.5, 2, 200 / 3]])
        out = m.run("tabulate([1 1 2])")
        assert "66.67%" in out
        m.run("c = tabulate({'x','y','x'});")
        c = m.val("c")
        assert c.shape == (2, 3)
        close(c.data[0, 1], 2)


class TestNanFunctions:
    def test_nan_reductions(self, m):
        A = np.array([[1.0, np.nan, 3], [4, 5, np.nan]])
        m.interp.set_var("A", A)
        close(m.num("nanmean(A)"), [np.nanmean(A, axis=0)])
        close(m.num("nanmean(A, 2)"), np.nanmean(A, axis=1).reshape(-1, 1))
        close(m.num("nansum(A)"), [np.nansum(A, axis=0)])
        close(m.num("nanstd([1 NaN 3 4])"), np.nanstd([1, np.nan, 3, 4], ddof=1))
        close(m.num("nanstd([1 NaN 3 4], 1)"), np.nanstd([1, np.nan, 3, 4]))
        close(m.num("nanmax(A)"), [np.nanmax(A, axis=0)])
        close(m.num("nanmin(A, [], 2)"), np.nanmin(A, axis=1).reshape(-1, 1))
        close(m.num("nanmax([1 NaN], [0 2])"), [[1, 2]])
        m.run("[v, i] = nanmin([4 NaN 1 2]);")
        close(m.val("v"), 1)
        close(m.val("i"), 3)
        assert np.isnan(m.num("nanmean([NaN NaN])"))
        close(m.num("nansum([NaN NaN])"), 0)


def _mov_ref(x, before, after, fn):
    n = len(x)
    return np.array([fn(x[max(0, i - before):min(n, i + after + 1)]) for i in range(n)])


class TestMoving:
    def test_moving_basic(self, m):
        x = np.array([4.0, 8, 6, -1, -2, -3, -1, 3, 4, 5])
        m.interp.set_var("x", x.reshape(1, -1))
        close(m.num("movmean(x, 3)").ravel(), _mov_ref(x, 1, 1, np.mean))
        close(m.num("movmean(x, 4)").ravel(), _mov_ref(x, 2, 1, np.mean))
        close(m.num("movsum(x, [2 0])").ravel(), _mov_ref(x, 2, 0, np.sum))
        close(m.num("movmax(x, 3)").ravel(), _mov_ref(x, 1, 1, np.max))
        close(m.num("movmin(x, 5)").ravel(), _mov_ref(x, 2, 2, np.min))
        close(m.num("movmedian(x, 3)").ravel(), _mov_ref(x, 1, 1, np.median))
        close(m.num("movstd(x, 3)").ravel(), _mov_ref(x, 1, 1, lambda v: np.std(v, ddof=1)))
        close(m.num("movstd(x, 3, 1)").ravel(), _mov_ref(x, 1, 1, np.std))
        close(m.num("movvar(x, 4)").ravel(), _mov_ref(x, 2, 1, lambda v: np.var(v, ddof=1)))
        assert m.num("movmean(x', 3)").shape == (10, 1)

    def test_moving_options(self, m):
        close(m.num("movmean([1 2 3 4 5], 3, 'Endpoints', 'discard')"), [[2, 3, 4]])
        r = m.num("movmean([1 2 3 4 5], 3, 'Endpoints', 'fill')")
        assert np.isnan(r[0, 0]) and np.isnan(r[0, -1])
        close(m.num("movmean([1 NaN 3], 3, 'omitnan')"), [[1, 2, 3]])
        assert np.isnan(m.num("movmean([1 NaN 3], 3)")[0, 1])
        A = np.arange(12.0).reshape(3, 4)
        m.interp.set_var("A", A)
        close(m.num("movsum(A, 2, 2)"), np.column_stack([A[:, 0], A[:, 0] + A[:, 1], A[:, 1] + A[:, 2],
                                                        A[:, 2] + A[:, 3]]))

    def test_smoothdata(self, m):
        x = np.array([1.0, 5, 2, 6, 3, 7, 4])
        m.interp.set_var("x", x.reshape(1, -1))
        close(m.num("smoothdata(x, 'movmean', 3)").ravel(), _mov_ref(x, 1, 1, np.mean))
        close(m.num("smoothdata(x, 'movmedian', 3)").ravel(), _mov_ref(x, 1, 1, np.median))
        w = np.exp(-0.5 * (np.arange(-2, 3) / 0.8) ** 2)
        g = m.num("smoothdata(x, 'gaussian', 5)").ravel()
        close(g[3], np.dot(w, x[1:6]) / w.sum())
        assert m.num("smoothdata(x)").shape == (1, 7)

    def test_normalize_rescale(self, m):
        A = np.array([[1.0, 10], [2, 30], [6, 20]])
        m.interp.set_var("A", A)
        close(m.num("normalize(A)"), st.zscore(A, axis=0, ddof=1))
        close(m.num("normalize(A, 'range')"), (A - A.min(0)) / (A.max(0) - A.min(0)))
        close(m.num("normalize(A, 'range', [-1 1])"), -1 + 2 * (A - A.min(0)) / (A.max(0) - A.min(0)))
        close(m.num("normalize(A, 'norm')"), A / np.linalg.norm(A, axis=0))
        close(m.num("normalize(A, 'norm', 1)"), A / np.abs(A).sum(axis=0))
        close(m.num("normalize(A, 'center')"), A - A.mean(0))
        close(m.num("normalize(A, 2, 'range')"), (A - A.min(1, keepdims=True)) / np.ptp(A, axis=1, keepdims=True))
        close(m.num("rescale([1 2 3 5])"), [[0, 0.25, 0.5, 1]])
        close(m.num("rescale(A, -1, 1)"), -1 + 2 * (A - A.min()) / (A.max() - A.min()))
        close(m.num("rescale([1 2 3], 'InputMin', 2)"), [[0, 0, 1]])


class TestDistances:
    def test_pdist_squareform(self, m):
        X = np.array([[0.0, 0], [3, 4], [6, 8], [1, 1]])
        m.interp.set_var("X", X)
        from scipy.spatial.distance import pdist, squareform
        close(m.num("pdist(X)").ravel(), pdist(X))
        assert m.num("pdist(X)").shape == (1, 6)
        close(m.num("pdist(X, 'cityblock')").ravel(), pdist(X, "cityblock"))
        close(m.num("pdist(X, 'minkowski', 3)").ravel(), pdist(X, "minkowski", p=3))
        close(m.num("squareform(pdist(X))"), squareform(pdist(X)))
        close(m.num("squareform(squareform(pdist(X)))").ravel(), pdist(X))

    def test_kmeans(self, m):
        m.run("[idx, C, sumd] = kmeans([1 1; 1.2 1; 5 5; 5.1 5.2], 2, 'Start', [0 0; 6 6]);")
        close(m.val("idx"), [[1], [1], [2], [2]])
        close(m.val("C"), [[1.1, 1.0], [5.05, 5.1]])
        close(m.val("sumd").ravel(), [0.02, 0.025])
        m.run("rng(0); idx = kmeans([1 1; 1.2 1; 5 5; 5.1 5.2; 9 9; 9 9.1], 3, 'Replicates', 3);")
        idx = m.val("idx").ravel()
        assert idx[0] == idx[1] and idx[2] == idx[3] and idx[4] == idx[5]
        assert len({idx[0], idx[2], idx[4]}) == 3


# =========================================================================================
# Optimization
# =========================================================================================

class TestOptim:
    def test_linprog(self, m):
        m.run("[x, fval, flag] = linprog([-1 -2], [1 1; 1 -1], [4; 2], [], [], [0 0], []);")
        close(m.val("x"), [[0], [4]], 1e-7)
        close(m.val("fval"), -8, 1e-7)
        close(m.val("flag"), 1)
        m.run("[x, fval] = linprog([1 2 3], [], [], [1 1 1], 6, [0 0 0], [3 3 3]);")
        close(m.val("x"), [[3], [3], [0]], 1e-7)
        m.run("[x, fval, flag] = linprog([1 1], [-1 -1], -5, [], [], [0 0], [1 1]);")
        close(m.val("flag"), -2)
        assert m.val("x").size == 0

    def test_intlinprog(self, m):
        m.run("[x, fval, flag] = intlinprog([-1 -1], [1 2], [2 1; 1 3], [4.5; 6], [], [], [0 0], []);")
        x = m.val("x").ravel()
        assert np.allclose(x, np.round(x))
        close(m.val("fval"), -2, 1e-7)
        close(m.val("flag"), 1)
        m.run("x = intlinprog([-3 -2], 1, [1 1], 3.5, [], [], [0 0], [2.5 10]);")
        x = m.val("x").ravel()
        assert x[0] == np.round(x[0])
        close(-3 * x[0] - 2 * x[1], -9.0, 1e-7)

    def test_quadprog(self, m):
        m.run("[x, fval, flag] = quadprog([2 0; 0 2], [-2 -5], [1 2], 3);")
        close(m.val("x"), [[0.4], [1.3]], 1e-6)
        close(m.val("fval"), 0.4 ** 2 + 1.3 ** 2 - 2 * 0.4 - 5 * 1.3, 1e-6)
        close(m.val("flag"), 1)
        close(m.num("quadprog([2 0; 0 2], [-2 -4])"), [[1], [2]])
        m.run("x = quadprog(eye(2), [0 0], [], [], [1 1], 1, [0.7 0], []);")
        close(m.val("x"), [[0.7], [0.3]], 1e-6)

    def test_fmincon(self, m):
        m.run("[x, fval, flag] = fmincon(@(x) (x(1)-1)^2 + (x(2)-2)^2, [0 0], [1 1], 1);")
        close(m.val("x"), [[0, 1]], 1e-5)
        close(m.val("fval"), 2, 1e-5)
        close(m.val("flag"), 1)
        m.run("x = fmincon(@(x) x(1)+x(2), [1; 1], [], [], [], [], [], [], "
              "@(x) deal(x(1)^2 + x(2)^2 - 2, []));")
        close(m.val("x"), [[-1], [-1]], 1e-5)
        m.run("x = fmincon(@(x) sum((x - 3).^2), [0 0 0], [], [], [], [], [0 0 0], [1 2 5]);")
        close(m.val("x"), [[1, 2, 3]], 1e-5)

    def test_lsqnonneg(self, m):
        from scipy.optimize import nnls
        C = np.array([[1.0, 0.5], [0.2, 1], [1, 1], [0.3, -0.4]])
        d = np.array([1.0, -1, 0.5, 2])
        m.interp.set_var("C", C)
        m.interp.set_var("d", d.reshape(-1, 1))
        m.run("[x, rn, res] = lsqnonneg(C, d);")
        x, rnorm = nnls(C, d)
        close(m.val("x").ravel(), x)
        close(m.val("rn"), rnorm ** 2)
        close(m.val("res").ravel(), d - C @ x)

    def test_lsqcurvefit_lsqnonlin(self, m):
        m.run("x = lsqcurvefit(@(p, t) p(1)*exp(p(2)*t), [1 0], 0:4, 2*exp(-0.5*(0:4)));")
        close(m.val("x"), [[2, -0.5]], 1e-6)
        m.run("x = lsqcurvefit(@(p, t) p(1)*t + p(2), [0 0], [1 2 3], [2 4 6], [0 0.5], [10 10]);")
        x = m.val("x").ravel()
        assert x[1] >= 0.5 - 1e-9
        m.run("[x, rn] = lsqnonlin(@(p) [p(1)-1, p(2)+2, p(1)*p(2)+2], [0 0]);")
        close(m.val("x"), [[1, -2]], 1e-6)
        close(m.val("rn"), 0, 1e-8)


# =========================================================================================
# Extras
# =========================================================================================

class TestIndexing:
    def test_sub2ind_ind2sub(self, m):
        close(m.num("sub2ind([3 4], [1 2 3], [1 2 4])"), [[1, 5, 12]])
        close(m.num("sub2ind([2 3 4], 2, 3, 4)"), np.ravel_multi_index((1, 2, 3), (2, 3, 4), order="F") + 1)
        m.run("[r, c] = ind2sub([3 4], [1 5 12]);")
        close(m.val("r"), [[1, 2, 3]])
        close(m.val("c"), [[1, 2, 4]])
        m.run("[r, c, p] = ind2sub([2 2 2], 7);")
        close([m.val("r"), m.val("c"), m.val("p")], [[[1]], [[2]], [[2]]])
        m.run("[r, c] = ind2sub([2 2 2], 7);")
        close(m.val("c"), 4)
        close(m.num("ind2sub([3 4], [7; 8])"), [[7], [8]])
        with pytest.raises(MatlabError):
            m.run("sub2ind([2 2], 3, 1)")

    def test_perms(self, m):
        close(m.num("perms([1 2 3])"), [[3, 2, 1], [3, 1, 2], [2, 3, 1], [2, 1, 3], [1, 3, 2], [1, 2, 3]])
        assert m.num("perms(1:4)").shape == (24, 4)
        p = m.interp.eval_expression("perms('ab')")
        assert p.dtype.kind == "U" and "".join(p[0]) == "ba"

    def test_bsxfun_idivide_vecnorm(self, m):
        close(m.num("bsxfun(@minus, [1 2; 3 4], [1 2])"), [[0, 0], [2, 2]])
        close(m.num("bsxfun(@(a, b) a.*b, [1; 2], [1 2 3])"), [[1, 2, 3], [2, 4, 6]])
        r = m.num("idivide(int32([7 -7 8]), int32(2))")
        assert r.dtype == np.int32
        close(r, [[3, -3, 4]])
        close(m.num("idivide(int32([7 -7]), int32(2), 'floor')"), [[3, -4]])
        close(m.num("idivide(int32([7 -7]), int32(2), 'ceil')"), [[4, -3]])
        close(m.num("idivide(int32([5 -5]), int32(2), 'round')"), [[3, -3]])
        close(m.num("idivide(int16(5), int16(0))"), 32767)
        A = np.array([[3.0, 4], [6, 8]])
        m.interp.set_var("A", A)
        close(m.num("vecnorm(A)"), [np.linalg.norm(A, axis=0)])
        close(m.num("vecnorm(A, 2, 2)"), np.linalg.norm(A, axis=1).reshape(-1, 1))
        close(m.num("vecnorm(A, 1)"), [[9, 12]])
        close(m.num("vecnorm([1 -3 2], Inf)"), 3)

    def test_nthargout(self, m):
        close(m.num("nthargout(2, @max, [1 5 3])"), 2)
        m.run("c = nthargout([1 2], @size, ones(2, 3));")
        c = m.val("c")
        assert isinstance(c, Cell)
        close(c.data[0, 1], 3)


class TestSscanf:
    def test_numbers(self, m):
        close(m.num("sscanf('1 2 3', '%d')"), [[1], [2], [3]])
        close(m.num("sscanf('3.5, 4.25, -1e2', '%f,')"), [[3.5], [4.25], [-100]])
        close(m.num("sscanf('x=10 y=20', 'x=%d y=%d')"), [[10], [20]])
        close(m.num("sscanf('1 2 3 4 5 6', '%d', [2 3])"), [[1, 3, 5], [2, 4, 6]])
        close(m.num("sscanf('1 2 3 4 5', '%d', [2 Inf])"), [[1, 3, 5], [2, 4, 0]])
        close(m.num("sscanf('1 2 3 4', '%g', 2)"), [[1], [2]])
        close(m.num("sscanf('ff 10', '%x')"), [[255], [16]])
        close(m.num("sscanf('12 skip 13', '%d %*s %d')"), [[12], [13]])

    def test_text_and_count(self, m):
        assert m.str("sscanf('hello', '%s')") == "hello"
        m.run("[a, n] = sscanf('12abc', '%d');")
        close(m.val("a"), 12)
        close(m.val("n"), 1)
        close(m.num("sscanf('3 apples', '%d %s')"), [[3], [97], [112], [112], [108], [101], [115]])


class TestValidation:
    def test_validateattributes(self, m):
        m.run("validateattributes(5, {'numeric'}, {'positive', 'scalar', 'integer'})")
        m.run("validateattributes([1 2 3], {'double'}, {'size', [1 3], '>=', 1, 'increasing', 'row'})")
        m.run("validateattributes(int8(4), {'integer'}, {'even', 'nonnegative'})")
        with pytest.raises(MatlabError, match="positive"):
            m.run("validateattributes(-1, {'numeric'}, {'positive'})")
        with pytest.raises(MatlabError, match="types"):
            m.run("validateattributes('a', {'numeric'}, {})")
        with pytest.raises(MatlabError, match="myfn"):
            m.run("validateattributes([1 2], {'double'}, {'scalar'}, 'myfn', 'x')")
        with pytest.raises(MatlabError):
            m.run("validateattributes([1 NaN], {'double'}, {'nonnan'})")
        with pytest.raises(MatlabError):
            m.run("validateattributes([3 2], {'double'}, {'<', 3})")

    def test_validatestring(self, m):
        assert m.str("validatestring('app', {'apple', 'banana'})") == "apple"
        assert isinstance(m.interp.eval_expression('validatestring("BAN", {\'apple\', \'banana\'})'), MString)
        assert m.str("validatestring('app', {'apple', 'applesauce'})") == "apple"
        with pytest.raises(MatlabError):
            m.run("validatestring('a', {'apple', 'apricot'})")
        with pytest.raises(MatlabError):
            m.run("validatestring('z', {'apple'})")


class TestText:
    def test_strjust_pad_strip(self, m):
        r = m.interp.eval_expression("strjust(['a  '; 'bb '], 'right')")
        assert ["".join(row) for row in r] == ["  a", " bb"]
        r = m.interp.eval_expression("strjust(['  a'; ' bb'])")
        assert ["".join(row) for row in r] == ["  a", " bb"]
        r = m.interp.eval_expression("strjust(['  a'; 'bb '], 'left')")
        assert ["".join(row) for row in r] == ["a  ", "bb "]
        assert m.str("pad('ab', 5)") == "ab   "
        assert m.str("pad('ab', 5, 'left', '*')") == "***ab"
        assert m.str("pad(\"x\", 3, 'both')") == " x "
        c = m.interp.eval_expression("pad({'a', 'abc'})")
        assert [m_ for m_ in map(lambda v: "".join(v.ravel()), c.data.ravel())] == ["a  ", "abc"]
        assert m.str("strip('  hi  ')") == "hi"
        assert m.str("strip('--hi--', '-')") == "hi"
        assert m.str("strip('  hi  ', 'left')") == "hi  "

    def test_extract(self, m):
        assert m.str("extractBefore('hello world', ' ')") == "hello"
        assert m.str("extractAfter('hello world', 'o')") == " world"
        assert m.str("extractBefore('abcdef', 3)") == "ab"
        assert m.str("extractAfter('abcdef', 2)") == "cdef"
        assert isinstance(m.interp.eval_expression('extractBefore("abc", 3)'), MString)
        c = m.interp.eval_expression("extractBetween('a[1]b[22]', '[', ']')")
        assert isinstance(c, Cell) and c.shape == (2, 1)
        assert to_s(c.data[1, 0]) == "22"
        assert m.str("extractBetween('abcdef', 2, 4)") == "bcd"
        assert m.str('extractBetween("x<y>z", "<", ">")') == "y"
        c = m.interp.eval_expression("extractBetween('x<y>z', '<', '>', 'Boundaries', 'inclusive')")
        assert to_s(c.data[0, 0]) == "<y>"

    def test_insert_count_replace_erase(self, m):
        assert m.str("insertAfter('abcabc', 'b', '-')") == "ab-cab-c"
        assert m.str("insertBefore('abcabc', 'c', '+')") == "ab+cab+c"
        assert m.str("insertBefore('hello', 1, '>> ')") == ">> hello"
        assert m.str('insertAfter("ab", 1, "X")') == "aXb"
        close(m.num("count('banana', 'an')"), 2)
        close(m.num("count('banana', {'a', 'n'})"), 5)
        close(m.num("count({'aaa', 'ba'}, 'a')"), [[3, 1]])
        close(m.num("count('ABab', 'a', 'IgnoreCase', true)"), 2)
        assert m.str("replace('a cat a hat', {'cat', 'hat'}, {'dog', 'cap'})") == "a dog a cap"
        assert m.str('replace("aaa", "a", "bb")') == "bbbbbb"
        assert m.str("replace('abc', {'a', 'b'}, 'X')") == "XXc"
        assert m.str("erase('hello world', 'o')") == "hell wrld"
        assert m.str("erase('abcabc', {'a', 'c'})") == "bb"

    def test_split_join(self, m):
        c = m.interp.eval_expression("split('a b  c')")
        assert c.shape == (3, 1) and [to_s(x) for x in c.data.ravel()] == ["a", "b", "c"]
        c = m.interp.eval_expression("split('a,b;c', {',', ';'})")
        assert [to_s(x) for x in c.data.ravel()] == ["a", "b", "c"]
        c = m.interp.eval_expression("split({'a b'; 'c d'})")
        assert c.shape == (2, 2) and to_s(c.data[1, 0]) == "c"
        c = m.interp.eval_expression("join({'a', 'b', 'c'})")
        assert c.shape == (1, 1) and to_s(c.data[0, 0]) == "a b c"
        c = m.interp.eval_expression("join({'a', 'b'}, '-')")
        assert to_s(c.data[0, 0]) == "a-b"
        c = m.interp.eval_expression("join({'a', 'b'; 'c', 'd'}, ',')")
        assert c.shape == (2, 1) and to_s(c.data[1, 0]) == "c,d"
        c = m.interp.eval_expression("join({'x', 'y', 'z'}, {'+', '='})")
        assert to_s(c.data[0, 0]) == "x+y=z"

    def test_compose(self, m):
        c = m.interp.eval_expression("compose('%d', [1 2 3])")
        assert c.shape == (1, 3) and [to_s(x) for x in c.data.ravel()] == ["1", "2", "3"]
        c = m.interp.eval_expression("compose('%d-%d', [1 2; 3 4])")
        assert c.shape == (2, 1) and to_s(c.data[1, 0]) == "3-4"
        assert m.str('compose("%.2f", pi)') == "3.14"
        c = m.interp.eval_expression("compose('%s=%d', {'a', 'b'}, [1 2])")
        assert [to_s(x) for x in c.data.ravel()] == ["a=1", "b=2"]


def to_s(v):
    from mylab.values import to_str
    return to_str(v)


class TestCells:
    def test_mat2cell(self, m):
        m.run("c = mat2cell(magic(4), [2 2], [3 1]);")
        c = m.val("c")
        M = np.array([[16.0, 2, 3, 13], [5, 11, 10, 8], [9, 7, 6, 12], [4, 14, 15, 1]])
        assert c.shape == (2, 2)
        close(c.data[0, 0], M[:2, :3])
        close(c.data[1, 1], M[2:, 3:])
        m.run("d = mat2cell((1:6)', [1 2 3]);")
        assert m.val("d").shape == (3, 1)
        close(m.val("d").data[2, 0], [[4], [5], [6]])
        with pytest.raises(MatlabError):
            m.run("mat2cell(ones(3), [1 1], 3)")

    def test_celldisp(self, m):
        out = m.run("celldisp({1, 'two', [3 4]})")
        assert "C{1} = 1" in out and "C{2} = 'two'" in out and "C{3} =" in out
        out = m.run("q = {5; 6}; q{1} = {7, 8}; celldisp(q, 'Q')")
        assert "Q{1}{1} = 7" in out and "Q{2} = 6" in out
        out = m.run("celldisp({1 2; 3 4})")
        assert "C{2,1} = 3" in out


class TestMatrixTests:
    def test_structure(self, m):
        assert m.num("isdiag(eye(3))").item() and not m.num("isdiag([1 2; 0 1])").item()
        assert m.num("istriu([1 2; 0 3])").item() and not m.num("istriu([1 0; 2 3])").item()
        assert m.num("istril([1 0; 2 3])").item() and not m.num("istril([1 2; 0 3])").item()
        assert m.num("istriu([1 2 3; 0 4 5])").item()
        assert m.num("issymmetric([1 2; 2 1])").item() and not m.num("issymmetric([1 2; 3 1])").item()
        assert m.num("issymmetric([0 1; -1 0], 'skew')").item()
        assert not m.num("issymmetric([1 2 3])").item()
        assert m.num("issymmetric([1 2; 2+1e-12 1], 1e-10)").item()
        assert m.num("ishermitian([1 1i; -1i 2])").item()
        assert not m.num("issymmetric([1 1i; -1i 2])").item()
        assert m.num("ishermitian([1i 2; -2 0], 'skew')").item()


class TestGeometry:
    def test_wrap(self, m):
        close(m.num("wrapTo180([190 -190 180 -180 540 360 0])"), [[-170, 170, 180, -180, 180, 0, 0]])
        close(m.num("wrapTo360([-10 0 360 720 370])"), [[350, 0, 360, 360, 10]])
        close(m.num("wrapToPi([3*pi/2 -pi pi 3*pi])"), [[-np.pi / 2, -np.pi, np.pi, np.pi]])
        close(m.num("wrapTo2Pi([-pi/2 2*pi 0])"), [[3 * np.pi / 2, 2 * np.pi, 0]])

    def test_coordinates(self, m):
        x, y, z = np.array([1.0, -2, 0.5]), np.array([1.0, 3, -4]), np.array([2.0, -1, 0])
        for n, v in zip("xyz", (x, y, z)):
            m.interp.set_var(n, v.reshape(1, -1))
        m.run("[th, r, zz] = cart2pol(x, y, z);")
        close(m.val("th").ravel(), np.arctan2(y, x))
        close(m.val("r").ravel(), np.hypot(x, y))
        close(m.val("zz").ravel(), z)
        m.run("[xx, yy] = pol2cart(th, r);")
        close(m.val("xx").ravel(), x)
        close(m.val("yy").ravel(), y)
        m.run("[az, el, rr] = cart2sph(x, y, z);")
        close(m.val("az").ravel(), np.arctan2(y, x))
        close(m.val("el").ravel(), np.arctan2(z, np.hypot(x, y)))
        close(m.val("rr").ravel(), np.sqrt(x ** 2 + y ** 2 + z ** 2))
        m.run("[a, b, c] = sph2cart(az, el, rr);")
        close(m.val("a").ravel(), x)
        close(m.val("c").ravel(), z)
        close(m.num("cart2pol([1 1; 0 2])"), [[np.pi / 4, np.sqrt(2)], [np.pi / 2, 2]])


class TestSpecial:
    def test_bessel(self, m):
        close(m.num("besselj(0, [0.5 1 2])"), [sp.jv(0, [0.5, 1, 2])])
        close(m.num("besselj([0 1 2], 1)"), [sp.jv([0, 1, 2], 1)])
        close(m.num("bessely(1, 2.5)"), sp.yv(1, 2.5))
        close(m.num("besseli(1, 1)"), sp.iv(1, 1))
        close(m.num("besselk(0, 1)"), sp.kv(0, 1))
        close(m.num("besseli(0, 10, 1)"), sp.ive(0, 10))
        close(m.num("besselk(1, 3, 1)"), sp.kve(1, 3))
        r = m.num("besselj(0.5, -1)")
        close(r, sp.jv(0.5, -1 + 0j))
        assert np.iscomplexobj(r)

    def test_airy_ellipke(self, m):
        ai, aip, bi, bip = sp.airy(1.3)
        close(m.num("airy(1.3)"), ai)
        close(m.num("airy(1, 1.3)"), aip)
        close(m.num("airy(2, 1.3)"), bi)
        close(m.num("airy(3, 1.3)"), bip)
        m.run("[K, E] = ellipke([0 0.5 0.9]);")
        close(m.val("K"), [sp.ellipk([0, 0.5, 0.9])])
        close(m.val("E"), [sp.ellipe([0, 0.5, 0.9])])
        with pytest.raises(MatlabError):
            m.run("ellipke(2)")

    def test_expint_gammainc_betainc_psi(self, m):
        close(m.num("expint([0.5 1 3])"), [sp.exp1([0.5, 1, 3])])
        close(m.num("expint(-1)"), -sp.expi(1) - 1j * np.pi)
        close(m.num("gammainc(1, 2)"), sp.gammainc(2, 1))
        close(m.num("gammainc([1 3], 2, 'upper')"), [sp.gammaincc(2, [1, 3])])
        close(m.num("betainc(0.5, 2, 3)"), sp.betainc(2, 3, 0.5))
        close(m.num("betainc(0.2, 2, 3, 'upper')"), 1 - sp.betainc(2, 3, 0.2))
        close(m.num("psi([1 2.5])"), [sp.digamma([1, 2.5])])
        close(m.num("psi(1, 1)"), np.pi ** 2 / 6)
        close(m.num("psi(2, 3)"), sp.polygamma(2, 3))


class TestTypecast:
    def test_typecast(self, m):
        r = m.num("typecast(single(1), 'uint32')")
        assert r.dtype == np.uint32
        close(r, np.frombuffer(np.float32(1).tobytes(), np.uint32).reshape(1, 1))
        close(m.num("typecast(uint8([1 0 0 0 2 0 0 0]), 'uint32')"), [[1, 2]])
        assert m.num("typecast(uint8([1 0 0 0 2 0 0 0])', 'uint32')").shape == (2, 1)
        close(m.num("typecast(int8(-1), 'uint8')"), 255)
        close(m.num("typecast(1, 'uint8')"), [np.frombuffer(np.float64(1).tobytes(), np.uint8)])
        with pytest.raises(MatlabError):
            m.run("typecast(uint8([1 2 3]), 'uint16')")
