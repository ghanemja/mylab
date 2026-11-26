"""Probability distributions: function forms, fits, distribution objects, multivariate and plots.

Expected values come from closed forms, from scipy.stats where the parameterisation is known to
match, or from GNU Octave's statistics package run as a black box (noted where used).
"""

import math

import numpy as np
import pytest
import scipy.stats as st

from mylab.errors import MatlabError

X10 = "[1.2 0.8 2.5 3.1 0.6 1.9 2.2 4.0 1.1 0.9]"
X10V = np.array([1.2, 0.8, 2.5, 3.1, 0.6, 1.9, 2.2, 4.0, 1.1, 0.9])


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def test_closed_form_values(m):
    close(m.num("wblcdf(1.5, 2, 3)"), 1 - math.exp(-(1.5 / 2) ** 3))
    close(m.num("wblpdf(1.5, 2, 3)"), 3 / 2 * (1.5 / 2) ** 2 * math.exp(-(1.5 / 2) ** 3))
    close(m.num("logncdf(exp(1), 1, 2)"), 0.5)
    close(m.num("evcdf(0, 0, 1)"), 1 - math.exp(-1))
    close(m.num("gevcdf(0.3, 0, 1, 0)"), math.exp(-math.exp(-0.3)))
    close(m.num("gpcdf(0.7, 0, 2, 0)"), 1 - math.exp(-0.35))
    close(m.num("raylcdf(1.5, 2)"), 1 - math.exp(-1.5 ** 2 / 8))
    close(m.num("geopdf(3, 0.2)"), 0.2 * 0.8 ** 3)
    close(m.num("nbinpdf(2, 3, 0.5)"), math.comb(4, 2) * 0.5 ** 5)
    close(m.num("hygepdf(1, 10, 3, 4)"), 0.5)
    close(m.num("unidcdf(2, 5)"), 0.4)
    close(m.num("unidpdf(2.5, 5)"), 0.0)
    close(m.num("expinv(0.5, 3)"), 3 * math.log(2))
    close(m.num("unifinv(0.25, 2, 6)"), 3.0)
    close(m.num("gampdf(2, 3)"), 2 ** 2 * math.exp(-2) / 2)
    close(m.num("betainv(0.6875, 2, 3)"), 0.5)
    close(m.num("poissinv(0.5, 3)"), 3)
    close(m.num("binoinv(0.5, 5, 0.5)"), 2)


def test_upper_tail_and_array_expansion(m):
    close(m.num("normcdf(10, 'upper')"), st.norm.sf(10), 1e-12)
    close(m.num("chi2cdf(200, 3, 'upper')"), st.chi2.sf(200, 3), 1e-10)
    close(m.num("tcdf([1 2; 3 4], 5)"), st.t.cdf([[1, 2], [3, 4]], 5))
    close(m.num("normpdf([0 1], [0 1], 2)").ravel(), st.norm.pdf([0, 1], [0, 1], 2))
    assert np.isnan(m.num("normpdf(0, 0, -1)"))
    assert np.isnan(m.num("norminv(1.5)"))
    assert np.isnan(m.num("binopdf(1, 2.5, 0.5)"))
    with pytest.raises(MatlabError, match="match in size"):
        m.run("normpdf([1 2 3], [1 2], 1)")
    with pytest.raises(MatlabError, match="Not enough"):
        m.run("gampdf(1)")


@pytest.mark.parametrize("name, params", [
    ("Normal", "1, 2"), ("Lognormal", "0.3, 0.5"), ("Exponential", "2"), ("Gamma", "2.5, 1.5"),
    ("Beta", "2, 3"), ("Weibull", "2, 1.5"), ("Uniform", "-1, 3"), ("Rayleigh", "2"),
    ("Extreme Value", "1, 2"), ("Generalized Extreme Value", "0.2, 1.5, 1"),
    ("Generalized Pareto", "0.3, 2, 1"), ("T", "4"), ("Chisquare", "5"), ("F", "3, 12"),
    ("Logistic", "1, 2"), ("Loglogistic", "0.5, 0.4"), ("Nakagami", "1.5, 2"), ("Rician", "1, 2"),
    ("InverseGaussian", "2, 3"), ("BirnbaumSaunders", "2, 0.5"), ("Burr", "2, 3, 1.5"),
    ("tLocationScale", "1, 2, 5"), ("Noncentral t", "5, 1"), ("Noncentral F", "3, 10, 2"),
    ("Noncentral Chi-square", "4, 1.5"),
])
def test_continuous_family_consistency(m, name, params):
    """icdf inverts cdf, and the pdf is the derivative of the cdf."""
    p = np.array([0.1, 0.5, 0.9])
    x = m.num(f"icdf('{name}', [0.1 0.5 0.9], {params})").ravel()
    close(m.num(f"cdf('{name}', [{x[0]:.15g} {x[1]:.15g} {x[2]:.15g}], {params})").ravel(), p, 1e-8)
    h = 1e-5 * max(abs(x[1]), 1)
    f = m.num(f"pdf('{name}', {x[1]:.15g}, {params})")
    dF = (m.num(f"cdf('{name}', {x[1] + h:.15g}, {params})") - m.num(f"cdf('{name}', {x[1] - h:.15g}, {params})")) / (2 * h)
    close(f, dF, 1e-5)
    close(m.num(f"cdf('{name}', {x[2]:.15g}, {params}, 'upper')"), 0.1, 1e-8)


def test_parameter_conventions_against_scipy(m):
    close(m.num("gamcdf(2, 3, 4)"), st.gamma.cdf(2, 3, scale=4))
    close(m.num("wblinv(0.3, 2, 5)"), st.weibull_min.ppf(0.3, 5, scale=2))
    close(m.num("exppdf(1, 4)"), st.expon.pdf(1, scale=4))
    close(m.num("lognpdf(2, 1, 0.5)"), st.lognorm.pdf(2, 0.5, scale=math.e))
    close(m.num("nbincdf(3, 2.5, 0.4)"), st.nbinom.cdf(3, 2.5, 0.4))
    close(m.num("geocdf(2, 0.3)"), 1 - 0.7 ** 3)
    close(m.num("hygecdf(1, 20, 5, 6)"), st.hypergeom.cdf(1, 20, 5, 6))
    close(m.num("evpdf(0.5, 1, 2)"), st.gumbel_l.pdf(0.5, 1, 2))
    close(m.num("gevpdf(0.5, 0.2, 1, 0)"), st.genextreme.pdf(0.5, -0.2))
    close(m.num("nctcdf(1, 5, 1)"), st.nct.cdf(1, 5, 1))


def test_stat_functions(m):
    m.run("[mn, vr] = wblstat(2, 3); [m2, v2] = binostat(10, 0.3); [m3, v3] = expstat(3);"
          "[m4, v4] = unifstat(1, 3); [m5, v5] = gamstat(2, 3); [m6, v6] = lognstat(0, 1);"
          "[m7, v7] = tstat(5); [m8, v8] = poisstat([1 2 3]); [m9, v9] = normstat(1, 2);")
    close(m.val("mn"), 2 * math.gamma(1 + 1 / 3))
    close(m.val("vr"), 4 * (math.gamma(1 + 2 / 3) - math.gamma(1 + 1 / 3) ** 2))
    close([m.val("m2"), m.val("v2")], [[[3.0]], [[2.1]]])
    close([m.val("m3"), m.val("v3"), m.val("m4"), m.val("v4")], [[[3]], [[9]], [[2]], [[1 / 3]]])
    close([m.val("m5"), m.val("v5")], [[[6]], [[18]]])
    close(m.val("m6"), math.exp(0.5))
    close([m.val("m7"), m.val("v7")], [[[0]], [[5 / 3]]])
    close(m.val("m8"), [[1, 2, 3]])
    close([m.val("m9"), m.val("v9")], [[[1]], [[4]]])


def test_random_numbers(m):
    m.run("rng(7); a = gamrnd(2, 3, 2000, 1); b = wblrnd(2, 3, [3 4]); c = unidrnd(6, 1, 500);"
          "d = hygernd(20, 5, 6, 200, 1); e = evrnd(0, 1, 3000, 1); f = nbinrnd(3, 0.5, 100, 1);"
          "g = normrnd([0 100], 1); h = random('Poisson', 4, 2, 3); k = geornd(0.3, 50, 1);"
          "rng(7); a2 = gamrnd(2, 3, 2000, 1);")
    a = m.val("a")
    assert a.shape == (2000, 1) and abs(a.mean() - 6) < 0.4 and np.all(a > 0)
    np.testing.assert_array_equal(a, m.val("a2"))
    assert m.val("b").shape == (3, 4)
    c = m.val("c")
    assert set(np.unique(c)) <= {1, 2, 3, 4, 5, 6} and c.shape == (1, 500)
    assert np.all((m.val("d") >= 0) & (m.val("d") <= 5))
    assert abs(m.val("e").mean() + 0.5772) < 0.1          # mean of the minimum-type extreme value law
    assert np.all(m.val("f") >= 0) and m.val("g").shape == (1, 2) and abs(m.val("g")[0, 1] - 100) < 6
    assert m.val("h").shape == (2, 3) and np.all(m.val("k") >= 0)
    with pytest.raises(MatlabError, match="unknown distribution"):
        m.run("random('nope', 1)")


def test_exact_fit_intervals(m):
    m.run(f"x = {X10}; [mu, sg, muci, sgci] = normfit(x); [mh, mci] = expfit(x, 0.1);"
          "[lh, lci] = poissfit([1 2 3 4 2 2]); [ph, pci] = binofit(7, 20); [a, b, aci, bci] = unifit(x);"
          "[rh, rci] = raylfit(x); [lg, lgci] = lognfit(x);")
    n, xb, s = 10, X10V.mean(), X10V.std(ddof=1)
    close([m.val("mu"), m.val("sg")], [[[xb]], [[s]]])
    t = st.t.ppf(0.975, 9)
    close(m.val("muci").ravel(), [xb - t * s / math.sqrt(n), xb + t * s / math.sqrt(n)])
    close(m.val("sgci").ravel(), [s * math.sqrt(9 / st.chi2.ppf(0.975, 9)), s * math.sqrt(9 / st.chi2.ppf(0.025, 9))])
    close(m.val("mci").ravel(), [2 * n * xb / st.chi2.ppf(0.95, 2 * n), 2 * n * xb / st.chi2.ppf(0.05, 2 * n)])
    close(m.val("lci").ravel(), [1.27566, 3.91494], 1e-5)                       # Octave reference
    close(m.val("pci").ravel(), [st.beta.ppf(0.025, 7, 14), st.beta.ppf(0.975, 8, 13)])
    close([m.val("a"), m.val("b")], [[[0.6]], [[4.0]]])
    close(m.val("rci").ravel(), [1.14465, 2.16056], 1e-5)                       # Octave reference
    close(m.val("lgci"), [[-0.0214191, 0.434302], [0.881939, 1.1527]], 1e-5)    # Octave reference


def test_maximum_likelihood_fits(m):
    m.run(f"x = {X10}; [g, gci] = gamfit(x); [w, wci] = wblfit(x); [e, eci] = evfit(x);"
          "[bt, bci] = betafit([0.2 0.5 0.7 0.3 0.4 0.6 0.55 0.35]);"
          "[gv, gvci] = gevfit([x 1.5 1.7 2.8 0.4 1.3]); p = mle(x, 'Distribution', 'gamma');"
          "pn = mle(x); [nl, av] = normlike([0 1], [-1 0 1]); nlg = gamlike(g, x);")
    a, _, scale = st.gamma.fit(X10V, floc=0)
    close(m.val("g"), [[a, scale]], 1e-6)
    close(m.val("p"), m.val("g"), 1e-9)
    close(m.val("gci"), [[1.31715, 0.244225], [6.96594, 1.4945]], 2e-5)         # Octave reference
    c, _, sc = st.weibull_min.fit(X10V, floc=0)
    close(m.val("w"), [[sc, c]], 1e-4)        # scipy's own optimiser stops early
    close(m.val("wci"), [[1.45185, 1.13928], [2.96042, 2.97963]], 2e-5)         # Octave reference
    close(m.val("eci"), [[1.64761, 0.720666], [3.13646, 1.76808]], 2e-5)        # Octave reference
    close(m.val("gvci"), [[-0.473515, 0.457247, 0.79849], [0.633561, 1.18133, 1.69519]], 5e-5)  # Octave reference
    ab = st.beta.fit([0.2, 0.5, 0.7, 0.3, 0.4, 0.6, 0.55, 0.35], floc=0, fscale=1)
    close(m.val("bt"), [[ab[0], ab[1]]], 1e-5)
    close(m.val("pn"), [[X10V.mean(), X10V.std(ddof=0)]])
    close(m.val("nl"), 1.5 * math.log(2 * math.pi) + 1)
    close(m.val("av"), np.eye(2) / 3, 1e-5)
    close(m.val("nlg"), -np.sum(st.gamma.logpdf(X10V, a, scale=scale)), 1e-8)


def test_censored_and_custom_mle(m):
    m.run("x = [1 2 3 4 5 6]; c = [0 0 0 0 1 1]; mu = expfit(x, 0.05, c);"
          "ph = mle(x, 'pdf', @(x, lam) lam * exp(-lam * x), 'Start', 1);"
          "[p2, ci2] = mle([2 3 2 5 4 3 2 1], 'Distribution', 'poisson');")
    close(m.val("mu"), 21 / 4)                 # total time on test divided by the number of failures
    close(m.val("ph"), 6 / 21, 1e-6)
    close(m.val("p2"), 2.75)
    assert m.val("ci2").shape == (2, 1)


def test_makedist_objects(m):
    m.run("pd = makedist('Normal', 'mu', 1, 'sigma', 2); v = [cdf(pd, 1) mean(pd) icdf(pd, 0.5) std(pd) var(pd)];"
          "w = makedist('Weibull', 'A', 2, 'B', 3); nm = w.ParameterNames; pv = w.ParameterValues;"
          "w.A = 5; t = truncate(makedist('Normal'), 0, Inf); tv = [mean(t) var(t) pdf(t, -1) cdf(t, 0)];"
          "b = makedist('Binomial', 'N', 10, 'p', 0.3); bv = [mean(b) var(b) pdf(b, 3) median(b)];"
          "tr = makedist('Triangular', 'A', 0, 'B', 1, 'C', 3); q = iqr(makedist('Uniform', 'Lower', 0, 'Upper', 4));"
          "rng(1); r = random(pd, 4, 2); up = cdf(pd, 1, 'upper'); cls = class(pd); tt = t.IsTruncated;")
    close(m.val("v"), [[0.5, 1, 1, 2, 4]])
    close(m.val("pv"), [[2, 3]])
    assert m.num("w.A") == 5 and m.val("r").shape == (4, 2)
    close(m.val("tv"), [[math.sqrt(2 / math.pi), 1 - 2 / math.pi, 0, 0]], 1e-8)
    close(m.val("bv"), [[3, 2.1, st.binom.pmf(3, 10, 0.3), 3]])
    close(m.num("mean(tr)"), 4 / 3)
    close(m.val("q"), 2)
    close(m.val("up"), 0.5)
    assert m.str("cls") == "prob.NormalDistribution" and bool(m.val("tt"))
    assert "Normal distribution" in m.run("pd") and "sigma = 2" in m.run("pd")
    with pytest.raises(MatlabError, match="not a parameter"):
        m.run("makedist('Normal', 'rate', 2)")


def test_fitdist(m):
    m.run(f"x = {X10}'; pd = fitdist(x, 'Normal'); ci = paramci(pd); nl = negloglik(pd);"
          "g = fitdist(x, 'Gamma'); gp = g.ParameterValues; k = fitdist(x, 'Kernel'); kb = k.Bandwidth;"
          "kc = cdf(k, [-100 100]); km = mean(k);"
          "[pdca, gn] = fitdist([1 2 3 4 10 11 12 13]', 'Normal', 'By', [1 1 1 1 2 2 2 2]');"
          "m2 = pdca{2}.mu; e = fitdist(x, 'Exponential'); em = e.mu; ld = fitdist(x, 'Logistic'); lm = ld.mu;")
    s = X10V.std(ddof=1)
    close([m.num("pd.mu").item(), m.num("pd.sigma").item()], [X10V.mean(), s])
    t = st.t.ppf(0.975, 9)
    close(m.val("ci")[:, 0], [X10V.mean() - t * s / math.sqrt(10), X10V.mean() + t * s / math.sqrt(10)])
    close(m.val("nl"), -np.sum(st.norm.logpdf(X10V, X10V.mean(), s)))
    a, _, scale = st.gamma.fit(X10V, floc=0)
    close(m.val("gp"), [[a, scale]], 1e-6)
    sig = np.median(np.abs(X10V - np.median(X10V))) / 0.6745
    close(m.val("kb"), sig * (4 / 30) ** 0.2)
    close(m.val("kc"), [[0, 1]], 1e-3)
    close(m.val("km"), X10V.mean())
    close(m.val("m2"), 11.5)
    close(m.val("em"), X10V.mean())
    assert abs(m.num("lm") - np.median(X10V)) < 0.5


def test_multivariate(m):
    m.run("p = mvncdf([0.3 -0.2], [0 0], eye(2)); box = mvncdf([-1 -1], [1 1], [0 0], eye(2));"
          "t0 = mvtpdf([0 0], eye(2), 3); rng(3); R = mvtrnd([1 0.8; 0.8 1], 5, 4000);"
          "W = wishrnd([2 0.5; 0.5 1], 10); U = copularnd('Gaussian', 0.7, 2000); pc = mvncdf([0 0], [0 0], [1 .5; .5 1]);"
          "V = copularnd('t', [1 .5; .5 1], 4, 100); C = copularnd('Clayton', 2, 100);")
    close(m.val("p"), st.norm.cdf(0.3) * st.norm.cdf(-0.2), 1e-6)
    close(m.val("box"), (st.norm.cdf(1) - st.norm.cdf(-1)) ** 2, 1e-6)
    close(m.val("t0"), 1 / (2 * math.pi), 1e-10)
    close(m.val("pc"), 0.25 + math.asin(0.5) / (2 * math.pi), 1e-5)
    R = m.val("R")
    assert R.shape == (4000, 2) and abs(np.corrcoef(R.T)[0, 1] - 0.8) < 0.05
    W = m.val("W")
    assert np.allclose(W, W.T) and np.all(np.linalg.eigvalsh(W) > 0)
    for name in ("U", "V", "C"):
        u = m.val(name)
        assert u.shape[1] == 2 and np.all((u > 0) & (u < 1))
    assert abs(st.spearmanr(m.val("U"))[0] - 0.683) < 0.05


def test_ecdf(m):
    m.run("[f, x] = ecdf([3 1 2]); [f2, x2, lo, up] = ecdf([3 1 2 2 5 4]', 'Censoring', [0 0 0 1 0 0]');"
          "[s, xs] = ecdf([1 2 3 4], 'Function', 'survivor'); [ff, xf] = ecdf([1 2], 'Frequency', [3 1]);")
    close(m.val("f").ravel(), [0, 1 / 3, 2 / 3, 1])
    close(m.val("x").ravel(), [1, 1, 2, 3])
    close(m.val("f2").ravel(), [0, 1 / 6, 1 / 3, 5 / 9, 7 / 9, 1])
    lo, up = m.val("lo").ravel(), m.val("up").ravel()
    close(lo[1:5], [0, 0, 0.120008, 0.400583], 1e-5)                # Octave reference
    close(up[1:5], [0.464866, 0.710529, 0.991103, 1], 1e-5)
    assert np.isnan(lo[0]) and np.isnan(up[-1])
    close(m.val("s").ravel(), [1, 0.75, 0.5, 0.25, 0])
    close(m.val("ff").ravel(), [0, 0.75, 1])


def test_ksdensity(m):
    m.run("x = [1 2 3 4 5]; [f, xi, bw] = ksdensity(x); f0 = ksdensity([0 1], 0.5, 'Bandwidth', 1);"
          "c = ksdensity([1 2 3], 2, 'Function', 'cdf'); q = ksdensity([1 2 3], 0.5, 'Function', 'icdf');"
          "[fp, xp] = ksdensity([1 2 3], 'Support', 'positive'); fb = ksdensity([1 2 3], [-1 2], 'Support', [0 5]);"
          "fe = ksdensity(x, 3, 'Kernel', 'epanechnikov'); fw = ksdensity([0 10], 0, 'Weights', [3 1], 'Bandwidth', 1);")
    f, xi = m.val("f").ravel(), m.val("xi").ravel()
    bw = m.num("bw").item()
    close(bw, (1 / 0.6745) * (4 / 15) ** 0.2)
    assert f.size == 100 and abs(xi[0] - (1 - 3 * bw)) < 1e-12
    close(np.sum((f[1:] + f[:-1]) / 2 * np.diff(xi)), 1.0, 5e-3)
    close(m.val("f0"), st.norm.pdf(0.5))
    close(m.val("c"), 0.5)
    close(m.val("q"), 2.0, 1e-8)
    assert np.all(m.val("xp") > 0)
    close(m.val("fb")[0, 0], 0)
    assert m.num("fe") > 0
    close(m.val("fw"), 0.75 * st.norm.pdf(0) + 0.25 * st.norm.pdf(10))
    with pytest.raises(MatlabError, match="unknown option"):
        m.run("ksdensity([1 2 3], 'Bandwidth', 1, 'Nope', 1)")


def test_distribution_plots(m):
    m.run("rng(1); x = randn(60, 1); figure; h = histfit(x); n1 = numel(h); figure; h2 = normplot(x);"
          "figure; h3 = qqplot(x); figure; qqplot(x, 2 * x + 1); figure; probplot('weibull', exp(x));"
          "figure; [h5, s] = cdfplot(x); figure; ecdf(x); figure; plot(makedist('Normal')); close all;")
    assert m.num("n1") == 2
    s = m.val("s")
    x = m.val("x")
    close([s.get_field(0, "mean"), s.get_field(0, "median")], [[[x.mean()]], [[np.median(x)]]])


def test_statset_and_help(m):
    m.run("o = statset('MaxIter', 50); o2 = statset(o, 'TolX', 1e-3);")
    assert m.num("o2.MaxIter") == 50 and m.num("o2.TolX") == 1e-3 and m.val("o").get_field(0, "TolFun").size == 0
    out = m.run("help wblfit")
    assert "wblfit(x" in out and "Weibull" in out
