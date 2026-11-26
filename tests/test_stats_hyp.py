"""Hypothesis tests. Expected values: scipy.stats where the test is the same, closed forms, and GNU
Octave's statistics package run as a black box (noted where used)."""

import math

import numpy as np
import pytest
import scipy.stats as st

from mylab.errors import MatlabError

A = np.array([1.1, 2.3, -0.5, 4, 3.2, -1, 2, 5, 1.7, 0.3])
AS = "[1.1 2.3 -0.5 4 3.2 -1 2 5 1.7 0.3]"
G3 = "[1 2 3 4; 5 6 7 8; 2 4 6 9]'"


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def field(m, name, f):
    return np.asarray(m.val(name).get_field(0, f), dtype=float)


def test_t_and_z_tests(m):
    m.run(f"x = {AS}; [h, p, ci, s] = ttest(x, 1); [h2, p2] = ttest(x, 1, 'Tail', 'right', 'Alpha', 0.2);"
          "[h3, p3] = ttest(x, 1, 0.05, 'left'); y = x + [0.5 -0.2 0.1 0.9 0.3 -0.4 0.2 0.6 0.1 0.4];"
          "[hp, pp] = ttest(x, y); [hz, pz, ciz, z] = ztest(x, 1, 2); [h4, p4, ci4, s4] = ttest2(x, y(1:6));"
          "[h5, p5, ci5, s5] = ttest2(x, y(1:6), 'Vartype', 'unequal'); [hm, pm] = ttest([x' 2 * x']);")
    r = st.ttest_1samp(A, 1)
    close([m.num("p").item(), field(m, "s", "tstat").item()], [r.pvalue, r.statistic])
    t = st.t.ppf(0.975, 9)
    half = t * A.std(ddof=1) / math.sqrt(10)
    close(m.val("ci"), [[A.mean() - half, A.mean() + half]])
    close(m.val("p2"), st.ttest_1samp(A, 1, alternative="greater").pvalue)
    close(m.val("p3"), st.ttest_1samp(A, 1, alternative="less").pvalue)
    assert m.num("h") == 0 and m.num("h2") == 1
    y = A + np.array([0.5, -0.2, 0.1, 0.9, 0.3, -0.4, 0.2, 0.6, 0.1, 0.4])
    close(m.val("pp"), st.ttest_rel(A, y).pvalue)
    zv = (A.mean() - 1) / (2 / math.sqrt(10))
    close([m.num("z").item(), m.num("pz").item()], [zv, 2 * st.norm.sf(abs(zv))])
    close(m.val("ciz"), [[A.mean() - 1.959963984540054 * 2 / math.sqrt(10), A.mean() + 1.959963984540054 * 2 / math.sqrt(10)]])
    close(m.val("p4"), st.ttest_ind(A, y[:6]).pvalue)
    w = st.ttest_ind(A, y[:6], equal_var=False)
    close([m.num("p5").item(), field(m, "s5", "tstat").item()], [w.pvalue, w.statistic])
    assert m.val("ci5").shape == (1, 2) and m.val("pm").shape == (1, 2)
    with pytest.raises(MatlabError, match="Tail"):
        m.run("ttest(x, 0, 'Tail', 'sideways')")
    with pytest.raises(MatlabError, match="unknown option"):
        m.run("ttest(x, 0, 'Alpha', 0.05, 'Nope', 1)")


def test_variance_tests(m):
    m.run("x = [1 2 3 4 5 9]; y = [2 3 4 5 5.5 4]; [h, p, ci, s] = vartest(x, 2); [h2, p2, ci2, s2] = vartest2(x, y);"
          "[p3, s3] = vartestn([1 2 3 4; 2 4 6 8; 1 5 9 13]', 'Display', 'off');"
          "p4 = vartestn([1 2 3 4; 2 4 6 8; 1 5 9 13]', 'Display', 'off', 'TestType', 'LeveneAbsolute');"
          "p5 = vartestn([1 2 3 4 2 4 6 8]', [1 1 1 1 2 2 2 2]', 'Display', 'off', 'TestType', 'BrownForsythe');"
          "[nd, pr, chi] = barttest([1 2 3; 2 1 4; 3 5 2; 5 4 7; 6 8 6; 7 6 9; 9 9 8; 8 7 12]);")
    x = np.array([1, 2, 3, 4, 5, 9.0])
    chi = 5 * x.var(ddof=1) / 2
    close(field(m, "s", "chisqstat"), chi)
    close(m.val("p"), 2 * min(st.chi2.cdf(chi, 5), st.chi2.sf(chi, 5)))
    close(m.val("ci").ravel(), [3.11709, 48.1225], 1e-5)                      # Octave reference
    close(m.val("ci2").ravel(), [0.681897, 34.825], 1e-5)                     # Octave reference
    g = [np.array(v, dtype=float) for v in ([1, 2, 3, 4], [2, 4, 6, 8], [1, 5, 9, 13])]
    close(m.val("p3"), st.bartlett(*g).pvalue)
    close(m.val("p4"), st.levene(*g, center="mean").pvalue)
    close(m.val("p5"), st.levene(g[0], g[1], center="median").pvalue)
    close(m.val("chi").ravel(), [25.2124, 11.0743], 1e-5)                     # Octave reference
    close(m.val("pr").ravel(), [0.00012677, 0.00393768], 1e-5)


def test_rank_tests(m):
    m.run(f"x = {AS}; [p, h, s] = signrank(x); p8 = signrank(1:8); [pt, ht, stt] = signtest(x);"
          "long = [x 2.2 3.1 4.1 0.2 0.9 1.3 2.8]; [pa, ha, sa] = signrank(long); pe = signrank(long, 0, 'Method', 'exact');"
          "pr = ranksum([1 2 3 4 5], [6 7 8 9 10]); [p2, h2, s2] = ranksum(1:12, 5:16); pl = ranksum([1 2 3 4 5], [6 7 8 9 10], 'Tail', 'left');"
          "pxy = signrank([1 2 3 4 5 6], [1.5 1.4 3.9 4.1 6.2 7.5]); pse = signtest(x, 1, 'Method', 'approximate');")
    close(m.val("p"), st.wilcoxon(A, mode="exact").pvalue)
    close(m.val("p8"), 2 / 256)
    close(field(m, "s", "signedrank"), np.sum(st.rankdata(np.abs(A))[A > 0]))
    close(m.val("pt"), 0.109375)
    long = np.concatenate([A, [2.2, 3.1, 4.1, 0.2, 0.9, 1.3, 2.8]])
    close(m.val("pa"), st.wilcoxon(long, mode="approx", correction=False).pvalue, 1e-8)
    close(m.val("pe"), st.wilcoxon(long, mode="exact").pvalue)
    close(m.val("pr"), 2 / 252)
    close(m.val("p2"), 0.0223416, 1e-5)                                        # Octave reference
    close(field(m, "s2", "ranksum"), 110)
    close(m.val("pl"), 1 / 252)
    d = np.array([1, 2, 3, 4, 5, 6]) - np.array([1.5, 1.4, 3.9, 4.1, 6.2, 7.5])
    close(m.val("pxy"), st.wilcoxon(d, mode="exact").pvalue)
    assert 0 < m.num("pse") <= 1


def test_distribution_tests(m):
    m.run("x = (-2:0.5:2)'; [h, p, k, cv] = kstest(x); [h2, p2, k2] = kstest2([1 2 3 4 5 6 7], [3 4 5 6 7 8 9 10 11]);"
          "y = [0.3 -1.2 0.5 1.7 -0.4 0.9 2.1 -0.7 0.1 1.3 -1.9 0.6 0.2 -0.3 1.1];"
          "[hc, pc] = kstest(y, 'CDF', makedist('Normal', 'mu', 0.2, 'sigma', 1.1)); [hr, prr] = kstest(y, 'Tail', 'larger');"
          "[ha, pa, ad] = adtest(y); [hl, pl, kl, cl] = lillietest(y); [hj, pj, jb] = jbtest(y);"
          "rng(5); e = exprnd(1, 80, 1); he = [lillietest(e) jbtest(e) adtest(e) kstest(e)];"
          "[hx, px] = adtest(e, 'Distribution', 'exp'); [hw, pw] = adtest(e, 'Distribution', 'weibull');"
          "hle = lillietest(e, 'Distribution', 'exponential'); [hd, pd0] = adtest(y, 'Distribution', makedist('Normal'));")
    r = st.kstest(np.arange(-2, 2.5, 0.5), "norm")
    close([m.num("p").item(), m.num("k").item()], [r.pvalue, r.statistic], 1e-8)
    close(m.val("cv"), 0.4300, 1e-4)                                           # Octave reference
    close([m.num("p2").item(), m.num("k2").item()], [0.3155, 0.4444], 2e-4)    # Octave reference
    y = np.array([0.3, -1.2, 0.5, 1.7, -0.4, 0.9, 2.1, -0.7, 0.1, 1.3, -1.9, 0.6, 0.2, -0.3, 1.1])
    close(m.val("pc"), st.kstest(y, st.norm(0.2, 1.1).cdf).pvalue, 1e-8)
    close(m.val("prr"), st.kstest(y, "norm", alternative="greater").pvalue, 1e-8)
    close(m.val("ad"), st.anderson(y).statistic, 1e-8)
    close(m.val("pa"), 0.99)                                                   # tabulated upper limit (Octave agrees)
    n = y.size
    z = np.sort((y - y.mean()) / y.std(ddof=1))
    F = st.norm.cdf(z)
    d = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    close(m.val("kl"), d)
    assert m.num("hl") == 0 and m.num("pl") == 0.5 and m.num("hj") == 0 and 0 < m.num("cl") < 1
    s, kt = st.skew(y), st.kurtosis(y, fisher=False)
    close(m.val("jb"), n * (s ** 2 / 6 + (kt - 3) ** 2 / 24))
    close(m.val("he"), [[1, 1, 1, 1]])
    assert m.num("hx") == 0 and m.num("hw") == 0 and m.num("hle") == 0 and m.num("px") > 0.05
    assert m.num("hd") == 0 and 0.05 < m.num("pd0") <= 1


def test_chi2gof(m):
    m.run("x = [1 2 2 3 3 3 4 4 5 6 7 8 2 3 4 5 6 3 4 5 4 4 5 5 6 6 7 3 2 1 4 5 6 5 4 5 4 5 6 5];"
          "[h, p, s] = chi2gof(x); [h0, p0, s0] = chi2gof([1 2 2 3 3 3 4 4 5]);"
          "[h2, p2, s2] = chi2gof(0:3, 'Frequency', [10 20 15 5], 'Expected', [12 18 14 6], 'Ctrs', 0:3);"
          "[h3, p3, s3] = chi2gof(x, 'CDF', makedist('Normal', 'mu', 4.4, 'sigma', 1.6), 'NBins', 6);"
          "[h4, p4, s4] = chi2gof(x, 'CDF', {@normcdf, 4.4, 1.6}, 'Edges', 1:8, 'EMin', 0);")
    close(field(m, "s", "chi2stat"), 15.4918, 1e-5)                            # Octave reference
    close(m.val("p"), 1.4411e-03, 1e-4)
    close(field(m, "s", "edges").ravel(), [1, 3.1, 3.8, 4.5, 5.2, 5.9, 8])
    close(field(m, "s", "O").ravel(), [12, 0, 9, 10, 0, 9])
    close(field(m, "s", "E").ravel(), [9.2658, 5.9324, 6.7476, 6.4089, 5.0830, 6.5623], 1e-4)
    assert field(m, "s", "df") == 3
    assert np.isnan(m.num("p0")) and m.num("h0") == 0 and field(m, "s0", "df") == 0   # too few data (as in Octave)
    chi = sum((o - e) ** 2 / e for o, e in zip([10, 20, 15, 5], [12, 18, 14, 6]))
    close([field(m, "s2", "chi2stat").item(), m.num("p2").item()], [chi, st.chi2.sf(chi, 3)])
    assert field(m, "s3", "df") == field(m, "s3", "O").size - 1
    assert field(m, "s4", "df") == 6 + 2 - 2 or field(m, "s4", "df") == 4


def test_one_way_layouts(m):
    m.run(f"X = {G3}; [p, tbl, stats] = anova1(X, [], 'off'); pk = kruskalwallis(X, [], 'off');"
          "[pf, tf, sf] = friedman([1 2 3; 2 3 1; 3 1 2; 1 3 2; 1 2 3], 1, 'off');"
          "y = X(:); g = repelem({'a'; 'b'; 'c'}, 4); p2 = anova1(y, g, 'off'); F = tbl{2, 5}; pk2 = kruskalwallis(y, g, 'off');"
          "[c, mm, hh, nms] = multcompare(stats, 'Display', 'off'); cb = multcompare(stats, 'Display', 'off', 'CType', 'bonferroni');"
          "cs = multcompare(stats, 'Display', 'off', 'CType', 'scheffe'); cl = multcompare(stats, 'Display', 'off', 'CType', 'lsd');"
          "[pp, tt, sk] = kruskalwallis(X, [], 'off'); ck = multcompare(sk, 'Display', 'off');")
    g = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = st.f_oneway(*g)
    close([m.num("p").item(), m.num("F").item(), m.num("p2").item()], [r.pvalue, r.statistic, r.pvalue])
    close(m.val("pk"), st.kruskal(*g).pvalue)
    close(m.val("pk2"), m.val("pk"))
    close(m.val("pf"), st.friedmanchisquare([1, 2, 3, 1, 1], [2, 3, 1, 3, 2], [3, 1, 2, 2, 3]).pvalue)
    hsd = st.tukey_hsd(*g)
    c = m.val("c")
    assert c.shape == (3, 6)
    close(c[:, 5], [hsd.pvalue[0, 1], hsd.pvalue[0, 2], hsd.pvalue[1, 2]], 1e-6)
    ci = hsd.confidence_interval()
    close(c[:, 2], [ci.low[0, 1], ci.low[0, 2], ci.low[1, 2]], 1e-6)
    close(c[:, 3], [-4, -2.75, 1.25])
    close(m.val("mm")[:, 0], [2.5, 6.5, 5.25])
    assert m.str("nms{2}") == "2"
    se = math.sqrt(sum(np.var(v, ddof=1) for v in g) / 3 * 0.5)
    tcrit = st.t.ppf(1 - 0.05 / 6, 9)
    close(m.val("cb")[0, 2], -4 - tcrit * se)
    close(m.val("cl")[0, 5], 2 * st.t.sf(4 / se, 9))
    close(m.val("cs")[0, 4], -4 + math.sqrt(2 * st.f.ppf(0.95, 2, 9)) * se)
    close(m.val("ck")[:, 3], [-5.875, -3.875, 2.0])
    close(m.val("ck")[0, 5], 0.0535, 2e-3)                                      # Octave reference
    out = m.run(f"p3 = anova1({G3});")
    assert "Source" in out and "Groups" in out


def test_anova2_and_anovan(m):
    m.run("[p, tbl, stats] = anova2([1 2 3; 2 3 4; 5 6 9; 6 7 8], 2, 'off'); p1 = anova2([1 2 3; 2 3 5; 4 6 9], 1, 'off');"
          "y = [1 2 3 4 5 6 7 8 9 10 12 11]'; g1 = [1 1 1 1 2 2 2 2 3 3 3 3]'; g2 = repmat({'a'; 'b'}, 6, 1);"
          "[pn, tn, sn] = anovan(y, {g1, g2}, 'display', 'off'); [pi, ti] = anovan(y, {g1, g2}, 'model', 'interaction', "
          "'display', 'off', 'varnames', {'A', 'B'}); src = ti{4, 1}; p1w = anovan(y, {g1}, 'display', 'off');"
          "c = multcompare(sn, 'Display', 'off'); pc = anovan(y, {g1, (1:12)'}, 'continuous', 2, 'display', 'off');"
          "c2 = multcompare(stats, 'Display', 'off', 'Estimate', 'row');")
    close(m.val("p").ravel(), [0.00702157, 4.1184e-05, 0.547708], 1e-5)        # Octave reference
    assert m.val("p1").shape == (1, 2)
    close(m.val("pn").ravel(), [8.66123e-05, 0.402759], 1e-5)                  # Octave reference
    close(m.val("pi").ravel(), [0.000783741, 0.462605, 0.860681], 1e-5)        # Octave reference
    assert m.str("src") == "A*B"
    close(m.val("p1w"), st.f_oneway([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 11]).pvalue)
    close(m.val("c")[:, 3], [-4, -8, -4])
    assert m.val("pc").shape == (2, 1) and m.val("c2").shape == (1, 6)
    with pytest.raises(MatlabError, match="not supported"):
        m.run("anovan(y, {g1, g2}, 'random', 1)")


def test_fisher_runs_power(m):
    m.run("[h, p, s] = fishertest([3 1; 1 3]); [hr1, pr] = fishertest([8 2; 1 5], 'Tail', 'right');"
          "[hr, prn, sr] = runstest([1 2 3 4 5 1 2 3 4 5 6 7 1 2 0 9]); [hu, pu] = runstest([1 3 2 5 4 7 6 9], 'ud');"
          "n = sampsizepwr('t', [100 5], 102, 0.80); pw = sampsizepwr('z', [100 5], 102, [], 50);"
          "n2 = sampsizepwr('t2', [100 5], 103, 0.9); p1 = sampsizepwr('z', [100 5], [], 0.8, 50);"
          "npr = sampsizepwr('p', 0.3, 0.4, 0.8); nv = sampsizepwr('var', 4, 8, 0.8);")
    close(m.val("p"), st.fisher_exact([[3, 1], [1, 3]])[1])
    close(field(m, "s", "OddsRatio"), 9)
    close(m.val("pr"), st.fisher_exact([[8, 2], [1, 5]], alternative="greater")[1])
    close(m.val("prn"), 0.216783, 1e-5)                                         # Octave reference
    close([field(m, "sr", "nruns").item(), field(m, "sr", "n1").item(), field(m, "sr", "n0").item()], [6, 7, 9])
    assert 0 < m.num("pu") <= 1
    assert m.num("n") == 52
    d = 2 / (5 / math.sqrt(50))
    close(m.val("pw"), st.norm.sf(1.959963984540054 - d) + st.norm.cdf(-1.959963984540054 - d))
    assert 55 <= m.num("n2") <= 65 and 150 <= m.num("npr") <= 200 and 30 <= m.num("nv") <= 50
    close(m.val("p1"), 100 + (1.959963984540054 + st.norm.ppf(0.8)) * 5 / math.sqrt(50), 1e-4)
