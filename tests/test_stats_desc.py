"""Descriptive statistics, correlation, grouped summaries, resampling and partitions."""

import math

import numpy as np
import pytest
import scipy.stats as st

from mylab.errors import MatlabError

XM = np.array([[1, 2, 3], [2, 4, 1], [3, 1, 5], [4, 3, 2], [5, 7, 9], [6, 2, 4], [2, 8, 3]], dtype=float)
XS = "[1 2 3; 2 4 1; 3 1 5; 4 3 2; 5 7 9; 6 2 4; 2 8 3]"


def close(a, b, tol=1e-10):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def test_grpstats_matrix(m):
    m.run("x = [1 10; 2 20; 3 30; 4 50; 5 70]; g = {'a'; 'a'; 'b'; 'b'; 'b'};"
          "mn = grpstats(x, g); [a, s, n, nm, md, r] = grpstats(x, g, {'mean', 'std', 'numel', 'gname', 'median', @range});"
          "ci = grpstats([1; 2; 3; 4; 5; 6], [1; 1; 1; 2; 2; 2], 'meanci'); one = grpstats(x(:, 1), []);"
          "two = grpstats((1:4)', {[1; 1; 2; 2], {'u'; 'v'; 'u'; 'v'}}); sem = grpstats(x(:, 1), g, 'sem');")
    close(m.val("mn"), [[1.5, 15], [4, 50]])
    close(m.val("s"), [[math.sqrt(0.5), math.sqrt(50)], [1, 20]])
    close(m.val("n"), [[2, 2], [3, 3]])
    close(m.val("md"), [[1.5, 15], [4, 50]])
    close(m.val("r"), [[1, 10], [2, 40]])
    assert [m.str("nm{1}"), m.str("nm{2}")] == ["a", "b"]
    t = st.t.ppf(0.975, 2)
    close(m.val("ci"), [[2 - t / math.sqrt(3), 2 + t / math.sqrt(3)], [5 - t / math.sqrt(3), 5 + t / math.sqrt(3)]])
    close(m.val("one"), 3)
    close(m.val("two").ravel(), [1, 2, 3, 4])
    close(m.val("sem").ravel(), [math.sqrt(0.5) / math.sqrt(2), 1 / math.sqrt(3)])
    with pytest.raises(MatlabError, match="unknown statistic"):
        m.run("grpstats(x, g, 'nope')")


def test_grpstats_table(m):
    m.run("T = table([1; 1; 2; 2], [1; 2; 3; 4], [10; 20; 30; 50], 'VariableNames', {'g', 'x', 'y'});"
          "S = grpstats(T, 'g', {'mean', 'max'}); names = S.Properties.VariableNames; cnt = S.GroupCount;"
          "my = S.mean_y; mx = S.max_x; S2 = grpstats(T, 'g', 'mean', 'DataVars', 'y'); w = width(S2);")
    assert [m.str(f"names{{{k}}}") for k in range(1, 7)] == ["g", "GroupCount", "mean_x", "max_x", "mean_y", "max_y"]
    close(m.val("cnt").ravel(), [2, 2])
    close(m.val("my").ravel(), [15, 40])
    close(m.val("mx").ravel(), [2, 4])
    assert m.num("w") == 3


def test_crosstab_tabulate_dummyvar(m):
    m.run("a = [1 1 2 2 1 2 1 1 2 2]; b = [1 1 1 2 1 2 2 1 2 2]; [tbl, chi2, p, lab] = crosstab(a, b);"
          "t3 = crosstab([1 2 1 2], [1 1 2 2], [1 1 1 2]); t = tabulate([1 1 3]); tc = tabulate({'x', 'y', 'x'});"
          "tg = tabulate(categorical({'u', 'v', 'u', 'u'})); d = dummyvar([1 3 2 1]'); tn = tabulate([0.5 0.5 2]);")
    obs = np.array([[4, 1], [1, 4]])
    close(m.val("tbl"), obs)
    chi, pv, _, _ = st.chi2_contingency(obs, correction=False)
    close([m.num("chi2").item(), m.num("p").item()], [chi, pv])
    assert m.val("t3").shape == (2, 2, 2) and m.str("lab{2, 1}") == "2"
    close(m.val("t"), [[1, 2, 200 / 3], [2, 0, 0], [3, 1, 100 / 3]])
    assert m.str("tc{1, 1}") == "x" and m.num("tc{1, 2}") == 2 and m.num("tg{1, 3}") == 75
    close(m.val("d"), [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    close(m.val("tn"), [[0.5, 2, 200 / 3], [2, 1, 100 / 3]])
    assert "Count" in m.run("tabulate([1 2 2])")


def test_simple_summaries(m):
    m.run("r = tiedrank([10 20 20 30]); [r2, adj] = tiedrank([1 2 2 3 3 3]); tm = trimmean([1 2 3 4 100], 40);"
          "tw = trimmean(1:10, 25, 'weighted'); gm = geomean([1 2 4]); hm = harmmean([1 2 4]);"
          "m3 = moment([1 2 3 4 10], 3); m2 = moment([1 2; 3 6], 2); mall = moment([1 2; 3 6], 2, 'all');"
          "md0 = mad([1 2 3 4 10]); md1 = mad([1 2 3 4 10], 1); gn = geomean([1 NaN 4], 'omitnan');"
          "gz = geomean([2 8; 4 2], 2); ma = mad(magic(3), 0, 'all');")
    close(m.val("r"), [[1, 2.5, 2.5, 4]])
    close(m.val("adj"), 15)                          # Octave reference
    close(m.val("tm"), 3)
    v = np.arange(1, 11.0)
    w = np.ones(10)
    w[0] = w[-1] = 0
    w[1] = w[-2] = 0.75
    close(m.val("tw"), np.sum(w * v) / np.sum(w))
    close([m.num("gm").item(), m.num("hm").item()], [2, 3 / 1.75])
    x = np.array([1, 2, 3, 4, 10.0])
    close(m.val("m3"), np.mean((x - 4) ** 3))
    close(m.val("m2"), [[1, 4]])
    close(m.val("mall"), np.var([1, 2, 3, 6]))
    close([m.num("md0").item(), m.num("md1").item()], [np.mean(np.abs(x - 4)), 1])
    close(m.val("gn"), 2)
    close(m.val("gz").ravel(), [4, math.sqrt(8)])
    close(m.val("ma"), np.mean(np.abs(np.arange(1, 10) - 5)))


def test_corr_types(m):
    m.run(f"X = {XS}; [r, p] = corr(X); [rs, ps] = corr(X, 'Type', 'Spearman'); [rk, pk] = corr(X, 'Type', 'Kendall');"
          "[rxy, pxy] = corr(X(:, 1), X(:, 2:3)); [rt, pt] = corr(X(:, 1), X(:, 3), 'Tail', 'right');"
          "Y = X; Y(2, 1) = NaN; rc = corr(Y, 'Rows', 'complete'); rp = corr(Y, 'Rows', 'pairwise'); ra = corr(Y);")
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            pr = st.pearsonr(XM[:, i], XM[:, j])
            sp = st.spearmanr(XM[:, i], XM[:, j])
            kd = st.kendalltau(XM[:, i], XM[:, j])
            close([m.val("r")[i, j], m.val("p")[i, j]], [pr[0], pr[1]], 1e-9)
            close(m.val("rs")[i, j], sp[0], 1e-9)
            close([m.val("rk")[i, j], m.val("pk")[i, j]], [kd[0], kd[1]], 1e-9)
    close(np.diag(m.val("r")), [1, 1, 1])
    assert m.val("rxy").shape == (1, 2)
    close(m.val("pt"), st.pearsonr(XM[:, 0], XM[:, 2], alternative="greater")[1], 1e-9)
    keep = [0, 2, 3, 4, 5, 6]
    close(m.val("rc"), np.corrcoef(XM[keep].T), 1e-10)
    close(m.val("rp")[1, 2], np.corrcoef(XM[:, 1], XM[:, 2])[0, 1])
    assert np.isnan(m.val("ra")[0, 1]) and not np.isnan(m.val("ra")[1, 2])
    with pytest.raises(MatlabError, match="Type"):
        m.run("corr(X, 'Type', 'nope')")


def test_spearman_exact_p_and_partialcorr(m):
    m.run("[r, p] = corr([1 2 3 4 5]', [2 1 4 3 5]', 'Type', 'Spearman'); x = [1 2 3 4 5 6 7 8]';"
          f"X = {XS}; [pr, pp] = partialcorr(X); [p2, q2] = partialcorr(X(:, 1), X(:, 2), X(:, 3));"
          "[R, s] = corrcov([4 2; 2 9]); c1 = nancov([1 2; 3 NaN; 5 7; 2 1]); c2 = nancov([1 2; 3 NaN; 5 7; 2 1], 'pairwise');")
    close(m.val("r"), 0.8)
    # exact permutation distribution: 2 * P(rho >= 0.8) over the 120 orderings
    from itertools import permutations
    base = np.arange(1, 6)
    d = np.array([np.sum((np.array(q) - base) ** 2) for q in permutations(base)])
    close(m.val("p"), 2 * np.mean(d <= 4))
    R = np.corrcoef(XM.T)
    P = np.linalg.inv(R)
    close(m.val("pr")[0, 1], -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1]))
    close(m.val("p2"), m.val("pr")[0, 1])
    tval = m.val("pr")[0, 1] * math.sqrt(4 / (1 - m.val("pr")[0, 1] ** 2))
    close(m.val("q2"), 2 * st.t.sf(abs(tval), 4))
    close(m.val("R"), [[1, 1 / 3], [1 / 3, 1]])
    close(m.val("s").ravel(), [2, 3])
    close(m.val("c1"), np.cov(np.array([[1, 2], [5, 7], [2, 1]]).T))
    close(m.val("c2")[0, 0], np.var([1, 3, 5, 2], ddof=1))


def test_quantile_forms_and_zscore(m):
    m.run("q = prctile(1:10, [25 50 75]); qa = quantile(magic(4), [0.25 0.75], 'all'); q2 = quantile(magic(4), 0.5, 2);"
          "qv = prctile(magic(4), 50, [1 2]); qm = quantile([1 2 3 4], 0.5, 'Method', 'exact');"
          "[z, mu, sg] = zscore([1 2 3; 4 6 8]); za = zscore(magic(3), 0, 'all'); z1 = zscore([2 4 6], 1);")
    close(m.val("q"), [[3, 5.5, 8]])
    v = np.sort(np.arange(1, 17.0))
    close(m.val("qa").ravel(), np.percentile(v, [25, 75], method="hazen"))
    assert m.val("qa").shape == (2, 1)
    close(m.val("q2").ravel(), [8, 9, 8, 9])
    close(m.val("qv"), 8.5)
    close(m.val("qm"), 2.5)
    close(m.val("z"), [[-math.sqrt(0.5)] * 3, [math.sqrt(0.5)] * 3])
    close(m.val("mu"), [[2.5, 4, 5.5]])
    close(m.val("za"), (np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]) - 5) / np.std(np.arange(1, 10), ddof=1))
    close(m.val("z1"), np.array([[-1, 0, 1]]) * math.sqrt(1.5))


def test_bootstrap_and_jackknife(m):
    m.run("rng(1); x = [1 2 3 4 5]; s = bootstrp(50, @mean, x); [bs, sam] = bootstrp(20, @(a, b) corr(a, b), (1:10)', (1:10)'.^2);"
          "j = jackknife(@mean, [1 2 3 4]'); rng(2); d = randn(60, 1) + 3; ci = bootci(300, @mean, d);"
          "cp = bootci(300, {@mean, d}, 'Type', 'per', 'Alpha', 0.1); cn = bootci(200, @mean, d, 'Type', 'norm');"
          "cs = bootci(100, @mean, d, 'Type', 'stud', 'NBootStd', 20); c2 = bootci(100, @(v) [mean(v) median(v)], d);")
    s = m.val("s")
    assert s.shape == (50, 1) and np.all((s >= 1) & (s <= 5))
    assert m.val("sam").shape == (10, 20) and m.val("bs").shape == (20, 1)
    close(m.val("j").ravel(), [3, 8 / 3, 7 / 3, 2])
    d = m.val("d")
    for name in ("ci", "cp", "cn", "cs"):
        ci = m.val(name).ravel()
        assert ci.shape == (2,) and ci[0] < d.mean() < ci[1] and ci[1] - ci[0] < 1.2
    assert m.val("c2").shape == (2, 2)
    width = m.val("ci")[1, 0] - m.val("ci")[0, 0]
    assert abs(width - 2 * 1.96 * d.std(ddof=1) / math.sqrt(60)) < 0.15


def test_cvpartition(m):
    m.run("rng(1); c = cvpartition(10, 'KFold', 5); nt = c.NumTestSets; ts = c.TestSize; tr = c.TrainSize;"
          "all_test = zeros(10, 1); for i = 1:5, all_test = all_test + test(c, i); end; t1 = test(c, 1); r1 = training(c, 1);"
          "g = [1 1 1 1 2 2 2 2 2 2]'; c2 = cvpartition(g, 'KFold', 2); s2 = test(c2, 1); h = cvpartition(10, 'HoldOut', 0.3);"
          "hs = [h.TrainSize h.TestSize]; lo = cvpartition(5, 'LeaveOut'); nl = lo.NumTestSets; c3 = cvpartition(11, 'KFold', 3);"
          "ts3 = sort(c3.TestSize); rs = cvpartition(6, 'Resubstitution'); rr = [sum(training(rs)) sum(test(rs))];"
          "hg = cvpartition(g, 'HoldOut', 0.5); tg = test(hg); ty = c.Type; c4 = repartition(c); n4 = c4.NumObservations;")
    assert m.num("nt") == 5
    close(m.val("ts"), [[2] * 5])
    close(m.val("tr"), [[8] * 5])
    close(m.val("all_test").ravel(), np.ones(10))
    assert np.array_equal(m.val("t1"), ~m.val("r1")) and m.val("t1").dtype == bool
    s2 = m.val("s2").ravel()
    assert s2[:4].sum() == 2 and s2[4:].sum() == 3
    close(m.val("hs"), [[7, 3]])
    assert m.num("nl") == 5
    close(m.val("ts3"), [[3, 4, 4]])
    close(m.val("rr"), [[6, 6]])
    tg = m.val("tg").ravel()
    assert tg[:4].sum() == 2 and tg[4:].sum() == 3
    assert m.str("ty") == "kfold" and m.num("n4") == 10
    assert "K-fold" in m.run("c")
    with pytest.raises(MatlabError, match="folds"):
        m.run("cvpartition(3, 'KFold', 5)")


def test_crossval_function(m):
    m.run("rng(2); X = randn(50, 2); y = X * [1; 2] + 0.1 * randn(50, 1);"
          "vals = crossval(@(xtr, ytr, xte, yte) mean((yte - xte * (xtr \\ ytr)).^2), X, y);"
          "mse = crossval('mse', X, y, 'Predfun', @(xtr, ytr, xte) xte * (xtr \\ ytr), 'KFold', 5);"
          "g = [ones(25, 1); 2 * ones(25, 1)]; mcr = crossval('mcr', X, g, 'Predfun', @(xtr, ytr, xte) ones(size(xte, 1), 1));"
          "v2 = crossval(@(xtr, xte) size(xte, 1), X, 'Holdout', 0.2); v3 = crossval(@(xtr, xte) size(xte, 1), X, 'Leaveout', 1);")
    assert m.val("vals").shape == (10, 1) and np.all(m.val("vals") < 0.1)
    assert 0.005 < m.num("mse") < 0.03
    close(m.val("mcr"), 0.5)
    close(m.val("v2"), 10)
    assert m.val("v3").shape == (50, 1)


def test_sampling(m):
    m.run("rng(3); a = randsample(10, 4); b = randsample([5 6 7 8], 2); c = randsample(3, 20, true, [0 0 1]);"
          "[d, i] = datasample([10 20 30 40], 3, 'Replace', false); figure; h = gscatter([1 2 3 4], [1 2 3 4], [1 1 2 2]);"
          "nh = numel(h); close all;")
    a = m.val("a")
    assert a.shape == (4, 1) and len(set(a.ravel())) == 4
    assert m.val("b").shape == (1, 2) and np.all(m.val("c") == 3)
    close(m.val("d"), np.array([[10, 20, 30, 40]])[:, m.val("i").astype(int).ravel() - 1])
    assert m.num("nh") == 2
