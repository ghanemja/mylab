# ruff: noqa: E501  (long MATLAB code strings)
"""Regression: linear models, generalized linear models, nonlinear, penalised and robust fits.

Expected values come from direct least-squares computations, from the estimating equations the
estimators must satisfy, and (where noted) from GNU Octave's statistics package as a black box.
"""

import math

import numpy as np
import pytest
import scipy.stats as st

from mylab.errors import MatlabError

XD = np.array([[0.5, 1.2], [-0.3, 0.8], [1.5, -0.2], [0.7, 0.3], [-1.1, 1.9], [2.2, 0.1], [0.1, -1.3], [-0.6, 0.4],
               [1.3, 1.1], [0.9, -0.7]])
YD = np.array([2.1, 0.4, 3.2, 1.9, -0.8, 4.6, 1.0, 0.1, 3.3, 2.4])
DATA = ("X = [0.5 1.2; -0.3 0.8; 1.5 -0.2; 0.7 0.3; -1.1 1.9; 2.2 0.1; 0.1 -1.3; -0.6 0.4; 1.3 1.1; 0.9 -0.7];"
        "y = [2.1; 0.4; 3.2; 1.9; -0.8; 4.6; 1.0; 0.1; 3.3; 2.4];")


def close(a, b, tol=1e-9):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=tol, atol=tol)


def ols(X, y):
    D = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    r = y - D @ b
    dfe = len(y) - D.shape[1]
    mse = r @ r / dfe
    cov = mse * np.linalg.inv(D.T @ D)
    return D, b, r, dfe, mse, cov


def test_fitlm_matches_least_squares(m):
    m.run(DATA + "mdl = fitlm(X, y); c = mdl.Coefficients; est = c.Estimate; se = c.SE; t = c.tStat; p = c.pValue;"
          "r2 = [mdl.Rsquared.Ordinary mdl.Rsquared.Adjusted]; v = [mdl.RMSE mdl.MSE mdl.SSE mdl.SSR mdl.SST mdl.DFE];"
          "n = [mdl.NumObservations mdl.NumCoefficients mdl.NumPredictors]; res = mdl.Residuals.Raw; fit = mdl.Fitted;"
          "ci = coefCI(mdl); [yp, yci] = predict(mdl, [1 1; 0 2]); [yo, oci] = predict(mdl, [1 1], 'Prediction', 'observation');"
          "f = feval(mdl, 1, 1); pv = coefTest(mdl); a = anova(mdl, 'summary'); ll = mdl.LogLikelihood; aic = mdl.ModelCriterion.AIC;"
          "nm = mdl.CoefficientNames; cov = mdl.CoefficientCovariance; lev = mdl.Diagnostics.Leverage; st = mdl.Residuals.Studentized;")
    D, b, r, dfe, mse, cov = ols(XD, YD)
    close(m.val("est").ravel(), b)
    se = np.sqrt(np.diag(cov))
    close(m.val("se").ravel(), se)
    close(m.val("t").ravel(), b / se)
    close(m.val("p").ravel(), 2 * st.t.sf(np.abs(b / se), dfe))
    sst = np.sum((YD - YD.mean()) ** 2)
    close(m.val("r2"), [[1 - r @ r / sst, 1 - (r @ r / dfe) / (sst / 9)]])
    close(m.val("v"), [[math.sqrt(mse), mse, r @ r, sst - r @ r, sst, dfe]])
    close(m.val("n"), [[10, 3, 2]])
    close(m.val("res").ravel(), r)
    close(m.val("fit").ravel(), D @ b)
    tc = st.t.ppf(0.975, dfe)
    close(m.val("ci"), np.column_stack([b - tc * se, b + tc * se]))
    x0 = np.array([1, 1, 1.0])
    half = tc * math.sqrt(x0 @ cov @ x0)
    close(m.val("yp").ravel(), [x0 @ b, np.array([1, 0, 2.0]) @ b])
    close(m.val("yci")[0], [x0 @ b - half, x0 @ b + half])
    close(m.val("oci"), [[x0 @ b - tc * math.sqrt(x0 @ cov @ x0 + mse), x0 @ b + tc * math.sqrt(x0 @ cov @ x0 + mse)]])
    close(m.val("f"), x0 @ b)
    F = ((sst - r @ r) / 2) / mse
    close(m.val("pv"), st.f.sf(F, 2, dfe))
    close(m.num("a.F(2)"), F)
    ll = -5 * (math.log(2 * math.pi) + math.log(r @ r / 10) + 1)
    close([m.num("ll").item(), m.num("aic").item()], [ll, -2 * ll + 6])
    assert [m.str(f"nm{{{k}}}") for k in (1, 2, 3)] == ["(Intercept)", "x1", "x2"]
    close(m.val("cov"), cov)
    H = D @ np.linalg.inv(D.T @ D) @ D.T
    close(m.val("lev").ravel(), np.diag(H))
    s2i = (r @ r - r ** 2 / (1 - np.diag(H))) / (dfe - 1)
    close(m.val("st").ravel(), r / np.sqrt(s2i * (1 - np.diag(H))))
    out = m.run("mdl")
    assert "Linear regression model:" in out and "y ~ 1 + x1 + x2" in out and "R-squared" in out


def test_fitlm_formulas_and_categorical(m):
    m.run("rng(1); x1 = randn(30, 1); x2 = randn(30, 1); x3 = randn(30, 1); y = 1 + 2 * x1 - x2 + 0.5 * x2 .* x3 + 0.1 * randn(30, 1);"
          "T = table(x1, x2, x3, y); m2 = fitlm(T, 'y ~ x1 + x2*x3'); n2 = m2.CoefficientNames; e2 = m2.Coefficients.Estimate; f2 = m2.Formula;"
          "m3 = fitlm([x1 x2 x3], y, 'quadratic'); n3 = m3.NumCoefficients; m0 = fitlm(T, 'y ~ x1 + x2 - 1'); n0 = m0.CoefficientNames;"
          "mq = fitlm(T, 'y ~ x1^2'); nq = mq.CoefficientNames; mi = fitlm([x1 x2], y, 'interactions'); ni = mi.CoefficientNames;"
          "g = categorical(repmat({'a'; 'b'; 'c'}, 10, 1)); T2 = table(x1, g, y); m4 = fitlm(T2, 'y ~ x1 + g'); n4 = m4.CoefficientNames;"
          "a4 = anova(m4); yp = predict(m4, table(0.5, categorical({'b'}), 'VariableNames', {'x1', 'g'}));"
          "m5 = fitlm([x1 x2], y, 'linear', 'Weights', (1:30)', 'Exclude', [1 2]); n5 = m5.NumObservations;"
          "m6 = fitlm([x1 x2 x3], y, 'y ~ a + c', 'VarNames', {'a', 'b', 'c', 'y'});"
          "m7 = fitlm([x1 x2], y, 'CategoricalVars', 2); m8 = addTerms(mi, 'x1^2'); m9 = removeTerms(m8, 'x1:x2');"
          "figure; plot(fitlm(x1, y)); figure; plotResiduals(m2, 'fitted'); close all;")
    names = [m.str(f"n2{{{k}}}") for k in range(1, 6)]
    assert names == ["(Intercept)", "x1", "x2", "x3", "x2:x3"] and m.str("f2") == "y ~ 1 + x1 + x2 + x3 + x2:x3"
    x1, x2, x3, y = (m.val(v).ravel() for v in ("x1", "x2", "x3", "y"))
    D = np.column_stack([np.ones(30), x1, x2, x3, x2 * x3])
    close(m.val("e2").ravel(), np.linalg.lstsq(D, y, rcond=None)[0], 1e-8)
    assert m.num("n3") == 10 and [m.str(f"n0{{{k}}}") for k in (1, 2)] == ["x1", "x2"]
    assert [m.str(f"nq{{{k}}}") for k in (1, 2, 3)] == ["(Intercept)", "x1", "x1^2"]
    assert m.str("ni{4}") == "x1:x2" and [m.str(f"n4{{{k}}}") for k in (3, 4)] == ["g_b", "g_c"]
    codes = np.arange(30) % 3
    Dg = np.column_stack([np.ones(30), x1, codes == 1, codes == 2]).astype(float)
    bg = np.linalg.lstsq(Dg, y, rcond=None)[0]
    close(m.val("yp"), bg[0] + 0.5 * bg[1] + bg[2], 1e-8)
    assert m.num("a4.DF(2)") == 2 and m.num("n5") == 28
    assert m.str("m6.CoefficientNames{3}") == "c" and m.num("m7.NumCoefficients") > 3
    assert m.num("m8.NumCoefficients") == 5 and m.num("m9.NumCoefficients") == 4
    with pytest.raises(MatlabError, match="not a variable"):
        m.run("fitlm(T, 'y ~ zz')")


def test_stepwise_and_robust_models(m):
    m.run("rng(1); x1 = randn(60, 1); x2 = randn(60, 1); x3 = randn(60, 1); y = 1 + 2 * x1 - x2 + 0.1 * randn(60, 1);"
          "ms = stepwiselm([x1 x2 x3], y, 'Upper', 'linear', 'Verbose', 0); f = ms.Formula; [b, se, pv, inm] = stepwisefit([x1 x2 x3], y, 'Display', 'off');"
          "yo = y; yo(1:3) = 50; mr = fitlm([x1 x2], yo, 'RobustOpts', 'on'); er = mr.Coefficients.Estimate;"
          "[br, sr] = robustfit([x1 x2], yo); bo = robustfit([x1 x2], yo, 'ols'); bh = robustfit([x1 x2], yo, 'huber', 1.345);"
          "D = x2fx([x1 x2], 'quadratic'); D1 = x2fx([x1 x2]);")
    assert m.str("f") == "y ~ 1 + x1 + x2"
    np.testing.assert_array_equal(m.val("inm"), [[True, True, False]])
    close(m.val("b").ravel()[:2], [2, -1], 0.05)
    close(m.val("er").ravel(), [1, 2, -1], 0.1)
    close(m.val("br").ravel(), m.val("er").ravel(), 1e-6)
    assert abs(m.val("bo")[0, 0] - 1) > 1            # ordinary least squares is pulled away by the outliers
    close(m.val("bh").ravel()[1:], [2, -1], 0.3)
    w = np.asarray(m.val("sr").get_field(0, "w")).ravel()
    assert np.all(w[:3] < 1e-6) and np.all(w[3:] > 0.2)
    assert m.val("D").shape == (60, 6) and m.val("D1").shape == (60, 3)


def test_regress_outputs(m):
    m.run(DATA + "[b, bint, r, rint, stats] = regress(y, [ones(10, 1) X]);")
    _, b, r, dfe, mse, cov = ols(XD, YD)
    close(m.val("b").ravel(), b)
    close(m.val("stats"), [[0.992453, 460.245, 3.73474e-08, 0.0266914]], 1e-5)       # Octave reference
    close(m.val("bint"), [[0.758737, 1.07612], [1.52376, 1.79439], [-0.0344001, 0.255801]], 1e-5)
    close(m.val("rint")[:2], [[-0.0871371, 0.527523], [-0.472223, 0.255688]], 1e-5)   # Octave reference


def test_glm_estimating_equations(m):
    m.run("x = (1:8)'; yc = [1 3 2 6 8 12 15 30]'; [bp, dp, sp] = glmfit(x, yc, 'poisson'); [bg, dg] = glmfit(x, yc, 'gamma');"
          "bl = glmfit(x, yc, 'normal', 'link', 'log'); [bb, db, sb] = glmfit((1:6)', [0 0 1 0 1 1]', 'binomial');"
          "bpr = glmfit(x, [yc 40 * ones(8, 1)], 'binomial', 'link', 'probit'); bi = glmfit(x, yc, 'inverse gaussian');"
          "bo = glmfit(x, yc, 'poisson', 'offset', log(x), 'weights', (1:8)'); bn = glmfit(x, yc, 'poisson', 'constant', 'off');"
          "[yh, lo, hi] = glmval(bb, [2; 5], 'logit', sb); yv = glmval(bp, 3, 'log');")
    x = np.arange(1, 9.0)
    yc = np.array([1, 3, 2, 6, 8, 12, 15, 30.0])
    D = np.column_stack([np.ones(8), x])
    mu = np.exp(D @ m.val("bp").ravel())
    close(D.T @ (yc - mu), [0, 0], 1e-7)                       # Poisson score equations
    close(m.val("dp"), 2 * np.sum(yc * np.log(yc / mu) - (yc - mu)), 1e-9)
    se = np.sqrt(np.diag(np.linalg.inv(D.T @ (D * mu[:, None]))))
    close(np.asarray(m.val("sp").get_field(0, "se")).ravel(), se, 1e-7)
    mug = 1 / (D @ m.val("bg").ravel())
    close(D.T @ (yc - mug), [0, 0], 1e-6)                       # canonical link: X'(y - mu) = 0
    mul = np.exp(D @ m.val("bl").ravel())
    close(D.T @ ((yc - mul) * mul), [0, 0], 1e-5)               # least squares with a log link
    xb = np.arange(1, 7.0)
    yb = np.array([0, 0, 1, 0, 1, 1.0])
    Db = np.column_stack([np.ones(6), xb])
    pb = 1 / (1 + np.exp(-Db @ m.val("bb").ravel()))
    close(Db.T @ (yb - pb), [0, 0], 1e-7)
    close(m.val("db"), -2 * np.sum(yb * np.log(pb) + (1 - yb) * np.log(1 - pb)), 1e-9)
    ppr = st.norm.cdf(D @ m.val("bpr").ravel())
    wts = st.norm.pdf(D @ m.val("bpr").ravel()) / (ppr * (1 - ppr))
    close(D.T @ ((yc / 40 - ppr) * wts), [0, 0], 1e-7)
    mui = (D @ m.val("bi").ravel()) ** -0.5
    close(D.T @ (yc - mui), [0, 0], 1e-5)
    muo = x * np.exp(D @ m.val("bo").ravel())
    close(D.T @ (x * (yc - muo)), [0, 0], 1e-6)
    assert m.val("bn").shape == (1, 1)
    close(m.val("yh").ravel(), 1 / (1 + np.exp(-np.array([[1, 2], [1, 5]]) @ m.val("bb").ravel())))
    assert np.all(m.val("lo") > 0) and np.all(m.val("hi") > 0)
    close(m.val("yv"), math.exp(m.val("bp")[0, 0] + 3 * m.val("bp")[1, 0]))


def test_fitglm_object(m):
    m.run("x = (1:8)'; yc = [1 3 2 6 8 12 15 30]'; mp = fitglm(x, yc, 'Distribution', 'poisson'); bp = glmfit(x, yc, 'poisson');"
          "e = mp.Coefficients.Estimate; dv = [mp.Deviance mp.DFE mp.Dispersion]; yp = predict(mp, [2; 9]); [y2, ci2] = predict(mp, 2);"
          "T = table(x, logical([0 0 1 0 1 1 0 1]'), 'VariableNames', {'dose', 'resp'}); mb = fitglm(T, 'resp ~ dose', 'Distribution', 'binomial');"
          "fb = mb.Formula; pb = predict(mb, table(4.5, 'VariableNames', {'dose'})); dt = devianceTest(mp); mn = fitglm(x, yc);"
          "lm = fitlm(x, yc); en = [mn.Coefficients.Estimate lm.Coefficients.Estimate]; ll = mp.LogLikelihood;"
          "mc = fitglm(x, [yc 40 * ones(8, 1)], 'Distribution', 'binomial', 'Link', 'comploglog'); ec = mc.Coefficients.Estimate;")
    close(m.val("e"), m.val("bp"))
    close(m.val("dv")[0, 1:], [6, 1])
    b = m.val("bp").ravel()
    close(m.val("yp").ravel(), np.exp(b[0] + b[1] * np.array([2, 9])))
    ci = m.val("ci2").ravel()
    assert ci[0] < m.num("y2") < ci[1]
    assert m.str("fb") == "logit(resp) ~ 1 + dose" and 0 < m.num("pb") < 1
    assert m.num("dt.pValue(2)") < 1e-6
    close(m.val("en")[:, 0], m.val("en")[:, 1])
    x = np.arange(1, 9.0)
    yc = np.array([1, 3, 2, 6, 8, 12, 15, 30.0])
    close(m.val("ll"), np.sum(st.poisson.logpmf(yc, np.exp(b[0] + b[1] * x))))
    assert m.val("ec").shape == (2, 1)
    out = m.run("mp")
    assert "Generalized linear regression model:" in out and "Distribution = Poisson" in out
    with pytest.raises(MatlabError, match="distribution"):
        m.run("fitglm(x, yc, 'Distribution', 'cauchy')")


def test_nonlinear_regression(m):
    m.run("beta = nlinfit((0:4)', 2 * exp(-0.5 * (0:4)'), @(b, x) b(1) * exp(b(2) * x), [1; -1]);"
          "x = (0:0.5:3)'; y = [3.1 2.2 1.4 1.1 0.7 0.5 0.4]'; f = @(b, x) b(1) * exp(b(2) * x);"
          "[b, R, J, C, mse] = nlinfit(x, y, f, [1 -1]); ci = nlparci(b, R, 'covar', C); ci2 = nlparci(b, R, 'jacobian', J);"
          "[yp, d] = nlpredci(f, [1; 2], b, R, 'Covar', C); nm = fitnlm(x, y, f, [1 -1]); en = nm.Coefficients.Estimate;"
          "T = table(x, y); n2 = fitnlm(T, 'y ~ a*exp(k*x)', [1 -1]); e2 = n2.Coefficients.Estimate; cn = n2.CoefficientNames;"
          "[p2, c2] = predict(n2, [1; 2]); r2 = nm.Rsquared.Ordinary; bw = nlinfit(x, y, f, [1 -1], 'Weights', [1 1 1 1 1 1 100]');")
    close(m.val("beta").ravel(), [2, -0.5], 1e-7)
    b = m.val("b").ravel()
    x = np.arange(0, 3.5, 0.5)
    y = np.array([3.1, 2.2, 1.4, 1.1, 0.7, 0.5, 0.4])
    r = y - b[0] * np.exp(b[1] * x)
    Jn = np.column_stack([np.exp(b[1] * x), b[0] * x * np.exp(b[1] * x)])
    close(Jn.T @ r, [0, 0], 1e-6)                               # normal equations at the optimum
    close(m.val("R").ravel(), r, 1e-7)
    close(m.val("J"), Jn, 1e-5)
    mse = r @ r / 5
    close(m.val("mse"), mse, 1e-7)
    cov = mse * np.linalg.inv(Jn.T @ Jn)
    close(m.val("C"), cov, 1e-5)
    t = st.t.ppf(0.975, 5)
    close(m.val("ci"), np.column_stack([b - t * np.sqrt(np.diag(cov)), b + t * np.sqrt(np.diag(cov))]), 1e-5)
    close(m.val("ci2"), m.val("ci"), 1e-5)
    g = np.array([math.exp(b[1]), b[0] * math.exp(b[1])])
    close(m.val("yp")[0, 0], b[0] * math.exp(b[1]), 1e-8)
    close(m.val("d")[0, 0], t * math.sqrt(g @ cov @ g), 1e-5)
    close(m.val("en").ravel(), b, 1e-7)
    close(m.val("e2").ravel(), b, 1e-6)
    assert [m.str("cn{1}"), m.str("cn{2}")] == ["a", "k"] and m.val("c2").shape == (2, 2) and m.num("r2") > 0.99
    assert abs(m.val("bw")[0, 0] * math.exp(3 * m.val("bw")[0, 1]) - 0.4) < 0.01
    assert "Nonlinear regression model:" in m.run("nm")


def test_lasso_ridge_and_pls(m):
    m.run(DATA + "[B, info] = lasso(X, y, 'Lambda', 0.1); B2 = lasso(X, y, 'Lambda', 0.5, 'Alpha', 0.5); B0 = lasso(X, y, 'Lambda', 1e-9);"
          "[Bp, ip] = lasso(X, y, 'NumLambda', 20); rng(1); [Bc, ic] = lasso(X, y, 'CV', 5, 'NumLambda', 10);"
          "r = ridge(y, X, [0 1 10]); r0 = ridge(y, X, 1, 0);"
          "[XL, YL, XS, YS, BETA, PCTVAR, MSE] = plsregress(X, y, 2); [~, ~, ~, ~, B1] = plsregress(X, y, 1);"
          "[p, S] = polyfit(1:10, [1.1 2.3 2.9 4.2 5.1 5.8 7.2 8.1 8.8 10.3], 1); [yy, dl] = polyconf(p, [11 12], S);")
    # the lasso solution satisfies the subgradient (KKT) conditions on the standardised problem
    Xc = XD - XD.mean(axis=0)
    sx = np.sqrt(np.mean(Xc ** 2, axis=0))
    b = m.val("B").ravel() * sx
    g = (Xc / sx).T @ (YD - YD.mean() - (Xc / sx) @ b) / 10
    for j in range(2):
        if b[j] != 0:
            close(g[j], 0.1 * np.sign(b[j]), 1e-6)
        else:
            assert abs(g[j]) <= 0.1 + 1e-9
    b0 = np.asarray(m.val("info").get_field(0, "Intercept")).item()
    close(b0, YD.mean() - XD.mean(axis=0) @ m.val("B").ravel(), 1e-10)
    assert m.val("B2").shape == (2, 1)
    close(m.val("B0").ravel(), np.linalg.lstsq(np.column_stack([np.ones(10), XD]), YD, rcond=None)[0][1:], 1e-6)
    lam = np.asarray(m.val("ip").get_field(0, "Lambda")).ravel()
    df = np.asarray(m.val("ip").get_field(0, "DF")).ravel()
    assert m.val("Bp").shape == (2, 20) and np.all(np.diff(lam) > 0) and df[-1] == 0 and df[0] == 2
    assert "LambdaMinMSE" in m.val("ic").fields and np.asarray(m.val("ic").get_field(0, "SE")).shape == (1, 10)
    close(m.val("r"), [[1.68614, 1.49516, 0.76404], [0.104916, 0.0341104, -0.103575]], 1e-5)     # Octave reference
    close(m.val("r0").ravel(), [1.04204, 1.47116, 0.035991], 1e-5)                                  # Octave reference
    close(m.val("B1").ravel(), [1.23025, 1.40111, -0.385615], 1e-5)                                 # Octave reference
    close(m.val("PCTVAR"), [[0.647872, 0.352128], [0.918491, 0.0739617]], 1e-5)                     # Octave reference
    close(m.val("MSE")[:, :2], [[1.738, 0.611998], [2.4756, 0.201784]], 1e-5)                       # Octave reference
    close(m.val("BETA").ravel(), ols(XD, YD)[1], 1e-8)        # all components: ordinary least squares
    x = np.arange(1, 11.0)
    yv = np.array([1.1, 2.3, 2.9, 4.2, 5.1, 5.8, 7.2, 8.1, 8.8, 10.3])
    _, bb, rr, dfe, mse, cov = ols(x.reshape(-1, 1), yv)
    v = np.array([1, 11.0])
    close(m.val("dl")[0, 0], st.t.ppf(0.975, dfe) * math.sqrt(mse + v @ cov @ v), 1e-8)


def test_multinomial_and_lassoglm(m):
    m.run(DATA + "Y3 = [1 2 3 1 2 3 3 2 1 3]'; Bo = mnrfit(X, Y3, 'model', 'ordinal'); po = mnrval(Bo, X(1:2, :), 'model', 'ordinal');"
          "rng(2); Z = randn(120, 2); P = exp([0.5 + Z * [1; -1], -0.2 + Z * [0.5; 0.5], zeros(120, 1)]); P = P ./ sum(P, 2);"
          "u = rand(120, 1); Y = 1 + (u > P(:, 1)) + (u > P(:, 1) + P(:, 2)); [B, dev, stt] = mnrfit(Z, Y); ph = mnrval(B, Z);"
          "pc = mnrval(B, Z(1:3, :), 'type', 'cumulative'); Bc = mnrfit(Z, categorical(Y)); Bh = mnrfit(Z, Y, 'model', 'hierarchical');"
          "[Bl, il] = lassoglm(Z, Y > 1, 'binomial', 'Lambda', 0.01); bg = glmfit(Z, Y > 1, 'binomial'); Bz = lassoglm(Z, Y > 1, 'binomial', 'Lambda', 1e-8);")
    close(m.val("Bo").ravel(), [-2.53504, -0.495194, 0.307044, 2.23265], 1e-5)       # Octave reference
    close(m.val("po").sum(axis=1), [1, 1])
    Z, Y, B, ph = m.val("Z"), m.val("Y").ravel().astype(int), m.val("B"), m.val("ph")
    assert B.shape == (3, 2)
    close(ph.sum(axis=1), np.ones(120))
    ind = np.zeros((120, 3))
    ind[np.arange(120), Y - 1] = 1
    D = np.column_stack([np.ones(120), Z])
    close(D.T @ (ind - ph)[:, :2], np.zeros((3, 2)), 1e-5)       # score equations of the multinomial logit model
    close(m.val("dev"), -2 * np.sum(np.log(ph[np.arange(120), Y - 1])), 1e-8)
    assert m.val("pc").shape == (3, 2) and np.all(np.diff(m.val("pc"), axis=1) > 0)
    close(m.val("Bc"), B, 1e-6)
    assert m.val("Bh").shape == (3, 2)
    bg = m.val("bg").ravel()
    assert np.all(np.abs(m.val("Bl").ravel()) < np.abs(bg[1:]))  # shrunk towards zero
    close(m.val("Bz").ravel(), bg[1:], 1e-4)
