"""Control design and model transformations: discretization, reduction, Riccati/Lyapunov, LQ, PID tuning."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), rtol=tol, atol=tol)


def col(m, expr):
    return np.asarray(m.num(expr)).ravel()


def eigs(m, expr):
    return np.sort_complex(np.asarray(m.num(expr), dtype=complex).ravel())


# -- discretization -------------------------------------------------------------------------------

def test_c2d_closed_forms_for_first_order(m):
    ts, e = 0.1, np.exp(-0.1)
    cases = {
        "": ([0, 1 - e], [1, -e]),
        ", 'zoh'": ([0, 1 - e], [1, -e]),
        ", 'tustin'": ([ts / (2 + ts), ts / (2 + ts)], [1, -(2 - ts) / (2 + ts)]),
        ", 'impulse'": ([ts, 0], [1, -e]),
        ", 'matched'": ([0, 1 - e], [1, -e]),
        ", 'foh'": ([(ts - 1 + e) / ts, (1 - e - ts * e) / ts], [1, -e]),
    }
    for arg, (num, den) in cases.items():
        m.run(f"Gd = c2d(tf(1, [1 1]), 0.1{arg}); [n, d] = tfdata(Gd, 'v');")
        assert close(m.num("n"), [num]) and close(m.num("d"), [den]), arg
        assert m.num("Gd.Ts") == 0.1
    with pytest.raises(MatlabError, match="unknown method"):
        m.run("c2d(tf(1, [1 1]), 0.1, 'magic')")
    with pytest.raises(MatlabError, match="already discrete"):
        m.run("c2d(tf(1, [1 -0.5], 0.1), 0.1)")


def test_c2d_zoh_matches_step_response_at_samples(m):
    m.run("G = tf([1 3], [1 2 5 0]); Gd = c2d(G, 0.2); t = 0:0.2:4;")
    assert close(col(m, "step(Gd, t)"), col(m, "step(G, t)"), 1e-10)
    m.run("S = ss([0 1; -5 -2], [0; 1], [3 1], 0.5); Sd = c2d(S, 0.2);")
    assert m.str("class(Sd)") == "ss"
    assert close(col(m, "step(Sd, t)"), col(m, "step(S, t)"), 1e-10)
    m.run("Zd = c2d(zpk(-3, [0 -1], 2), 0.2);")
    assert m.str("class(Zd)") == "zpk"
    assert close(col(m, "step(Zd, t)"), col(m, "step(zpk(-3, [0 -1], 2), t)"), 1e-9)


def test_c2d_foh_matches_ramp_response_and_tustin_prewarp(m):
    m.run("G = tf(1, [1 2 5]); t = (0:0.1:3)'; Gd = c2d(G, 0.1, 'foh');")
    assert close(col(m, "lsim(Gd, t)"), col(m, "lsim(G, t, t, 'foh')"), 1e-10)
    m.run("w0 = 3; Gp = c2d(G, 0.1, 'prewarp', w0); Gt = c2d(G, 0.1, 'tustin');")
    exact = m.num("evalfr(G, 3i)")
    assert close(m.num("evalfr(Gp, exp(0.3i))"), exact, 1e-10)             # exact at the prewarp frequency
    assert abs(m.num("evalfr(Gt, exp(0.3i))") - exact) > 1e-4
    m.run("Go = c2d(G, 0.1, c2dOptions('Method', 'tustin', 'PrewarpFrequency', 3));")
    assert close(m.num("evalfr(Go, exp(0.3i))"), exact, 1e-10)
    # the bilinear map sends s = (2/Ts)(z - 1)/(z + 1)
    z = 0.7 + 0.4j
    assert close(m.num("evalfr(Gt, 0.7+0.4i)"), 1 / ((20 * (z - 1) / (z + 1)) ** 2 + 2 * 20 * (z - 1) / (z + 1) + 5))


def test_fractional_delays_are_exact_with_zoh(m):
    m.run("G = tf(1, [1 1], 'InputDelay', 0.25); Gd = c2d(G, 0.1); t = 0:0.1:2;")
    assert close(col(m, "step(Gd, t)"), col(m, "step(G, t)"), 1e-12)
    assert m.num("Gd.Ts") == 0.1
    m.run("S = ss([0 1; -2 -3], [0 1; 1 0], eye(2), [0 0; 1 0]);")
    m.run("S.InputDelay = [0.03; 0.17]; S.OutputDelay = [0.12; 0.05]; Sd = c2d(S, 0.1);")
    assert np.max(np.abs(m.num("step(Sd, t)") - m.num("step(S, t)"))) < 1e-12
    m.run("Gw = c2d(tf(1, [1 1], 'InputDelay', 0.3), 0.1, 'tustin');")     # whole samples for any method
    assert m.num("totaldelay(Gw)") == 3
    assert "rounded" in m.run("Gr = c2d(tf(1, [1 1], 'InputDelay', 0.26), 0.1, 'tustin');")
    assert m.num("totaldelay(Gr)") == 3


def test_d2c_and_d2d(m):
    m.run("G = tf([1 3], [1 2 5]);")
    for method in ("zoh", "tustin", "matched"):
        m.run(f"Gc = d2c(c2d(G, 0.1, '{method}'), '{method}');")
        assert close(m.num("evalfr(Gc, 1+2i)"), m.num("evalfr(G, 1+2i)"), 1e-8), method
        assert m.num("Gc.Ts") == 0
    m.run("S = ss([0 1; -5 -2], [0; 1], [3 1], 0); Sc = d2c(c2d(S, 0.1));")
    assert close(m.num("Sc.A"), m.num("S.A"), 1e-9)
    m.run("G1 = c2d(tf(1, [1 1]), 0.1); [n, d] = tfdata(d2d(G1, 0.2), 'v'); [n2, d2] = tfdata(d2d(G1, 0.05), 'v');")
    assert close(m.num("d"), [[1, -np.exp(-0.2)]]) and close(m.num("n"), [[0, 1 - np.exp(-0.2)]])
    assert close(m.num("d2"), [[1, -np.exp(-0.05)]]) and close(m.num("n2"), [[0, 1 - np.exp(-0.05)]])
    with pytest.raises(MatlabError, match="negative real axis"):
        m.run("d2c(tf(1, [1 0.5], 0.1))")


def test_pade(m):
    m.run("[num, den] = pade(1, 1);")
    assert m.num("num").tolist() == [[-1, 2]] and m.num("den").tolist() == [[1, 2]]
    m.run("[num, den] = pade(0.5, 2);")
    assert close(m.num("num"), [[1, -12, 48]]) and close(m.num("den"), [[1, 12, 48]])
    m.run("G = tf(1, [1 1], 'InputDelay', 0.5); P = pade(G, 3);")
    assert not m.num("hasdelay(P)") and m.num("order(P)") == 4
    assert abs(m.num("evalfr(P, 1i)") - m.num("evalfr(G, 1i)")) < 1e-5
    assert close(m.num("dcgain(feedback(pade(G, 2), 1))"), 0.5)
    m.run("Pk = pade(G, Inf);")
    assert m.num("totaldelay(Pk)") == 0.5


# -- reduction and realizations -------------------------------------------------------------------

def test_minreal_and_sminreal(m):
    m.run("[n, d] = tfdata(minreal(tf([1 1], [1 2 1])), 'v');")
    assert close(m.num("n"), [[0, 1]]) and close(m.num("d"), [[1, 1]])
    out = m.run("s = minreal(ss([-1 0; 0 -2], [1; 0], [1 1], 0));")
    assert "1 state removed." in out and m.num("order(s)") == 1
    assert close(m.num("dcgain(s)"), 1)
    m.run("z = minreal(zpk([-1 -2], [-1 -3 -2.0000000001], 4));")
    assert m.num("order(z)") == 1
    assert m.num("order(minreal(ss([-1 1; 0 -2], [0; 1], [1 0], 0)))") == 2      # nothing to remove
    assert m.num("order(sminreal(ss([-1 0; 0 -2], [1; 0], [1 1], 0)))") == 1
    assert m.num("order(ss(tf(1, [1 3 2]) * tf([1 1], 1), 'minimal'))") == 1


def test_gramians_and_covariance(m):
    assert close(m.num("gram(ss(-2, 3, 1, 0), 'c')"), 9 / 4)
    assert close(m.num("gram(ss(-2, 3, 5, 0), 'o')"), 25 / 4)
    assert close(m.num("gram(ss(0.5, 1, 1, 0, 1), 'c')"), 4 / 3)
    m.run("s = ss([0 1; -2 -3], [0; 1], [1 0], 0); W = gram(s, 'c'); R = gram(s, 'cf');")
    assert close(m.num("s.A * W + W * s.A' + s.B * s.B'"), np.zeros((2, 2)))
    assert close(m.num("R' * R"), m.num("W"))
    m.run("[P, Q] = covar(tf(1, [1 1]), 2);")
    assert close(m.num("P"), 1) and close(m.num("Q"), 1)
    assert close(m.num("covar(tf(1, [1 -0.5], 1), 3)"), 3 / 0.75)
    with pytest.raises(MatlabError, match="stable"):
        m.run("gram(ss(1, 1, 1, 0), 'c')")


def test_balreal_hsvd_balred(m):
    m.run("s = ss(tf([1 5], [1 6 11 6])); [sb, g, T, Ti] = balreal(s);")
    assert close(m.num("gram(sb, 'c')"), np.diag(col(m, "g")), 1e-8)
    assert close(m.num("gram(sb, 'o')"), np.diag(col(m, "g")), 1e-8)
    assert close(m.num("T * Ti"), np.eye(3), 1e-8)
    assert close(m.num("sb.A"), m.num("T * s.A * Ti"), 1e-8)
    assert close(col(m, "hsvd(s)"), col(m, "g"), 1e-8) and np.all(np.diff(col(m, "g")) <= 0)
    assert close(m.num("hsvd(tf(1, [1 1]))"), 0.5)
    m.run("r = balred(tf([1 5], [1 6 11 6]), 2); rt = balred(s, 1, 'StateProjection', 'Truncate');")
    assert m.str("class(r)") == "tf" and m.num("order(r)") == 2 and m.num("order(rt)") == 1
    assert close(m.num("dcgain(r)"), 5 / 6, 1e-8)                 # the default keeps the DC gain
    assert m.num("isstable(r)") and m.num("norm(r - tf([1 5], [1 6 11 6]), inf)") < 2 * m.num("g(3)") + 1e-9
    m.run("[gs, gu] = stabsep(tf(1, [1 1 -2]));")
    assert close(m.num("evalfr(gs + gu, 2i)"), m.num("evalfr(tf(1, [1 1 -2]), 2i)"))
    assert close(m.num("pole(gs)"), -2) and close(m.num("pole(gu)"), 1)
    m.run("ru = balred(tf(1, [1 1 -2]) + tf(1, [1 5]) + tf(1, [1 50]), 2);")
    assert np.any(np.abs(col(m, "pole(ru)") - 1) < 1e-8)          # the unstable pole is kept


def test_ctrb_obsv_and_staircase_forms(m):
    assert m.num("ctrb([0 1; 0 0], [0; 1])").tolist() == [[0, 1], [1, 0]]
    assert m.num("obsv([0 1; 0 0], [1 0])").tolist() == [[1, 0], [0, 1]]
    assert m.num("rank(ctrb(ss([-1 0; 0 -2], [1; 0], [1 1], 0)))") == 1
    m.run("A = [1 1; 4 -2]; B = [1 -1; 1 -1]; C = eye(2); [Ab, Bb, Cb, T, k] = ctrbf(A, B, C);")
    assert close(m.num("T * T'"), np.eye(2)) and close(m.num("Ab"), m.num("T * A * T'"))
    assert abs(m.num("Ab(1, 2)")) < 1e-12 and np.allclose(m.num("Bb(1, :)"), 0, atol=1e-12)
    assert close(m.num("Ab(1, 1)"), -3) and m.num("sum(k)") == 1
    m.run("[Ao, Bo, Co, To, ko] = obsvf(A, B, C);")
    assert m.num("sum(ko)") == 2 and close(m.num("Ao"), m.num("A"))
    m.run("[Ao, Bo, Co, To, ko] = obsvf([-1 0; 0 -2], [1; 1], [1 0]);")
    assert m.num("sum(ko)") == 1 and abs(m.num("Ao(2, 1)")) < 1e-12 and abs(m.num("Co(1)")) < 1e-12


def test_canonical_forms_and_state_transformations(m):
    m.run("s = ss([0 1 0; 0 0 1; -6 -11 -6], [0; 0; 1], [1 0 0], 0); [c, T] = canon(s, 'modal');")
    assert close(m.num("c.A"), np.diag([-1, -2, -3]), 1e-8)
    assert close(m.num("evalfr(c, 2i)"), m.num("evalfr(s, 2i)"), 1e-9)
    assert close(m.num("T * s.A / T"), m.num("c.A"), 1e-8)
    m.run("c2 = canon(ss(tf(1, [1 2 5])), 'modal');")
    assert close(m.num("c2.A"), [[-1, 2], [-2, -1]], 1e-9)
    m.run("[cc, Tc] = canon(s, 'companion');")
    assert close(m.num("cc.A"), [[0, 0, -6], [1, 0, -11], [0, 1, -6]], 1e-9) and close(m.num("cc.B"), [[1], [0], [0]])
    m.run("t2 = ss2ss(s, [1 2 0; 0 1 0; 1 0 3]); xp = xperm(s, [3 1 2]);")
    assert close(m.num("evalfr(t2, 1+1i)"), m.num("evalfr(s, 1+1i)"))
    assert close(m.num("evalfr(xp, 1+1i)"), m.num("evalfr(s, 1+1i)"))
    assert m.num("xp.A(1, 1)") == -6
    with pytest.raises(MatlabError, match="defective"):
        m.run("canon(ss([0 1; 0 0], [0; 1], [1 0], 0), 'modal')")


# -- Lyapunov and Riccati equations ---------------------------------------------------------------

def test_lyapunov_equations(m):
    assert close(m.num("lyap([-1 0; 0 -2], eye(2))"), np.diag([0.5, 0.25]))
    assert close(m.num("dlyap(0.5, 1)"), 4 / 3)
    m.run("A = [0 1; -2 -3]; Q = [2 1; 1 3]; X = lyap(A, Q);")
    assert close(m.num("A*X + X*A' + Q"), np.zeros((2, 2)))
    m.run("B = [-1 2; 0 -4]; C = [1 2; 3 4]; X = lyap(A, B, C);")
    assert close(m.num("A*X + X*B + C"), np.zeros((2, 2)))
    m.run("E = [2 1; 0 1]; X = lyap(A, Q, [], E);")
    assert close(m.num("A*X*E' + E*X*A' + Q"), np.zeros((2, 2)))
    m.run("Ad = [0.5 0.1; 0 0.3]; X = dlyap(Ad, Q); Xs = dlyap(Ad, [0.2 0; 0.1 0.4], C);")
    assert close(m.num("Ad*X*Ad' - X + Q"), np.zeros((2, 2)))
    assert close(m.num("Ad*Xs*[0.2 0; 0.1 0.4] - Xs + C"), np.zeros((2, 2)))
    m.run("R = lyapchol(A, [1; 2]); X = lyap(A, [1; 2] * [1 2]);")
    assert close(m.num("R' * R"), m.num("X"))
    with pytest.raises(MatlabError, match="no unique solution"):
        m.run("lyap([1 0; 0 -1], eye(2))")


def test_riccati_equations(m):
    m.run("[X, L, G] = care(-1, 1, 1, 1);")
    assert close(m.num("X"), np.sqrt(2) - 1) and close(m.num("L"), -np.sqrt(2)) and close(m.num("G"), np.sqrt(2) - 1)
    m.run("[X, K, L] = icare(-1, 1, 1, 1);")                   # same solution, different output order
    assert close(m.num("X"), np.sqrt(2) - 1) and close(m.num("K"), np.sqrt(2) - 1) and close(m.num("L"), -np.sqrt(2))
    m.run("[X, L, G] = dare(1, 1, 1, 1);")                     # x = x - x^2/(x + 1) + 1: the golden ratio
    phi = (1 + np.sqrt(5)) / 2
    assert close(m.num("X"), phi) and close(m.num("G"), phi / (1 + phi)) and close(m.num("L"), 1 - phi / (1 + phi))
    m.run("A = [0 1; -1 0.5]; B = [0; 1]; Q = [2 0; 0 1]; R = 3; S = [0.1; 0.2]; X = care(A, B, Q, R, S);")
    assert close(m.num("A'*X + X*A - (X*B + S)/R*(B'*X + S') + Q"), np.zeros((2, 2)), 1e-8)
    assert np.all(m.num("real(eig(A - B*((B'*X + S')/R)))") < 0)
    m.run("X = idare(A, B, Q, R, S);")
    assert close(m.num("A'*X*A - X - (A'*X*B + S)/(B'*X*B + R)*(B'*X*A + S') + Q"), np.zeros((2, 2)), 1e-8)
    m.run("E = [1 0.2; 0 2]; X = care(A, B, Q, R, [], E);")
    assert close(m.num("A'*X*E + E'*X*A - E'*X*B/R*B'*X*E + Q"), np.zeros((2, 2)), 1e-8)
    with pytest.raises(MatlabError, match="stabilizing"):
        m.run("care(1, 0, 1, 1)")


# -- linear-quadratic design, estimators, pole placement ------------------------------------------

def test_lqr_family(m):
    m.run("[K, S, P] = lqr([0 1; 0 0], [0; 1], eye(2), 1);")
    assert close(m.num("K"), [[1, np.sqrt(3)]]) and close(m.num("S"), [[np.sqrt(3), 1], [1, np.sqrt(3)]])
    assert close(eigs(m, "P"), eigs(m, "eig([0 1; 0 0] - [0; 1] * K)"))
    phi = (1 + np.sqrt(5)) / 2
    assert close(m.num("dlqr(1, 1, 1, 1)"), phi / (1 + phi))
    assert close(m.num("lqr(ss(1, 1, 1, 0, 1), 1, 1)"), phi / (1 + phi))       # discrete model: dlqr
    assert close(m.num("lqr(ss([0 1; 0 0], [0; 1], [1 0], 0), eye(2), 1)"), [[1, np.sqrt(3)]])
    # cross term: completing the square moves it into A and Q
    m.run("A = [0 1; -1 -1]; B = [0; 1]; N = [0.2; 0.1]; K = lqr(A, B, eye(2), 2, N);")
    m.run("K0 = lqr(A - B*N'/2, B, eye(2) - N*N'/2, 2) + N'/2;")
    assert close(m.num("K"), m.num("K0"))
    m.run("K = lqry(ss(A, B, [1 0], 0), 5, 2); Kx = lqr(A, B, [5 0; 0 0], 2);")
    assert close(m.num("K"), m.num("Kx"))
    m.run("Kd = lqrd(A, B, eye(2), 2, 1e-4); Kc = lqr(A, B, eye(2), 2);")       # fast sampling: the same gain
    assert np.max(np.abs(m.num("Kd") - m.num("Kc"))) < 1e-3
    with pytest.raises(MatlabError, match="positive definite"):
        m.run("lqr(A, B, eye(2), 0)")


def test_lqi_adds_integral_action(m):
    m.run("s = ss([0 1; -2 -3], [0; 1], [1 0], 0); [K, S, e] = lqi(s, eye(3), 1);")
    assert m.num("size(K)").tolist() == [[1, 3]] and np.all(m.num("real(e)") < 0)
    # closed loop with u = -Kx*x - Ki*xi, xi' = r - y tracks a constant reference
    m.run("Acl = [s.A - s.B*K(1:2), -s.B*K(3); -s.C, 0]; T = ss(Acl, [0; 0; 1], [s.C 0], 0);")
    assert close(m.num("dcgain(T)"), 1) and close(eigs(m, "eig(Acl)"), eigs(m, "e"))


def test_place_and_acker(m):
    assert close(m.num("place([0 1; 0 0], [0; 1], [-1 -2])"), [[2, 3]])
    assert close(m.num("acker([0 1; 0 0], [0; 1], [-1 -1])"), [[1, 2]])
    m.run("A = [0 1 0; 0 0 1; -1 -2 -3]; B = [0 0; 1 0; 0 1]; p = [-2 -3+1i -3-1i];")
    m.run("[K, prec] = place(A, B, p);")
    assert close(eigs(m, "eig(A - B*K)"), np.sort_complex([-2, -3 + 1j, -3 - 1j]), 1e-8) and m.num("prec") >= 8
    m.run("K1 = place(A, B(:, 2), p); K2 = acker(A, B(:, 2), p);")
    assert close(m.num("K1"), m.num("K2"), 1e-8)
    assert close(eigs(m, "eig(A - B(:, 2)*K1)"), np.sort_complex([-2, -3 + 1j, -3 - 1j]), 1e-8)
    with pytest.raises(MatlabError, match="rank"):
        m.run("place([0 1; 0 0], [0; 1], [-1 -1])")
    with pytest.raises(MatlabError, match="not controllable"):
        m.run("place([-1 0; 0 -2], [1; 0], [-3 -4])")
    with pytest.raises(MatlabError, match="conjugate pairs"):
        m.run("acker([0 1; 0 0], [0; 1], [-1+1i -2])")


def test_kalman_and_estimators(m):
    m.run("[kest, L, P] = kalman(ss(-1, [1 1], 1, [0 0]), 1, 1);")
    assert close(m.num("L"), np.sqrt(2) - 1) and close(m.num("P"), np.sqrt(2) - 1)
    assert m.num("size(kest)").tolist() == [[2, 2]] and close(m.num("kest.A"), -np.sqrt(2))
    m.run("[L2, P2, E2] = lqe(-1, 1, 1, 1, 1);")
    assert close(m.num("L2"), np.sqrt(2) - 1) and close(m.num("E2"), -np.sqrt(2))
    # discrete scalar filter: P = a^2 P - a^2 P^2/(P + r) + q
    m.run("[kd, Ld, Pd, Md, Zd] = kalman(ss(0.5, [1 1], 1, [0 0], 1), 1, 1);")
    pd = m.num("Pd").item()
    assert abs(0.25 * pd - 0.25 * pd ** 2 / (pd + 1) + 1 - pd) < 1e-10
    assert close(m.num("Md"), pd / (pd + 1)) and close(m.num("Ld"), 0.5 * pd / (pd + 1))
    assert close(m.num("Zd"), pd / (pd + 1))
    m.run("[Mq, Pq, Zq] = dlqe(0.5, 1, 1, 1, 1);")
    assert close(m.num("Mq"), m.num("Md")) and close(m.num("Pq"), pd)
    with pytest.raises(MatlabError, match="only 1 input"):
        m.run("kalman(ss(-1, 1, 1, 0), eye(2), 1)")


def test_separation_principle_for_regulators(m):
    m.run("A = [0 1; -2 0.5]; B = [0; 1]; C = [1 0]; s = ss(A, B, C, 0);")
    m.run("K = place(A, B, [-2 -3]); L = place(A', C', [-8 -9])'; r = reg(s, K, L);")
    m.run("cl = feedback(s, r, +1);")                           # regulators use positive feedback
    assert close(eigs(m, "pole(cl)"), np.sort_complex([-9, -8, -3, -2]), 1e-8)
    m.run("e = estim(s, L);")
    assert m.num("size(e)").tolist() == [[3, 1]] and close(eigs(m, "pole(e)"), [-9, -8], 1e-8)
    m.run("e2 = estim(s, L, 1, 1);")                            # with the input known
    assert m.num("size(e2)").tolist() == [[3, 2]]
    m.run("rl = lqg(s, eye(3), eye(3)); Kq = lqr(A, B, eye(2), 1); Lq = lqe(A, eye(2), C, eye(2), 1);")
    both = np.concatenate([m.num("eig(A - B*Kq)").ravel(), m.num("eig(A - Lq*C)").ravel()])
    assert close(eigs(m, "pole(feedback(s, rl, +1))"), np.sort_complex(both), 1e-7)
    m.run("[kest, Lk] = kalman(ss(A, [B eye(2)], C, [0 0 0]), eye(2), 1); rk = lqgreg(kest, Kq);")
    assert close(m.num("evalfr(rk, 1i)"), m.num("evalfr(rl, 1i)"), 1e-8)
    m.run("sd = c2d(s, 0.1); rd = lqg(sd, eye(3), eye(3));")
    assert np.all(np.abs(m.num("pole(feedback(sd, rd, +1))")) < 1)


# -- PID tuning -----------------------------------------------------------------------------------

def test_pidtune_reference_design(m):
    # at the crossover the loop gain is 1 and the phase margin is the target
    m.run("G = zpk([], [-1 -1 -1], 1); [C, info] = pidtune(G, 'PI');")
    assert m.str("class(C)") == "pid" and m.num("info.Stable")
    wc = m.num("info.CrossoverFrequency").item()
    assert abs(m.num("info.PhaseMargin") - 60) < 0.5 and 0.45 < wc < 0.6
    assert abs(m.num("C.Kp") - 1.14) < 0.03 and abs(m.num("C.Ki") - 0.454) < 0.005 and m.num("C.Kd") == 0
    m.run("[C1, info1] = pidtune(G, 'PI', 1.0);")              # crossover given: exact gain there
    phi = np.radians(-1.0)                                      # the least lag a PI controller can add
    assert close(m.num("C1.Kp"), np.cos(phi) * 2 ** 1.5, 1e-6) and close(m.num("C1.Ki"), -np.sin(phi) * 2 ** 1.5, 1e-6)
    assert abs(m.num("info1.PhaseMargin") - 44) < 0.05 and close(m.num("info1.CrossoverFrequency"), 1, 1e-6)
    m.run("[C2, info2] = pidtune(G, 'PIDF', 1.0);")
    assert abs(m.num("info2.PhaseMargin") - 60) < 0.01 and m.num("C2.Tf") > 0 and m.num("C2.Kd") > 0
    m.run("[Gm, Pm, Wcg, Wcp] = margin(C2 * G);")
    assert abs(m.num("Pm") - 60) < 0.01 and abs(m.num("Wcp") - 1) < 1e-6


@pytest.mark.parametrize("plant,kind", [
    ("tf(1, [1 1])", "PI"), ("tf(1, [1 1])", "P"), ("tf(1, [1 1])", "I"), ("tf(1, [1 0])", "P"),
    ("tf(1, [1 0 0])", "PD"), ("tf(1, [1 0 0])", "PDF"), ("tf(1, [1 3 3 1])", "PID"), ("tf(-2, [5 1])", "PID"),
    ("tf(1, [1 -1])", "PI"), ("tf(1, [10 1], 'InputDelay', 2)", "PI"), ("tf(1, [10 1], 'InputDelay', 2)", "PIDF"),
    ("tf([1 -1], [1 3 2])", "PI"), ("c2d(tf(1, [1 3 2]), 0.1)", "PI"), ("c2d(tf(1, [1 3 2]), 0.1)", "PIDF"),
    ("ss([0 1; -1 -0.1], [0; 1], [1 0], 0)", "PID"), ("tf(1, [1 2 0])", "PIDF"),
])
def test_pidtune_stabilizes_with_margin(m, plant, kind):
    m.run(f"G = {plant}; [C, info] = pidtune(G, '{kind}');")
    assert m.num("info.Stable")
    if "Delay" in plant:
        m.run("T = feedback(pade(C * G, 8), 1);")
    else:
        m.run("T = feedback(C * G, 1);")
    assert m.num("isstable(T)")
    m.run("S = allmargin(C * G);")
    assert np.min(np.abs(m.num("S.PhaseMargin"))) >= 44
    if "I" in kind:
        assert close(m.num("dcgain(T)"), 1, 1e-6)       # integral action removes the steady-state error
    expected = {"P": "P", "I": "I", "PI": "PI", "PD": "PD", "PDF": "PDF", "PID": "PID", "PIDF": "PIDF"}[kind]
    got = ("P" if m.num("C.Kp") != 0 else "") + ("I" if m.num("C.Ki") != 0 else "") + \
        ("D" if m.num("C.Kd") != 0 else "") + ("F" if m.num("C.Tf") != 0 else "")
    assert got == expected


def test_pidtune_options_and_forms(m):
    m.run("G = tf(1, [1 3 3 1]); o = pidtuneOptions('PhaseMargin', 45); [C, info] = pidtune(G, 'PID', o);")
    assert abs(m.num("info.PhaseMargin") - 45) < 0.5
    m.run("[Cs, i2] = pidtune(G, pidstd(1, 1, 1));")
    assert m.str("class(Cs)") == "pidstd" and m.num("Cs.Ti") > 0 and m.num("Cs.Td") > 0 and m.num("i2.Stable")
    m.run("Cd = pidtune(c2d(G, 0.1), pid(1, 1, 'Ts', 0.1, 'IFormula', 'Trapezoidal'));")
    assert m.num("Cd.Ts") == 0.1 and m.str("Cd.IFormula") == "Trapezoidal" and m.num("Cd.Kd") == 0
    m.run("o2 = pidtuneOptions('DesignFocus', 'reference-tracking'); Cr = pidtune(G, 'PID', 1, o2);"
          "o3 = pidtuneOptions('DesignFocus', 'disturbance-rejection'); Cq = pidtune(G, 'PID', 1, o3);")
    assert m.num("Cq.Ki") > m.num("Cr.Ki")                       # stronger integral action
    with pytest.raises(MatlabError, match="2-degree-of-freedom"):
        m.run("pidtune(G, 'PI2')")
    with pytest.raises(MatlabError, match="SISO"):
        m.run("pidtune([G, G], 'PI')")
    with pytest.raises(MatlabError, match="interactive design app"):
        m.run("sisotool(G)")


# -- polynomial, zero-pole and state-space data ---------------------------------------------------

def test_data_conversions(m):
    m.run("[b, a] = ss2tf([0 1; -2 -3], [0; 1], [1 0; 1 1], [0; 0]);")
    assert m.num("b").tolist() == [[0, 0, 1], [0, 1, 1]] and m.num("a").tolist() == [[1, 3, 2]]
    m.run("[A, B, C, D] = tf2ss([1 2; 0 3], [1 3 2]);")
    assert m.num("A").tolist() == [[-3, -2], [1, 0]] and m.num("C").tolist() == [[1, 2], [0, 3]]
    m.run("[A, B, C, D] = tf2ss([2 5 3], [1 3 2]);")           # biproper: 2 + (-s - 1)/(s^2 + 3 s + 2)
    assert m.num("D") == 2 and m.num("C").tolist() == [[-1, -1]]
    m.run("[num, den] = zp2tf([-1; -2], [-3 -4 -5], 2);")
    assert m.num("num").tolist() == [[0, 2, 6, 4]] and m.num("den").tolist() == [[1, 12, 47, 60]]
    m.run("[z, p, k] = tf2zp([2 6 4], [1 12 47 60]);")
    assert sorted(col(m, "z")) == [-2, -1] and close(sorted(col(m, "p").real), [-5, -4, -3]) and m.num("k") == 2
    m.run("[z, p, k] = ss2zp([0 1; -2 -3], [0; 1], [1 1], 0); [A, B, C, D] = zp2ss(-1, [-2 -3], 4);")
    assert close(m.num("z"), -1) and m.num("k") == 1
    assert close(m.num("C / (2i*eye(2) - A) * B + D"), 4 * (2j + 1) / ((2j + 2) * (2j + 3)))
    with pytest.raises(MatlabError, match="several inputs"):
        m.run("ss2tf(zeros(2), eye(2), [1 0], [0 0])")
