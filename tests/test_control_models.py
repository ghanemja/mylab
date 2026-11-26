"""Control-system model objects: construction, properties, arithmetic, conversion, display."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def close(a, b, tol=1e-10):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), rtol=tol, atol=tol)


# -- construction and data access ---------------------------------------------------------------

def test_tf_data_is_padded_to_equal_length(m):
    m.run("G = tf(1, [1 2 1]); [num, den, ts] = tfdata(G, 'v');")
    assert m.num("num").tolist() == [[0, 0, 1]]
    assert m.num("den").tolist() == [[1, 2, 1]]
    assert m.num("ts") == 0
    m.run("[n2, d2] = tfdata(tf([1 2 3], [4 5]), 'v');")       # improper: the denominator is padded
    assert m.num("d2").tolist() == [[0, 4, 5]]
    assert m.str("class(tfdata(G))") == "cell"


def test_tf_from_laplace_variable_uses_exact_polynomial_algebra(m):
    m.run("s = tf('s'); G = (2*s + 1) / (s^2 + 3*s + 2); [num, den] = tfdata(G, 'v');")
    assert m.num("num").tolist() == [[0, 2, 1]]
    assert m.num("den").tolist() == [[1, 3, 2]]
    m.run("H = 1 / (s * (s + 1)^2); [n, d] = tfdata(H, 'v');")
    assert m.num("d").tolist() == [[1, 2, 1, 0]]


def test_discrete_variable_and_filter_form(m):
    m.run("z = tf('z', 0.1); H = (z + 1) / (z^2 - 0.5*z + 0.1);")
    assert m.num("H.Ts") == 0.1
    assert m.num("H.den{1}").tolist() == [[1, -0.5, 0.1]]
    # coefficients of z^-1 in ascending powers: (1 + 2 z^-1) / (1 + 0.5 z^-1 + 0.25 z^-2)
    m.run("F = filt([1 2], [1 0.5 0.25], 0.1);")
    assert m.num("F.num{1}").tolist() == [[1, 2, 0]]
    assert m.str("F.Variable") == "z^-1"
    assert close(m.num("evalfr(F, 2)"), (1 + 2 / 2) / (1 + 0.5 / 2 + 0.25 / 4))
    assert m.num("tf('z').Ts") == -1


def test_mimo_tf_from_cell_arrays(m):
    m.run("M = tf({1, [1 0]; [2 1], 3}, {[1 1], [1 2]; [1 3 2], 1});")
    assert m.num("size(M)").tolist() == [[2, 2]]
    assert close(m.num("dcgain(M)"), [[1, 0], [0.5, 3]])
    m.run("Mc = tf({1; [1 0]}, [1 2 1]);")          # one denominator shared by both outputs
    assert m.num("Mc.den{2}").tolist() == [[1, 2, 1]]
    assert close(m.num("dcgain(tf([1 2; 3 4]))"), [[1, 2], [3, 4]])


def test_zpk_and_ss_construction(m):
    m.run("Z = zpk([-1 -3], [0 -2 -2], 5); [z, p, k] = zpkdata(Z, 'v');")
    assert sorted(m.num("z").ravel().tolist()) == [-3, -1]
    assert m.num("p").shape == (3, 1)
    assert m.num("k") == 5
    m.run("sys = ss([0 1; -2 -3], [0; 1], [1 0], 0); [A, B, C, D] = ssdata(sys);")
    assert m.num("A").tolist() == [[0, 1], [-2, -3]]
    m.run("s2 = ss([0 1; -2 -3], [0 0; 1 1], [1 0; 0 1], 0);")     # D = 0 expands to a zero matrix
    assert m.num("s2.D").tolist() == [[0, 0], [0, 0]]
    with pytest.raises(MatlabError, match="same number of rows"):
        m.run("ss([0 1; -2 -3], [1 2 3]', [1 0], 0)")
    with pytest.raises(MatlabError, match="denominator"):
        m.run("tf(1, 0)")


def test_properties_get_set_and_aliases(m):
    m.run("G = tf(1, [1 2 1]); G.Numerator = [1 5]; G.den{1} = [1 4 4];")
    assert m.num("G.num{1}").tolist() == [[0, 1, 5]]
    assert m.num("dcgain(G)") == 1.25
    m.run("G.InputName = 'u'; G.OutputName = 'y'; G.Name = 'plant'; G.InputDelay = 0.5;")
    assert m.str("G.InputName{1}") == "u"
    assert m.str("get(G, 'Name')") == "plant"
    assert m.num("G.inputdelay") == 0.5
    m.run("H = set(G, 'InputDelay', 2, 'OutputName', 'out');")
    assert m.num("H.InputDelay") == 2 and m.num("G.InputDelay") == 0.5     # value semantics
    m.run("sys = ss(-1, 2, 3, 0); sys.A = -4; sys.b(1) = 5;")
    assert m.num("sys.a") == -4 and m.num("sys.B") == 5
    assert m.num("dcgain(sys)") == 3 * 5 / 4
    assert "Numerator" in m.run("disp(fieldnames(get(tf(1, [1 1])))')")
    with pytest.raises(MatlabError, match="Unrecognized property"):
        m.run("G.Bogus")
    with pytest.raises(MatlabError, match="whole number"):
        m.run("Gd = tf(1, [1 -0.5], 0.1); Gd.InputDelay = 1.5;")


def test_name_value_pairs_in_constructor(m):
    m.run("G = tf(1, [1 1], 'InputDelay', 0.5, 'OutputName', 'y');")
    assert m.num("G.InputDelay") == 0.5
    assert m.num("hasdelay(G)") and m.num("totaldelay(G)") == 0.5
    m.run("Gd = tf([1 2], [1 3 5], 'Ts', 0.1, 'Variable', 'z^-1');")
    assert m.num("Gd.Ts") == 0.1 and m.num("Gd.num{1}").tolist() == [[1, 2, 0]]
    with pytest.raises(MatlabError, match="name/value"):
        m.run("tf(1, [1 1], 'InputDelay')")


# -- arithmetic and interconnection -------------------------------------------------------------

def test_series_parallel_feedback_closed_forms(m):
    m.run("G1 = tf(1, [1 1]); G2 = tf(2, [1 2]);")
    m.run("[n, d] = tfdata(series(G1, G2), 'v');")
    assert m.num("n").tolist() == [[0, 0, 2]] and m.num("d").tolist() == [[1, 3, 2]]
    m.run("[n, d] = tfdata(parallel(G1, G2), 'v');")       # 1/(s+1) + 2/(s+2) = (3s+4)/(s^2+3s+2)
    assert m.num("n").tolist() == [[0, 3, 4]] and m.num("d").tolist() == [[1, 3, 2]]
    m.run("[n, d] = tfdata(feedback(G1, G2), 'v');")       # (s+2)/((s+1)(s+2)+2)
    assert m.num("n").tolist() == [[0, 1, 2]] and m.num("d").tolist() == [[1, 3, 4]]
    m.run("[n, d] = tfdata(feedback(G1, 2, +1), 'v');")    # positive feedback: 1/(s - 1)
    assert m.num("d").tolist() == [[1, -1]]
    m.run("[n, d] = tfdata(G1 + G1, 'v');")                # equal denominators are not multiplied
    assert m.num("n").tolist() == [[0, 2]] and m.num("d").tolist() == [[1, 1]]


def test_operators_with_numbers(m):
    m.run("G = tf(1, [1 1]);")
    assert close(m.num("dcgain(2 * G)"), 2)
    assert close(m.num("dcgain(1 - G)"), 0)
    assert m.num("tfdata(1 / G, 'v')").tolist() == [[1, 1]]
    assert close(m.num("evalfr(G / 2, 1i)"), 0.5 / (1 + 1j))
    assert close(m.num("evalfr(G \\ 1, 1i)"), 1 + 1j)
    assert close(m.num("evalfr(-G, 0)"), -1)
    assert close(m.num("evalfr(G^3, 1)"), 1 / 8)
    assert close(m.num("evalfr(G^-2, 1)"), 4)
    assert m.num("isstatic(G^0)")
    assert close(m.num("evalfr(G', 2)"), 1 / (1 - 2))      # conjugate system G(-s)
    with pytest.raises(MatlabError, match="integer"):
        m.run("G^0.5")


def test_type_precedence(m):
    m.run("G = tf(1, [1 1]); Z = zpk(-1, [-2 -3], 4); S = ss(-1, 2, 3, 0); C = pid(2, 3);")
    assert m.str("class(G * Z)") == "zpk"
    assert m.str("class(G + S)") == "ss"
    assert m.str("class(Z * S)") == "ss"
    assert m.str("class(C * G)") == "tf"
    assert m.str("class(C * S)") == "ss"
    assert m.str("class(frd(G, [1 2]) * S)") == "frd"
    assert m.str("class([G, S])") == "ss"
    assert close(m.num("dcgain(G + S)"), 1 + 6)
    assert close(m.num("evalfr(G * Z, 1i)"), 4 * (1 + 1j) / ((1 + 1j) * (2 + 1j) * (3 + 1j)))


def test_zpk_arithmetic_keeps_roots(m):
    m.run("Z = zpk(-1, [-2 -3], 4) * zpk([], -5, 2); [z, p, k] = zpkdata(Z, 'v');")
    assert sorted(m.num("p").ravel().tolist()) == [-5, -3, -2]
    assert m.num("k") == 8
    m.run("Zs = zpk([], -1, 1) + zpk([], -2, 1); [z, p, k] = zpkdata(Zs, 'v');")    # (2s+3)/((s+1)(s+2))
    assert close(m.num("z"), -1.5) and m.num("k") == 2
    m.run("T = feedback(zpk([], [0 -2], 3), 1);")           # 3/(s^2 + 2s + 3)
    assert close(np.sort_complex(m.num("pole(T)").ravel()), [-1 - np.sqrt(2) * 1j, -1 + np.sqrt(2) * 1j])


def test_state_space_arithmetic_matches_transfer_functions(m):
    m.run("G1 = tf([1 2], [1 3 5]); G2 = tf(4, [1 6]); S1 = ss(G1); S2 = ss(G2); w = 1.7i;")
    for expr, ref in (("S1 * S2", "G1 * G2"), ("S1 + S2", "G1 + G2"), ("S1 - S2", "G1 - G2"),
                      ("feedback(S1, S2)", "feedback(G1, G2)"), ("inv(S2 + 1)", "inv(G2 + 1)"),
                      ("[S1, S2] * [S2; S1]", "2 * G1 * G2")):
        assert close(m.num(f"evalfr({expr}, w)"), m.num(f"evalfr({ref}, w)")), expr
    assert m.num("order(S1 * S2)") == 3


def test_concatenation_and_indexing(m):
    m.run("M = [tf(1, [1 1]), tf(1, [1 2]); tf(1, [1 3]), tf(1, [1 4])];")
    assert m.num("size(M)").tolist() == [[2, 2]]
    assert close(m.num("dcgain(M)"), [[1, 0.5], [1 / 3, 0.25]])
    assert close(m.num("dcgain(M(2, 1))"), 1 / 3)
    assert m.num("size(M(:, 2))").tolist() == [[2, 1]]
    assert close(m.num("dcgain(M(end, end))"), 0.25)
    m.run("M2 = M; M2(1, :) = []; M3 = M; M3(2, 2) = tf(5, [1 5]); M4 = [M; 1 2]; M5 = [1, tf(1, [1 1])];")
    assert m.num("size(M2)").tolist() == [[1, 2]]
    assert close(m.num("dcgain(M3)"), [[1, 0.5], [1 / 3, 1]])
    assert m.num("size(M4)").tolist() == [[3, 2]] and m.num("size(M5)").tolist() == [[1, 2]]
    assert close(m.num("dcgain(M.')"), [[1, 1 / 3], [0.5, 0.25]])
    m.run("[ny, nu] = size(M);")
    assert m.num("ny") == 2 and m.num("nu") == 2
    with pytest.raises(MatlabError, match="two subscripts"):
        m.run("M(3)")
    with pytest.raises(MatlabError, match="same number of outputs"):
        m.run("[M, tf(1, [1 1])]")


def test_named_channels_index_and_connect(m):
    m.run("P = tf(1, [1 2]); P.InputName = 'u'; P.OutputName = 'y';")
    m.run("C = pid(2, 3); C.InputName = 'e'; C.OutputName = 'u'; S = sumblk('e = r - y');")
    m.run("T = connect(P, C, S, 'r', 'y'); T2 = feedback(P * C, 1);")
    assert m.str("class(T)") == "ss"
    assert close(np.sort_complex(m.num("pole(T)").ravel()), np.sort_complex(m.num("pole(T2)").ravel()))
    assert close(m.num("dcgain(T)"), 1)
    assert m.str("T.InputName{1}") == "r" and m.str("T.OutputName{1}") == "y"
    m.run("T3 = connect(P, C, S, 'r', {'y', 'u'});")
    assert m.num("size(T3)").tolist() == [[2, 1]]
    assert close(m.num("evalfr(T3('u', 'r'), 1i)"), m.num("evalfr(feedback(C, P), 1i)"))
    with pytest.raises(MatlabError, match="no block has an input named"):
        m.run("connect(P, C, S, 'nope', 'y')")


def test_index_based_connect_and_append(m):
    m.run("blk = append(tf(1, [1 1]), tf(2, [1 2]));")
    # input 1 = external - output 2, input 2 = output 1: a loop of the two blocks
    m.run("T = connect(blk, [1 -2; 2 1], 1, 2); R = feedback(tf(1, [1 1]) * tf(2, [1 2]), 1);")
    assert close(m.num("evalfr(T, 0.5i)"), m.num("evalfr(R, 0.5i)"))
    assert m.num("size(append(tf(1, [1 1]), ss(-2, 1, 1, 0), 3))").tolist() == [[3, 3]]


def test_mimo_feedback_inverse_series_parallel_lft(m):
    m.run("M = [tf(1, [1 1]), tf(1, [1 2]); tf(1, [1 3]), tf(1, [1 4])]; K = dcgain(M);")
    assert close(m.num("dcgain(feedback(M, eye(2)))"), m.num("K / (eye(2) + K)"))
    assert close(m.num("dcgain(inv(M))"), m.num("inv(K)"), 1e-8)      # improper inverse via polynomials
    assert close(m.num("dcgain(series(M, M, [1 2], [2 1]))"), m.num("K(:, [2 1]) * K"))
    assert m.num("size(parallel(M, M, 1, 1, 2, 2))").tolist() == [[3, 3]]
    # lower LFT closes the last channel: P11 + P12 K (1 - P22 K)^-1 P21
    m.run("L = lft(ss(M), ss(tf(2, [1 5])));")
    assert close(m.num("dcgain(L)"), 1 + 0.5 * 0.4 / (1 - 0.25 * 0.4) / 3)
    m.run("F = feedback(M, tf(1, [1 1]), 2, 1);")          # loop from output 1 back to input 2 only
    g = m.num("K")
    h = 1.0
    ref = g + np.outer(g[:, 1], g[0, :]) * (-h) / (1 + g[0, 1] * h)
    assert close(m.num("dcgain(F)"), ref)


def test_sample_times_must_agree(m):
    m.run("Gc = tf(1, [1 1]); Gd = tf(1, [1 -0.5], 0.1);")
    with pytest.raises(MatlabError, match="Sampling times must agree"):
        m.run("Gc * Gd")
    with pytest.raises(MatlabError, match="Sampling times must agree"):
        m.run("Gd + tf(1, [1 -0.5], 0.2)")
    assert m.num("getfield(2 * Gd, 'Ts')") == 0.1
    assert m.num("getfield(tf(3) * Gd, 'Ts')") == 0.1          # a static gain adapts


# -- conversions ----------------------------------------------------------------------------------

def test_conversions_round_trip(m):
    m.run("G = tf([2 5], [1 4 13 0]); Z = zpk(G); S = ss(G); w = [0.3i, 1+2i];")
    assert m.str("class(Z)") == "zpk" and m.str("class(S)") == "ss"
    for x in ("Z", "S", "tf(Z)", "tf(S)", "zpk(S)", "ss(Z)"):
        assert close(m.num(f"[evalfr({x}, w(1)), evalfr({x}, w(2))]"),
                     m.num("[evalfr(G, w(1)), evalfr(G, w(2))]"), 1e-9), x
    m.run("[n, d] = tfdata(tf(ss([0 1; -2 -3], [0; 1], [1 0], 0)), 'v');")
    assert m.num("n").tolist() == [[0, 0, 1]] and m.num("d").tolist() == [[1, 3, 2]]
    m.run("[A, B, C, D] = ssdata(tf(1, [1 3 2]));")       # controller canonical form
    assert m.num("A").tolist() == [[-3, -2], [1, 0]] and m.num("C").tolist() == [[0, 1]]


def test_mimo_tf_to_ss_shares_denominators(m):
    m.run("M = tf({1, [1 0]; [2 1], 3}, {[1 1], [1 2]; [1 3 2], 1}); S = ss(M);")
    assert m.num("order(M)") == 3            # column 1: (s+1)(s+2), column 2: (s+2)
    assert close(m.num("evalfr(S, 0.7i)"), m.num("evalfr(M, 0.7i)"))
    m.run("T = tf(ss(diag([-1 -2]), eye(2), eye(2), 0));")      # structurally decoupled channels
    assert m.num("T.den{1, 1}").tolist() == [[1, 1]] and m.num("T.num{1, 2}").tolist() == [[0]]
    with pytest.raises(MatlabError, match="improper"):
        m.run("ss(tf([1 0 0], [1 1]))")


def test_queries(m):
    m.run("G = tf(1, [1 1]); Gd = tf(1, [1 -0.5], 0.1); M = [G, G];")
    assert m.num("isct(G)") and not m.num("isdt(G)") and m.num("isdt(Gd)") and not m.num("isct(Gd)")
    assert m.num("issiso(G)") and not m.num("issiso(M)")
    assert m.num("isstable(G)") and not m.num("isstable(tf(1, [1 -1]))")
    assert m.num("isstable(Gd)") and not m.num("isstable(tf(1, [1 -1.5], 0.1))")
    assert m.num("isproper(G)") and not m.num("isproper(tf([1 0 0], [1 1]))")
    assert m.num("order(tf(1, [1 3 3 1]))") == 3 and m.num("size(M, 2)") == 2
    assert m.num("size(ss(tf(1, [1 3 3 1])), 'order')") == 3
    assert m.num("isa(G, 'tf')") and m.num("isa(G, 'lti')") and not m.num("isa(G, 'ss')")
    assert m.num("isequal(tf(1, [1 1]), tf(1, [1 1]))") and not m.num("isequal(G, tf(1, [1 2]))")
    assert "Transfer function with 1 outputs and 2 inputs." in m.run("size(M)")
    assert "State-space model with 1 outputs, 1 inputs, and 3 states." in m.run("size(ss(tf(1, [1 3 3 1])))")


# -- delays ---------------------------------------------------------------------------------------

def test_delay_algebra(m):
    m.run("s = tf('s'); G = exp(-0.3*s) / (s + 1);")
    assert close(m.num("totaldelay(G)"), 0.3)
    assert close(m.num("evalfr(G, 2i)"), np.exp(-0.6j) / (1 + 2j))
    assert close(m.num("totaldelay(G * G)"), 0.6)
    assert close(m.num("totaldelay(G + G)"), 0.3)
    assert close(m.num("totaldelay([G, tf(1, [1 2])])"), [[0.3, 0]])
    with pytest.raises(MatlabError, match="pade"):
        m.run("feedback(G, 1)")
    with pytest.raises(MatlabError, match="pade"):
        m.run("G + tf(1, [1 1])")
    with pytest.raises(MatlabError, match="predict the future"):
        m.run("inv(G)")
    with pytest.raises(MatlabError, match="tau >= 0"):
        m.run("exp(s)")


def test_discrete_delays_are_absorbed_when_needed(m):
    m.run("Gd = tf(1, [1 -0.5], 0.1, 'InputDelay', 2); T = feedback(Gd, 1);")
    assert not m.num("hasdelay(T)")
    assert m.num("order(T)") == 3                          # z^2 (z - 0.5) + 1
    assert close(np.sort(m.num("abs(pole(T))").ravel()), np.sort(np.abs(np.roots([1, -0.5, 0, 1]))))
    assert m.num("order(absorbDelay(Gd))") == 3
    assert close(m.num("evalfr(absorbDelay(Gd), 0.3+0.2i)"), m.num("evalfr(Gd, 0.3+0.2i)"))


# -- PID objects ----------------------------------------------------------------------------------

def test_pid_to_tf_and_back(m):
    m.run("C = pid(1, 2, 3, 0.1); [n, d] = tfdata(tf(C), 'v');")
    # Kp + Ki/s + Kd s/(Tf s + 1) = ((Kp Tf + Kd) s^2 + (Kp + Ki Tf) s + Ki) / (Tf s^2 + s)
    assert close(m.num("n"), [[3.1, 1.2, 2]]) and close(m.num("d"), [[0.1, 1, 0]])
    m.run("[kp, ki, kd, tf_, ts] = piddata(tf([3.1 1.2 2], [0.1 1 0]));")
    assert [m.num(v).item() for v in ("kp", "ki", "kd", "tf_", "ts")] == pytest.approx([1, 2, 3, 0.1, 0])
    m.run("[kp, ti, td, n] = pidstddata(pid(2, 4, 1));")
    assert [m.num(v).item() for v in ("kp", "ti", "td")] == [2, 0.5, 0.5] and np.isinf(m.num("n"))
    assert close(m.num("evalfr(pidstd(2, 0.5, 0.5, 10), 1i)"), 2 * (1 + 1 / 0.5j + 0.5j / (0.05j + 1)))
    with pytest.raises(MatlabError, match="PID controller"):
        m.run("pid(tf(1, [1 1 1]))")


def test_pid_arithmetic_keeps_type_when_possible(m):
    m.run("C = pid(1, 2) + pid(3, 4, 5);")
    assert m.str("class(C)") == "pid"
    assert [m.num("C.Kp"), m.num("C.Ki"), m.num("C.Kd")] == [4, 6, 5]
    assert m.str("class(3 * pid(1, 2))") == "pid" and m.num("getfield(3 * pid(1, 2), 'Ki')") == 6
    assert m.str("class(pid(1, 2) * pid(1, 2))") == "tf"       # a double integrator is not a PID
    assert m.str("class(-pid(1, 2))") == "pid"
    assert close(m.num("dcgain(feedback(pid(2, 1) * tf(1, [1 1]), 1))"), 1)


def test_discrete_pid_formulas(m):
    ts = 0.01
    z = np.exp(0.3j)
    m.run("C = pid(2, 3, 0.5, 0.1, 0.01, 'IFormula', 'Trapezoidal', 'DFormula', 'BackwardEuler');")
    integ = ts / 2 * (z + 1) / (z - 1)
    deriv = 1 / (0.1 + ts * z / (z - 1))
    assert close(m.num("evalfr(C, exp(0.3i))"), 2 + 3 * integ + 0.5 * deriv)
    m.run("C2 = pid(tf(C), 'IFormula', 'Trapezoidal', 'DFormula', 'BackwardEuler');")
    assert close(m.num("[C2.Kp C2.Ki C2.Kd C2.Tf]"), [[2, 3, 0.5, 0.1]], 1e-8)


# -- frequency-response data ----------------------------------------------------------------------

def test_frd_models(m):
    m.run("w = [0.1 1 10]; F = frd(tf(1, [1 1]), w); [r, f] = frdata(F, 'v');")
    assert close(m.num("r").ravel(), 1 / (1 + 1j * np.array([0.1, 1, 10])))
    m.run("r2 = frdata(F * F + 1, 'v'); r3 = frdata(feedback(F, 1), 'v'); r4 = frdata(inv(F), 'v');")
    g = 1 / (1 + 1j * np.array([0.1, 1, 10]))
    assert close(m.num("r2").ravel(), g * g + 1)
    assert close(m.num("r3").ravel(), g / (1 + g))
    assert close(m.num("r4").ravel(), 1 / g)
    m.run("Fh = frd([1 2 3], [1 10 100], 'FrequencyUnit', 'Hz');")
    assert m.str("Fh.FrequencyUnit") == "Hz"
    assert close(m.num("freqresp(Fh, 2*pi*10)"), 2)
    with pytest.raises(MatlabError, match="different frequencies"):
        m.run("F + frd(tf(1, [1 1]), [1 2 3])")
    with pytest.raises(MatlabError, match="cannot be converted"):
        m.run("tf(F)")


# -- display --------------------------------------------------------------------------------------

def test_tf_display(m):
    out = m.run("G = tf([1 2], [1 3 2])")
    assert out == ("\nG =\n \n      s + 2\n  -------------\n  s^2 + 3 s + 2\n \n"
                   "Continuous-time transfer function.\n\n")
    out = m.run("H = tf([1 -0.5], [1 0.2 0.04], 0.1)")
    assert "  z^2 + 0.2 z + 0.04\n" in out
    assert "Sample time: 0.1 seconds\nDiscrete-time transfer function." in out
    assert m.run("K = tf(5)") == "\nK =\n \n  5\n \nStatic gain.\n\n"
    assert "  exp(-0.5*s) * -----\n" in m.run("D = tf(1, [1 1], 'InputDelay', 0.5)")
    out = m.run("M = [tf(1, [1 1]); tf(-2, [1 0])]")
    assert "From input 1 to output...\n" in out and "   2:  --\n" in out
    assert "1 + 2 z^-1" in m.run("F = filt([1 2], [1 0.5])")


def test_zpk_ss_pid_display(m):
    out = m.run("Z = zpk([-1 -3], [0 -2 -2 -1+2i -1-2i], 5)")
    assert "       5 (s+1) (s+3)\n" in out and "  s (s+2)^2 (s^2 + 2s + 5)\n" in out
    assert out.rstrip().endswith("Continuous-time zero/pole/gain model.")
    out = m.run("sys = ss([0 1; -2 -3], [0; 1], [1 0], 0)")
    assert "  A = \n       x1  x2\n   x1   0   1\n   x2  -2  -3\n" in out
    assert "  D = \n       u1\n   y1   0\n" in out
    assert out.rstrip().endswith("Continuous-time state-space model.")
    out = m.run("C = pid(2, 1)")
    assert "  Kp + Ki * ---" in out and "with Kp = 2, Ki = 1" in out
    assert out.rstrip().endswith("Continuous-time PI controller in parallel form.")
    assert "P-only controller." in m.run("P = pid(3)")
    assert "PIDF controller in standard form." in m.run("Q = pidstd(2, 3, 0.5, 10)")
    assert "Frequency(rad/s)" in m.run("F = frd([1 2], [1 10])")
