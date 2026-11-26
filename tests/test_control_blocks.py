"""Control-system models used together with block diagrams.

Block parameters are MATLAB expressions, so the data of a model object can be typed into a
Transfer Fcn or State-Space block directly, and a linearized diagram becomes a model again.
"""

import numpy as np


def test_tf_data_feeds_a_transfer_fcn_block(m):
    m.run(
        "G = tf([2 1], [1 3 2]); [num, den] = tfdata(G, 'v');"
        "new_system('ctl1'); add_block('simulink/Sources/Step', 'ctl1/Step', 'Time', '0');"
        "add_block('simulink/Continuous/Transfer Fcn', 'ctl1/G', 'Numerator', 'num', 'Denominator', 'den');"
        "add_block('simulink/Sinks/Out1', 'ctl1/Out');"
        "add_line('ctl1', 'Step/1', 'G/1'); add_line('ctl1', 'G/1', 'Out/1');"
        "out = sim('ctl1', 'StopTime', '4', 'RelTol', '1e-9', 'AbsTol', '1e-11');"
        "err = max(abs(out.yout - step(G, out.tout)));"
    )
    assert m.num("err") < 1e-7


def test_model_properties_as_block_parameters(m):
    m.run(
        "s = tf('s'); G = 3 / (s^2 + 0.8*s + 4); S = ss(G);"
        "new_system('ctl2'); add_block('simulink/Sources/Step', 'ctl2/Step', 'Time', '0');"
        "add_block('simulink/Continuous/Transfer Fcn', 'ctl2/G', 'Numerator', 'G.num{1}', 'Denominator', 'G.den{1}');"
        "add_block('simulink/Continuous/State-Space', 'ctl2/S', 'A', 'S.A', 'B', 'S.B', 'C', 'S.C', 'D', 'S.D');"
        "add_block('simulink/Sinks/Out1', 'ctl2/O1'); add_block('simulink/Sinks/Out1', 'ctl2/O2');"
        "add_line('ctl2', 'Step/1', 'G/1'); add_line('ctl2', 'Step/1', 'S/1');"
        "add_line('ctl2', 'G/1', 'O1/1'); add_line('ctl2', 'S/1', 'O2/1');"
        "out = sim('ctl2', 'StopTime', '5', 'RelTol', '1e-9', 'AbsTol', '1e-11');"
        "ref = step(G, out.tout); e1 = max(abs(out.yout(:, 1) - ref)); e2 = max(abs(out.yout(:, 2) - ref));"
    )
    assert m.num("e1") < 1e-7 and m.num("e2") < 1e-7


def test_linearized_diagram_becomes_a_model(m):
    m.run(
        "new_system('ctl3'); add_block('simulink/Sources/In1', 'ctl3/u');"
        "add_block('simulink/Math Operations/Sum', 'ctl3/Sum', 'Inputs', '+-');"
        "add_block('simulink/Continuous/Transfer Fcn', 'ctl3/P', 'Numerator', '[5]', 'Denominator', '[1 2 0]');"
        "add_block('simulink/Sinks/Out1', 'ctl3/y');"
        "add_line('ctl3', 'u/1', 'Sum/1'); add_line('ctl3', 'Sum/1', 'P/1');"
        "add_line('ctl3', 'P/1', 'Sum/2'); add_line('ctl3', 'P/1', 'y/1');"
        "[A, B, C, D] = linmod('ctl3'); T = ss(A, B, C, D); R = feedback(tf(5, [1 2 0]), 1);"
        "e = abs(evalfr(T, 1.3i) - evalfr(R, 1.3i)); p = sort(pole(T)); q = sort(pole(R));"
    )
    assert m.num("e") < 1e-6
    assert np.allclose(m.num("p"), m.num("q"), atol=1e-6)


def test_tuned_pid_gains_drive_a_pid_block(m):
    m.run(
        "G = tf(1, [1 3 3 1]); C = pidtune(G, 'PI'); T = feedback(C * G, 1);"
        "new_system('ctl4'); add_block('simulink/Sources/Step', 'ctl4/r', 'Time', '0');"
        "add_block('simulink/Math Operations/Sum', 'ctl4/e', 'Inputs', '+-');"
        "add_block('simulink/Continuous/PID Controller', 'ctl4/C', 'P', 'C.Kp', 'I', 'C.Ki', 'D', '0');"
        "add_block('simulink/Continuous/Transfer Fcn', 'ctl4/G', 'Numerator', 'G.num{1}', 'Denominator', 'G.den{1}');"
        "add_block('simulink/Sinks/Out1', 'ctl4/y');"
        "add_line('ctl4', 'r/1', 'e/1'); add_line('ctl4', 'e/1', 'C/1'); add_line('ctl4', 'C/1', 'G/1');"
        "add_line('ctl4', 'G/1', 'e/2'); add_line('ctl4', 'G/1', 'y/1');"
        "out = sim('ctl4', 'StopTime', '20', 'RelTol', '1e-8', 'AbsTol', '1e-10');"
        "err = max(abs(out.yout - step(T, out.tout)));"
    )
    assert m.num("err") < 1e-5
