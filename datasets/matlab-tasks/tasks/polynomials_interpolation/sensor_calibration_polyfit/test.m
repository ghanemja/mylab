[c, r] = calibrate_sensor([100 200 300], [5 10 15], 1);
assert(isequal(size(c), [1 2]) && all(abs(c - [0.05 0]) < 1e-12) && r < 1e-12, 'example');
raw = [102 215 331 452 560 688 803 921]; ref = [0; 10.1; 20.3; 30.9; 40.2; 51.3; 61; 71.5];
[c2, r2] = calibrate_sensor(raw, ref, 2);
assert(isequal(size(c2), [1 3]), 'quadratic has three coefficients');
assert(all(abs((c2 - [-1.7686687710258462e-06 0.08882342640526984 -8.962644621085014]) ./ [-1.7686687710258462e-06 0.08882342640526984 -8.962644621085014]) < 1e-8), 'quadratic calibration');
assert(abs(r2 - 0.10715580217761612) < 1e-9, 'quadratic rmse');
[c1, r1] = calibrate_sensor(raw', ref', 1);
assert(all(abs((c1 - [0.08701509068254189 -8.62818115741379]) ./ [0.08701509068254189 -8.62818115741379]) < 1e-10) && abs(r1 - 0.15426373895387677) < 1e-9, 'linear calibration');
assert(r2 <= r1, 'higher degree never fits worse');
[c3, r3] = calibrate_sensor([1 2 3], [2 0 4], 2);
assert(all(abs(c3 - [3 -11 10]) < 1e-9) && r3 < 1e-9, 'interpolating parabola');
threw = false;
try
    calibrate_sensor([1 2], [3 4], 2);
catch
    threw = true;
end
assert(threw, 'too few points must throw');
