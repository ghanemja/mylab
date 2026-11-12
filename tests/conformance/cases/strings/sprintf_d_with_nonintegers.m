% MATLAB switches %d to an exponent format when the value is not an integer.
d1 = sprintf('%d', 1.5);
d2 = sprintf('%d', pi);
d3 = sprintf('%d', 0.1);
d4 = sprintf('%i', -2.5);
d5 = sprintf('%d', 1e20);
d6 = sprintf('%d %d', 1, 2.5);
d7 = sprintf('%x', 2.5);
d8 = sprintf('%5d|', 1.5);
d9 = sprintf('%u', -1);
d10 = sprintf('%d', 3.0);
d12 = sprintf('%d', 1e15);
d13 = sprintf('%d', 123456789.5);
