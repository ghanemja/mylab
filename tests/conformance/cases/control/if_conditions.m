r = zeros(1, 12);
if 1, r(1) = 1; end
if 0, r(2) = 1; end
if [1 2 3], r(3) = 1; end
if [1 0 3], r(4) = 1; end
if [], r(5) = 1; end
if 'a', r(6) = 1; end
if '', r(7) = 1; end
if -1, r(8) = 1; end
if 0.001, r(9) = 1; end
if Inf, r(10) = 1; end
if [1 1; 1 1], r(11) = 1; end
if ~isempty([0]), r(12) = 1; end
x = 5;
if x > 10
    b = 'big';
elseif x > 3
    b = 'mid';
elseif x > 4
    b = 'never';
else
    b = 'small';
end
if x == 5 && x ~= 6 || false
    c = 1;
else
    c = 0;
end
d = 0;
if ~(x < 3), d = 1; end
e = 0;
if x, e = 1; end
f = 0;
if zeros(1, 0), f = 1; end
