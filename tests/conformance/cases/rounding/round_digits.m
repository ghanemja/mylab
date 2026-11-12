% round(x, n) rounds to n decimal digits (negative n: left of the point).
try, d1 = round(3.14159, 2); catch, end
try, d2 = round(1234.5678, -2); catch, end
try, d5 = round([1.25 1.35; -1.25 -1.35], 1); catch, end
try, d6 = round(1234.5678, 0); catch, end
try, d7 = round(0.000123456, 3, 'significant'); catch, end
try, d8 = round(123456, 2, 'significant'); catch, end
try, d9 = round(15, -1); catch, end
try, d10 = round(25, -1); catch, end
