% Only whether an error is raised is compared here; wording and identifiers differ between products.
t = zeros(1, 14);
try, undefined_thing_cfh + 1; catch, t(1) = 1; end
try, x = [1 2 3]; x(7); catch, t(2) = 1; end
try, [1 2 3] * [4 5 6]; catch, t(3) = 1; end
try, s.a = 1; s.b; catch, t(4) = 1; end
try, c = {1}; c{2}; catch, t(5) = 1; end
try, ones(2, 3) \ ones(3, 1); catch, t(6) = 1; end
try, str2func(5); catch, t(7) = 1; end
try, z = zeros(2); z(3, 3); catch, t(8) = 1; end
try, cell2mat(5); catch, t(9) = 1; end
try, x = 1; x{1}; catch, t(10) = 1; end
try, feval('cfh_not_a_function'); catch, t(11) = 1; end
try, a = 1; a.b = 2; catch, t(13) = 1; end
try, q = {1}; q.f = 1; catch, t(14) = 1; end
