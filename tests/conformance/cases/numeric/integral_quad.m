% tol: 1e-6
q1 = integral(@(x) x.^2, 0, 1);
q2 = integral(@sin, 0, pi);
q3 = integral(@(x) exp(-x.^2), -Inf, Inf);
q4 = integral(@(x) 1 ./ x, 1, exp(1));
q5 = quad(@(x) x.^2, 0, 1);
q6 = quadgk(@(x) cos(x), 0, pi / 2);
q7 = integral(@(x) x, 1, 0);
q8 = integral(@(x) ones(size(x)), 0, 0);
q9 = quadl(@(x) x.^3, 0, 2);
q10 = integral(@(x) 1 ./ sqrt(x), 0, 1);
q11 = integral2(@(x, y) x .* y, 0, 1, 0, 2);
q12 = integral(@(x) [x, 2 * x], 0, 1, 'ArrayValued', true);
