function q = adaptive_simpson(f, a, b, tol)
%ADAPTIVE_SIMPSON Adaptive Simpson quadrature of F over [A, B].
if a == b
    q = 0;
    return
end
if a > b
    q = -adaptive_simpson(f, b, a, tol);
    return
end
fa = f(a);
fb = f(b);
m = (a + b) / 2;
fm = f(m);
S = (b - a) / 6 * (fa + 4 * fm + fb);
q = refine(f, a, b, fa, fm, fb, S, tol, 50);
end

function q = refine(f, a, b, fa, fm, fb, S, tol, depth)
m = (a + b) / 2;
lm = (a + m) / 2;
rm = (m + b) / 2;
flm = f(lm);
frm = f(rm);
SL = (m - a) / 6 * (fa + 4 * flm + fm);
SR = (b - m) / 6 * (fm + 4 * frm + fb);
S2 = SL + SR;
if depth <= 0
    q = S2;
elseif abs(S2 - S) <= 15 * tol
    q = S2 + (S2 - S) / 15;
else
    q = refine(f, a, m, fa, flm, fm, SL, tol / 2, depth - 1) + ...
        refine(f, m, b, fm, frm, fb, SR, tol / 2, depth - 1);
end
end
