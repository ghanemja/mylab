[Q, R] = mgs_qr([3 1; 4 2]);
assert(max(max(abs(R - [5 2.2; 0 0.4]))) < 1e-14 && max(max(abs(Q - [0.6 -0.8; 0.8 0.6]))) < 1e-14, 'example');
A = [1  2  0; 1  0  3; 1  1  1; 1  -1  2];
[Q, R] = mgs_qr(A);
assert(isequal(size(Q), [4 3]) && isequal(size(R), [3 3]), 'economy sizes');
assert(max(max(abs(R - [2  1  3; 0  2.23606797749979  -1.7888543819998317; 0  0  1.3416407864998734]))) < 1e-13, 'R matches');
assert(max(max(abs(Q - [0.5  0.6708203932499367  -0.22360679774997894; 0.5  -0.223606797749979  0.8198915917499228; 0.5  0.2236067977499789  -0.07453559924999277; 0.5  -0.6708203932499368  -0.5217491947499511]))) < 1e-13, 'Q matches');
assert(isequal(tril(R, -1), zeros(3)) && all(diag(R) > 0), 'R upper triangular with positive diagonal');
t = linspace(0, 1, 12)';
V = [ones(12, 1), t, t.^2, t.^3, t.^4, t.^5, t.^6];
[Qv, Rv] = mgs_qr(V);
assert(max(max(abs(Qv*Rv - V))) < 1e-12, 'reconstruction of an ill-conditioned Vandermonde matrix');
assert(max(max(abs(Qv'*Qv - eye(7)))) < 1e-8, 'orthogonality is kept (modified, not classical, Gram-Schmidt)');
assert(max(max(abs(Rv - [3.4641016151377544  1.7320508075688774  1.2071869264873996  0.9447549859466605  0.7881633652273105  0.6844919044186548  0.6110554137147818; 0  1.087114613009218  1.087114613009218  1.0062548484052267  0.9253950838012351  0.8565640445102344  0.7997617305322237; 0  0  0.3019263638307495  0.4528895457461243  0.5275690891021361  0.5641068131784664  0.5808975213722215; 0  0  0  0.08085976460399147  0.161719529207983  0.2275805128661468  0.27844271557848294; 0  0  0  0  0.020955811984299567  0.05238952996074919  0.08647618131011728; 0  0  0  0  0  0.00522164930155715  0.015664947904672048; 0  0  0  0  0  0  0.0012375995101181184]))) < 1e-9, 'Vandermonde R');
[q1, r1] = mgs_qr([0; -2; 0]);
assert(isequal(q1, [0; -1; 0]) && r1 == 2, 'single column');
threw = false;
try
    mgs_qr([1 2; 2 4; 3 6]);
catch
    threw = true;
end
assert(threw, 'dependent columns must throw');
threw = false;
try
    mgs_qr([1 2 3; 4 5 6]);
catch
    threw = true;
end
assert(threw, 'wide matrix must throw');
