[x1, w1] = gauss_legendre(1);
assert(abs(x1) < 1e-15 && abs(w1 - 2) < 1e-14, 'one point');
[x2, w2] = gauss_legendre(2);
assert(isequal(size(x2), [2 1]) && isequal(size(w2), [2 1]), 'column outputs');
assert(all(abs(x2 - [-1; 1]/sqrt(3)) < 1e-14) && all(abs(w2 - 1) < 1e-14), 'two points');
[x, w] = gauss_legendre(3);
assert(all(abs(x - [-0.7745966692414834; 0; 0.7745966692414834]) < 1e-12), 'nodes for n = 3');
assert(all(abs(w - [0.5555555555555557; 0.8888888888888888; 0.5555555555555557]) < 1e-12), 'weights for n = 3');
[x, w] = gauss_legendre(5);
assert(all(abs(x - [-0.906179845938664; -0.5384693101056831; 0; 0.5384693101056831; 0.906179845938664]) < 1e-12), 'nodes for n = 5');
assert(all(abs(w - [0.23692688505618928; 0.4786286704993663; 0.5688888888888887; 0.4786286704993663; 0.23692688505618928]) < 1e-12), 'weights for n = 5');
[x, w] = gauss_legendre(12);
assert(all(abs(x - [-0.9815606342467192; -0.9041172563704748; -0.7699026741943047; -0.5873179542866175; -0.3678314989981802; -0.1252334085114689; 0.1252334085114689; 0.3678314989981802; 0.5873179542866175; 0.7699026741943047; 0.9041172563704748; 0.9815606342467192]) < 1e-12), 'nodes for n = 12');
assert(all(abs(w - [0.04717533638651141; 0.10693932599531907; 0.16007832854334642; 0.20316742672306573; 0.2334925365383546; 0.2491470458134027; 0.2491470458134027; 0.2334925365383546; 0.20316742672306573; 0.16007832854334642; 0.10693932599531907; 0.04717533638651141]) < 1e-12), 'weights for n = 12');
assert(all(diff(x) > 0), 'nodes ascending');
[x, w] = gauss_legendre(5);
assert(abs(sum(w .* x.^9) - 0) < 1e-13 && abs(sum(w .* x.^8) - 2/9) < 1e-13, 'degree 2n-1 is exact');
assert(abs(sum(w .* x.^10) - 2/11) > 1e-6, 'degree 2n is not exact');
assert(abs(sum(w .* exp(x)) - 2.3504023872876028) < 1e-8, 'smooth integrand');
threw = false;
try
    gauss_legendre(0);
catch
    threw = true;
end
assert(threw, 'n < 1 must throw');
