assert(abs(colebrook_secant(100000, 0.0001) / 0.01851386607747164 - 1) < 1e-10, 'Re = 100000, rr = 0.0001');
assert(abs(colebrook_secant(4000, 0) / 0.039907014055634925 - 1) < 1e-10, 'Re = 4000, rr = 0');
assert(abs(colebrook_secant(2500000, 0.002) / 0.023495671176029628 - 1) < 1e-10, 'Re = 2500000, rr = 0.002');
assert(abs(colebrook_secant(50000, 0.05) / 0.07200997690051911 - 1) < 1e-10, 'Re = 50000, rr = 0.05');
L = colebrook_secant([10000; 100000; 1000000], 1e-3);
assert(isequal(size(L), [3 1]), 'output has the size of Re');
assert(all(abs(L ./ [0.03238180636309272; 0.022174535944515076; 0.019943465840476866] - 1) < 1e-10), 'vector of Reynolds numbers');
assert(L(1) > L(2) && L(2) > L(3), 'friction decreases with Re');
Lr = colebrook_secant([2e4 3e4], 0);
assert(isequal(size(Lr), [1 2]), 'row input gives row output');
Rv = [2e4 3e4]; res = 1./sqrt(Lr) + 2*log10(2.51./(Rv.*sqrt(Lr)));
assert(all(abs(res) < 1e-9), 'Colebrook residual is small');
threw = false;
try
    colebrook_secant(2000, 1e-4);
catch
    threw = true;
end
assert(threw, 'laminar Reynolds number must throw');
