yq = pchip_eval([0 1 2 3], [0 0 1 1], [0.5 1.5 2.5]);
assert(all(abs(yq - [0 0.5 1]) < 1e-14), 'example');
yq = pchip_eval([0 1 2 3 4.5 6 7], [0 0.2 2 2.1 2.1 5 4], [0 0.5 1.3 2.5 3.2 4 5.1 6.6 7]);
assert(isequal(size(yq), [1 9]) && all(abs(yq - [0 0.055000000000000014 0.6297831578947369 2.0736842105263156 2.1 2.1 3.120799999999999 4.664960000000001 4]) < 1e-12), 'mixed data with a flat stretch');
yq = pchip_eval([1; 2; 5; 6], [10; 9; 3; 2.9], [1; 1.5; 2; 2.5; 3; 3.5; 4; 4.5; 5; 5.5; 6]);
assert(isequal(size(yq), [11 1]) && all(abs(yq - [10; 9.564144736842106; 9; 8.128453317201048; 6.919822544867917; 5.588384754990925; 4.348417019560395; 3.414196410566646; 3; 2.929310344827586; 2.8999999999999995]) < 1e-12), 'decreasing data');
assert(all(diff(yq) <= 1e-14), 'monotone data stay monotone');
yq = pchip_eval([0 1 1.2 4], [0 1 0.5 0], [0.1 0.5 1.1 2 3.5]);
assert(all(abs(yq - [0.271 0.875 0.7611166007905135 0.21937911245808314 0.009524913861648693]) < 1e-12), 'end slopes with a sign change');
f = pchip_eval([0 1 2 3 4], [1 1 1 5 5], linspace(0, 2, 9));
assert(all(abs(f - 1) < 1e-14), 'no overshoot on a flat stretch');
assert(all(abs(pchip_eval([2 4], [1 5], [2 3 4]) - [1 3 5]) < 1e-14), 'two points give a line');
assert(all(isnan(pchip_eval([0 1 2], [0 1 4], [-1 3]))), 'outside the range');
threw = false;
try
    pchip_eval([0 0 1], [1 2 3], 0.5);
catch
    threw = true;
end
assert(threw, 'repeated x must throw');
