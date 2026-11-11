[r, rho] = correlations([1 2 3 4 5 6], [1 2 3 4 5 6].^3);
assert(abs(rho - 1) < 1e-14 && abs(r - 0.9379313244517968) < 1e-12, 'example');
[r, rho] = correlations([2 5 1 8 6 3 7 4 6 NaN 9], [55; 71; 50; 88; 70; 62; 85; 64; 74; 80; NaN]);
assert(abs(r - 0.9757127498716116) < 1e-12, 'Pearson with missing pairs dropped');
assert(abs(rho - 0.9707197941566387) < 1e-12, 'Spearman with ties and missing pairs');
[r, rho] = correlations([3 1 4 1 5 9 2 6], [2 7 1 8 2 8 1 8]);
assert(abs(r - 0.2096553190730121) < 1e-12 && abs(rho - 0.19885368120992467) < 1e-12, 'many ties in both variables');
[r, rho] = correlations([1 2 3], [3 2 1]);
assert(abs(r + 1) < 1e-14 && abs(rho + 1) < 1e-14, 'perfect negative');
[r, rho] = correlations([1 2 3], [5 5 5]);
assert(isnan(r) && isnan(rho), 'constant variable');
[r, rho] = correlations([1 NaN], [2 3]);
assert(isnan(r) && isnan(rho), 'fewer than two pairs');
threw = false;
try
    correlations([1 2 3], [1 2]);
catch
    threw = true;
end
assert(threw, 'length mismatch must throw');
