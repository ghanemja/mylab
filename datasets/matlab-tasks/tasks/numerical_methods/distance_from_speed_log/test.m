assert(isequal(cumulative_distance([0 1 3], [2 4 4]), [0 3 11]), 'example');
d = cumulative_distance([0 0.5 1.2 2 3.5 4], [0 2 4.5 6 6.2 5]);
assert(isequal(size(d), [1 6]), 'row in, row out');
assert(all(abs(d - [0 0.5 2.775 6.975 16.125 18.925]) < 1e-12), 'irregular samples');
dc = cumulative_distance([0; 0.5; 1.2; 2; 3.5; 4], [0; 2; 4.5; 6; 6.2; 5]);
assert(isequal(size(dc), [6 1]) && all(abs(dc - [0; 0.5; 2.775; 6.975; 16.125; 18.925]) < 1e-12), 'column in, column out');
dm = cumulative_distance([0; 0.5; 1.2; 2; 3.5; 4], [0 2 4.5 6 6.2 5]);
assert(isequal(size(dm), [1 6]) && all(abs(dm - [0 0.5 2.775 6.975 16.125 18.925]) < 1e-12), 'orientation follows v');
assert(cumulative_distance(5, 3) == 0, 'single sample');
dn = cumulative_distance([0 2 4], [1 -1 -1]);
assert(all(abs(dn - [0 0 -2]) < 1e-12), 'negative speed reduces distance');
threw = false;
try
    cumulative_distance([0 1 2], [1 2]);
catch
    threw = true;
end
assert(threw, 'size mismatch must throw');
