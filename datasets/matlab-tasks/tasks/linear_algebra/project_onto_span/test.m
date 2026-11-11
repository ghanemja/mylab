[p, r] = project_onto_span([1; 1; 0], [2 0 5]);
assert(isequal(size(p), [3 1]) && isequal(size(r), [3 1]), 'column outputs');
assert(all(abs(p - [1; 1; 0]) < 1e-14) && all(abs(r - [1; -1; 5]) < 1e-14), 'example');
V = [1  2  3; 0  1  1; 1  0  1; 2  1  3];
[p, r] = project_onto_span(V, [3 -1 2 5]);
assert(all(abs(p - [2.5; 4.440892098500626e-16; 2.499999999999999; 5]) < 1e-12), 'dependent columns');
assert(all(abs(V' * r) < 1e-12), 'residual is orthogonal to every column');
[p2, ~] = project_onto_span(V, p);
assert(all(abs(p2 - p) < 1e-12), 'projection is idempotent');
[p, r] = project_onto_span([1 0; 0 0; 0 1], [4; 7; -2]);
assert(all(abs(p - [4; 0; -2]) < 1e-14) && all(abs(r - [0; 7; 0]) < 1e-14), 'coordinate plane');
[p, r] = project_onto_span(zeros(3, 2), [1 2 3]);
assert(isequal(p, zeros(3, 1)) && isequal(r, [1; 2; 3]), 'zero matrix');
[p, r] = project_onto_span(eye(4), [1; 2; 3; 4]);
assert(all(abs(p - [1; 2; 3; 4]) < 1e-14) && all(abs(r) < 1e-14), 'full space');
threw = false;
try
    project_onto_span(eye(3), [1 2]);
catch
    threw = true;
end
assert(threw, 'size mismatch must throw');
