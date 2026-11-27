% mex_demo.m - build a small C function with mex and call it like any other function.
%
% Run this script from the examples/mex folder. It needs a C compiler (gcc, clang or cc);
% see docs/MEX.md for how MEX files work in mylab.

mex -silent colstats.c          % writes colstats.<mexext> into the current folder
fprintf('built %s\n', which('colstats'));

X = [1 2 3; 4 5 6; 7 8 10];
[mu, sd] = colstats(X);
fprintf('column means:     %s\n', mat2str(mu, 4));
fprintf('column std devs:  %s\n', mat2str(sd, 4));

% the C code agrees with the built-in functions
assert(max(abs(mu - mean(X))) < 1e-12 && max(abs(sd - std(X))) < 1e-12);

% errors raised in C with mexErrMsgIdAndTxt are ordinary errors here
try
    colstats(int8(X));
catch err
    fprintf('caught %s: %s\n', err.identifier, err.message);
end
