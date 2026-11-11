function t = fall_time(h, m, c)
%FALL_TIME Time to fall height h with quadratic drag, solved with fzero.
g = 9.81;
if c == 0
    t = sqrt(2*h/g);
    return
end
k = sqrt(g*c/m);
logcosh = @(z) z + log((1 + exp(-2*z)) / 2);
d = @(t) (m/c) * logcosh(k*t);
% Upper bound: free fall is fastest, terminal speed is slowest.
tmax = sqrt(2*h/g);          % free fall is faster, so the root lies beyond this
while d(tmax) < h
    tmax = 2*tmax;
end
t = fzero(@(t) d(t) - h, [0, tmax], optimset('TolX', 1e-14));
end
