function p = fit_decay(t, y)
%FIT_DECAY Least-squares fit of y = A*exp(-k*t) with fminsearch.
t = t(:);
y = y(:);
c = polyfit(t, log(y), 1);
p0 = [exp(c(2)), -c(1)];
sse = @(q) sum((q(1)*exp(-q(2)*t) - y).^2);
opts = optimset('TolX', 1e-10, 'TolFun', 1e-12, 'MaxFunEvals', 1e4, 'MaxIter', 1e4);
p = fminsearch(sse, p0, opts);
p = reshape(p, 1, 2);
end
