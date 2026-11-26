% run one benchmark case: name in variable CASE
best = Inf; val = NaN; err = '';
for rep = 1:3
    try
        t0 = tic; val = feval(CASE); el = toc(t0);
        best = min(best, el);
    catch e
        err = e.message; break
    end
    if el > 15, break; end
end
fprintf('BENCH\t%s\t%.4f\t%.10g\t%s\n', CASE, best, val, strrep(err, sprintf('\n'), ' '));
