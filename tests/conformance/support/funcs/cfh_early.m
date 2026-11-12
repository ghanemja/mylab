function r = cfh_early(x)
% Early return from inside nested control flow.
r = 'start';
for k = 1:10
    if k == x
        r = sprintf('stopped at %d', k);
        return;
    end
end
r = 'finished';
end
