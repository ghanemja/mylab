function [gap, p, q] = largest_prime_gap(n)
%LARGEST_PRIME_GAP Largest gap between consecutive primes up to N.
pr = primes(n);
if numel(pr) < 2
    gap = 0;
    p = NaN;
    q = NaN;
    return
end
[gap, k] = max(diff(pr));
p = pr(k);
q = pr(k + 1);
end
