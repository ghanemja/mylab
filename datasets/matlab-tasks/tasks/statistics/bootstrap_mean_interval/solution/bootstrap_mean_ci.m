function [ci, bootmeans] = bootstrap_mean_ci(x, nboot, alpha)
%BOOTSTRAP_MEAN_CI Percentile bootstrap confidence interval of the mean.
x = x(:);
n = numel(x);
bootmeans = zeros(nboot, 1);
for b = 1:nboot
    idx = randi(n, n, 1);
    bootmeans(b) = mean(x(idx));
end
s = sort(bootmeans);
lo = s(max(1, floor(nboot * alpha / 2)));
hi = s(min(nboot, ceil(nboot * (1 - alpha / 2))));
ci = [lo hi];
end
