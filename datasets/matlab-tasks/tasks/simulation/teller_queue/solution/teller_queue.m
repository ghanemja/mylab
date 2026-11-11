function [wait, finish, maxQueue] = teller_queue(arrive, service)
%TELLER_QUEUE FIFO single-server queue: waits, finish times, peak queue length.
arrive = reshape(arrive, 1, []);
service = reshape(service, 1, []);
n = numel(arrive);
wait = zeros(1, n);
finish = zeros(1, n);
startT = zeros(1, n);
free = -Inf;
for i = 1:n
    startT(i) = max(arrive(i), free);
    wait(i) = startT(i) - arrive(i);
    finish(i) = startT(i) + service(i);
    free = finish(i);
end
% sweep over waiting intervals [arrive, start); ends sorted before starts at ties
m = wait > 0;
times = [startT(m), arrive(m)];
delta = [-ones(1, nnz(m)), ones(1, nnz(m))];
[~, order] = sortrows([times(:), delta(:)]);
level = 0;
maxQueue = 0;
for k = order'
    level = level + delta(k);
    maxQueue = max(maxQueue, level);
end
end
