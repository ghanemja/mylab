function r = cfh_counter(cmd)
% Persistent counter: each call adds one; cfh_counter('reset') sets it to zero.
persistent count;
if isempty(count)
    count = 0;
end
if nargin > 0 && strcmp(cmd, 'reset')
    count = 0;
else
    count = count + 1;
end
r = count;
end
