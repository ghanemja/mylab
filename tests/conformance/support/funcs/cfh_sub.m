function r = cfh_sub(x)
% Calls two local functions defined later in this file.
r = local_double(x) + local_inc(x);
end

function y = local_double(x)
y = 2 * x;
end

function y = local_inc(x)
y = x + 1;
end
