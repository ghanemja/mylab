function f = cfh_makeadder(n)
% Returns an anonymous function that captured n.
f = @(x) x + n;
end
