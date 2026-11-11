function n = bounce_count(h0, r, hmin)
%BOUNCE_COUNT Number of rebounds that reach at least HMIN.
n = 0;
h = h0 * r;
while h >= hmin
    n = n + 1;
    h = h * r;
end
end
