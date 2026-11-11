function s = format_duration(sec)
%FORMAT_DURATION Seconds as M:SS or H:MM:SS.
h = floor(sec / 3600);
m = floor(mod(sec, 3600) / 60);
r = mod(sec, 60);
if h > 0
    s = sprintf('%d:%02d:%02d', h, m, r);
else
    s = sprintf('%d:%02d', m, r);
end
end
