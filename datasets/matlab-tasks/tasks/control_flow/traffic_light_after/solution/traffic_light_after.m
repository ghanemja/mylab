function [state, remaining] = traffic_light_after(state, t)
%TRAFFIC_LIGHT_AFTER State and remaining time of a green/yellow/red cycle.
names = {'green', 'yellow', 'red'};
durations = [25 5 30];
k = find(strcmp(names, state));
if isempty(k)
    error('traffic_light_after:badState', 'Unknown state "%s".', state);
end
t = mod(t, sum(durations));
while t >= durations(k)
    t = t - durations(k);
    k = mod(k, 3) + 1;
end
state = names{k};
remaining = durations(k) - t;
end
