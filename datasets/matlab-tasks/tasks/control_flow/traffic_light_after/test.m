[s, r] = traffic_light_after('green', 0);
assert(strcmp(s, 'green') && r == 25, 'no time passed');
[s, r] = traffic_light_after('green', 25);
assert(strcmp(s, 'yellow') && r == 5, 'boundary');
[s, r] = traffic_light_after('green', 24.5);
assert(strcmp(s, 'green') && abs(r - 0.5) < 1e-12, 'fractional time');
[s, r] = traffic_light_after('red', 100);
assert(strcmp(s, 'green') && r == 15, 'more than one cycle');
[s, r] = traffic_light_after('yellow', 5);
assert(strcmp(s, 'red') && r == 30, 'yellow to red');
[s, r] = traffic_light_after('red', 59);
assert(strcmp(s, 'yellow') && r == 1, 'almost a full cycle');
[s, r] = traffic_light_after('green', 6e8 + 26);
assert(strcmp(s, 'yellow') && r == 4, 'huge t');
assert(ischar(s), 'char state');
ok = false;
try
    traffic_light_after('blue', 3);
catch err
    ok = strcmp(err.identifier, 'traffic_light_after:badState');
end
assert(ok, 'bad state');
