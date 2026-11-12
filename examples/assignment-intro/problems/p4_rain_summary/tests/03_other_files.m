% name: Other files: ties and no dry days
% points: 12
% hidden: true
[total, wettest_day, dry_days] = rain_summary('rain_twelve.csv');
assert(abs(total - 43.5) < 1e-9);
assert(isequal(wettest_day, 3));
assert(isequal(dry_days, [2 6 7]));
assert(isequal(size(dry_days), [1 3]));

[total, wettest_day, dry_days] = rain_summary('rain_wet.csv');
assert(abs(total - 8) < 1e-9);
assert(isequal(wettest_day, 4));
assert(isempty(dry_days));
