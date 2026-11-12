% name: Wettest day and dry days
% points: 10
[total, wettest_day, dry_days] = rain_summary('rain_week.csv');
assert(abs(total - 24) < 1e-9, 'The total for rain_week.csv should be 24');
assert(isequal(wettest_day, 2), 'The wettest day in rain_week.csv is day 2');
assert(isequal(dry_days, [1 4 5]), 'The dry days in rain_week.csv are [1 4 5]');
