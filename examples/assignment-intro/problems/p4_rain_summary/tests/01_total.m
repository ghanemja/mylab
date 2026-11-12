% name: Total rainfall
% points: 8
total = rain_summary('rain_week.csv');
assert(abs(total - 24) < 1e-9, 'The total for rain_week.csv should be 24');
