function [total, wettest_day, dry_days] = rain_summary(filename)
% RAIN_SUMMARY  Total rainfall, wettest day and dry days from a data file.

rain = readmatrix(filename);   % the rainfall values as a row vector

total = sum(rain);
[~, wettest_day] = max(rain);
dry_days = find(rain == 0);

end
