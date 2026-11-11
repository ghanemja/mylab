function [nDays, total] = merge_daily_files(files, outfile)
%MERGE_DAILY_FILES Combine date,amount CSV files, summing per date.
dates = {};
amounts = [];
for f = 1:numel(files)
    fid = fopen(files{f}, 'r');
    if fid < 0
        error('merge_daily_files:open', 'Cannot open %s', files{f});
    end
    fgetl(fid);   % header
    line = fgetl(fid);
    while ischar(line)
        line = strtrim(line);
        if ~isempty(line)
            comma = find(line == ',', 1);
            dates{end + 1} = strtrim(line(1:comma - 1)); %#ok<AGROW>
            amounts(end + 1) = str2double(line(comma + 1:end)); %#ok<AGROW>
        end
        line = fgetl(fid);
    end
    fclose(fid);
end
total = sum(amounts);
if isempty(amounts)
    total = 0;
    days = {};
    sums = [];
else
    [days, ~, g] = unique(dates);
    sums = accumarray(g(:), amounts(:));
end
nDays = numel(days);
fid = fopen(outfile, 'w');
fprintf(fid, 'date,amount\n');
for k = 1:nDays
    fprintf(fid, '%s,%.2f\n', days{k}, sums(k));
end
fclose(fid);
end
