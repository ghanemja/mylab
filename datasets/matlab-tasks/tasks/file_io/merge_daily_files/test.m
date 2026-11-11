nl = char(10);
out = 'mdf_merged_out.csv';
[n, t] = merge_daily_files({'a.csv', 'b.csv'}, out);
assert(n == 2 && abs(t - 13.25) < 1e-12, 'example outputs');
assert(isequal(fileread(out), ['date,amount' nl '2024-05-01,3.00' nl '2024-05-02,10.25' nl]), 'example file');
[n, t] = merge_daily_files({'c.csv', 'a.csv', 'd.csv'}, out);
assert(n == 4 && abs(t - (3 + 10.5 + 1.111 - 7 + 0.5 + 100 - 0.004)) < 1e-9, 'three files');
expected = ['date,amount' nl '2023-12-31,100.00' nl '2024-01-15,1.11' nl ...
            '2024-05-01,-3.50' nl '2024-05-02,10.50' nl];
assert(isequal(fileread(out), expected), 'sorted across years, blank lines skipped, sums rounded on output');
[n, t] = merge_daily_files({'empty.csv'}, out);
assert(n == 0 && t == 0 && isequal(fileread(out), ['date,amount' nl]), 'header-only input');
[n, t] = merge_daily_files({}, out);
assert(n == 0 && t == 0, 'no files');
delete(out);
