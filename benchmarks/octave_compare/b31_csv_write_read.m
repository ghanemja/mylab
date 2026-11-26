function r = b31_csv_write_read()
M = reshape(mod((1:300000) * 7919, 1009), 100000, 3) / 7;
fn = [tempname '.csv'];
dlmwrite(fn, M, 'precision', 10);
N = dlmread(fn);
delete(fn);
r = N(end);
end
