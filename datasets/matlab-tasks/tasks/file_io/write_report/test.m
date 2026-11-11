nl = char(10);
f = 'wr_report_out.txt';
t = write_report(f, {'rent', 'electricity'}, [950 61.5]);
assert(abs(t - 1011.5) < 1e-12, 'example total');
expected = ['rent             950.00' nl 'electricity       61.50' nl ...
            repmat('-', 1, 23) nl 'TOTAL           1011.50' nl];
assert(isequal(fileread(f), expected), 'example file');
t = write_report(f, {'a', 'bb'}, [-3.456; 2]);
expected = ['a           -3.46' nl 'bb           2.00' nl repmat('-', 1, 17) nl 'TOTAL       -1.46' nl];
assert(abs(t + 1.456) < 1e-12 && isequal(fileread(f), expected), 'TOTAL sets the width, rounding, negatives');
t = write_report(f, {}, []);
assert(t == 0 && isequal(fileread(f), [repmat('-', 1, 17) nl 'TOTAL        0.00' nl]), 'no items');
txt = fileread(f);
lines = strsplit(txt(1:end-1), nl);
assert(numel(lines) == 2 && all(cellfun(@numel, lines) == 17), 'line widths');
delete(f);
