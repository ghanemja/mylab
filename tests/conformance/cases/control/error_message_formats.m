try, error('abc %d', 5); catch e, m1 = e.message; end
try, error('abc %d'); catch e, m2 = e.message; end
try, error('rate 100%'); catch e, m3 = e.message; end
try, error('a:b', 'rate %d%%', 50); catch e, m4 = e.message; end
try, error('with\ttab'); catch e, m5 = double(e.message); end
try, error('a:b', 'with\ttab'); catch e, m6 = double(e.message); end
try, error('%s and %s', 'x', 'y'); catch e, m7 = e.message; end
try, error('vals: %d %d %d', [1 2 3]); catch e, m8 = e.message; end
try, error('not:an id with space', 5); catch e, m9 = e.message; id9 = e.identifier; end
try, error('a:b', '%s', 'only text %d'); catch e, m10 = e.message; end
try, error('single word'); catch e, id11 = e.identifier; end
try, error('Odd:id', 'pi=%.3f', pi); catch e, m12 = e.message; id12 = e.identifier; end
try, error(struct('message', 'S msg', 'identifier', 'S:id')); catch e, m13 = e.message; id13 = e.identifier; end
try, error('trailing newline\n'); catch e, m14 = double(e.message); end
try, error('x:y', 'fmt newline\n'); catch e, m15 = double(e.message); end
