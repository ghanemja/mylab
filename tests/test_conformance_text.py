"""Regression tests for MATLAB conformance of text, struct, cell and regexp functions.

Each entry is a MATLAB expression that must be true.
"""

import pytest

from mylab.errors import MatlabError

NUM2STR = [
    "isequal(num2str([1.5 2.5]), '1.5         2.5')",
    "isequal(num2str([1 -2 3]), '1 -2  3')",
    "isequal(num2str(1:10), '1   2   3   4   5   6   7   8   9  10')",
    "isequal(num2str([10 200; 3 4]), ['10  200'; ' 3    4'])",
    "isequal(num2str([1 NaN 3]), '1  NaN    3')",
    "isequal(num2str([1.5 NaN Inf]), '1.5         NaN         Inf')",
    "isequal(num2str([100 2.5]), '100           2.5')",
    "isequal(num2str([1e-3 1e3]), '0.001           1000')",
    "isequal(num2str(int8([1 -2 3])), '1 -2  3')",
    "isequal(num2str([true false]), '1  0')",
    "isequal(num2str([0.5; -0.25]), ['  0.5'; '-0.25'])",
    "isequal(num2str(pi, '%8.3f'), '3.142')",
    "isequal(num2str(42, '%5d|'), '42|')",
    "isequal(num2str([1 2 3], '%d '), '1 2 3')",
    "isequal(num2str([1 2; 3 4], '%d'), ['12'; '34'])",
    "isequal(num2str(1e16), '1e+16')",
    "isequal(num2str(1e15), '1000000000000000')",
    "isequal(num2str(pi), '3.1416')",
    "isequal(num2str(pi, 8), '3.1415927')",
    "isequal(num2str([pi 1; 2 3], 4), ['3.142          1'; '    2          3'])",
    "isequal(num2str(123456789.123), '123456789.123')",
    "isequal(num2str(-7), '-7')",
    "isequal(int2str([10 200]), '10  200')",
    "isequal(int2str([1.2 2.7; -3.5 4]), [' 1  3'; '-4  4'])",
    "isequal(int2str(2.5), '3')",
]

SPRINTF = [
    "isequal(sprintf('%d ', 'abc'), '97 98 99 ')",
    "isequal(sprintf('%c-', 'abc'), 'a-b-c-')",
    "isequal(sprintf('%s-', 'abc'), 'abc-')",
    "isequal(sprintf('%s=%d,', 'ab', 5, 'cd', 6), 'ab=5,cd=6,')",
    "isequal(double(sprintf('\\101')), 65)",
    "isequal(double(sprintf('\\7x')), [7 120])",
    "isequal(sprintf('%.3d', 5), '005')",
    "isequal(sprintf('%5.3d|', 7), '  007|')",
    "isequal(sprintf('%#o', 8), '010')",
    "isequal(sprintf('%#x', 255), '0xff')",
    "isequal(sprintf('%ld %lu %li %hd %lld', 1, 2, 3, 4, 5), '1 2 3 4 5')",
    "isequal(sprintf('%lf', 1.5), '1.500000')",
]

BASES = [
    "isequal(dec2bin([1 2 3]), ['01'; '10'; '11'])",
    "isequal(dec2bin(5, 8), '00000101')",
    "isequal(dec2bin(0), '0')",
    "isequal(dec2hex([10 255]), ['0A'; 'FF'])",
    "isequal(dec2hex(255, 4), '00FF')",
    "isequal(bin2dec(['101'; '011']), [5; 3])",
    "isequal(bin2dec('1 0 1'), 5)",
    "isequal(hex2dec({'A', '10'}), [10; 16])",
    "isequal(hex2dec('ff'), 255)",
    "isequal(dec2base(10, 3), '101')",
    "isequal(dec2base(255, 16), 'FF')",
    "isequal(dec2base(5, 2, 6), '000101')",
    "isequal(dec2base(23, 'ab'), 'babbb')",
    "isequal(base2dec('101', 3), 10)",
    "isequal(base2dec({'zz', '10'}, 36), [1295; 36])",
    "isnan(base2dec('12', 2))",
]

TEXT = [
    "isequal(char({'p', 'qr'}, 'stu'), ['p  '; 'qr '; 'stu'])",
    "isequal(char('a', '', 'bcd'), ['a  '; '   '; 'bcd'])",
    "isequal(char({'a'; 'bb'}), ['a '; 'bb'])",
    "isequal(char(65:67), 'ABC')",
    "isequal(strtrim(['  ab '; ' cd  ']), [' ab'; 'cd '])",
    "isequal(strtrim([' a'; 'b ']), [' a'; 'b '])",
    "isequal(strtrim(sprintf('\\t x y \\n')), 'x y')",
    "isequal(strtrim({' a ', 'b  '}), {'a', 'b'})",
    "isequal(deblank(['ab  '; 'c   ']), ['ab'; 'c '])",
    "isequal(strcat(['a '; 'b '], 'x'), ['ax'; 'bx'])",
    "isequal(strcat('a ', 'b ', 'c'), 'abc')",
    "isequal(strcat({'a '}, 'b'), {'a b'})",
    "isequal(str2num(['1 2'; '3 4']), [1 2; 3 4])",
    "isequal(str2num('[1 2 3]*2'), [2 4 6])",
    "isequal(upper(5), 5)",
    "isequal(lower([1 2]), [1 2])",
    "isequal(upper({'ab', 'c'}), {'AB', 'C'})",
    "isequal(strjoin({'x', 'y', 'z'}, {'1', '2'}), 'x1y2z')",
    "isequal(strjoin({'x', 'y', 'z'}, '-'), 'x-y-z')",
    "isequal(mat2str([]), 'zeros(0,0)')",
    "isequal(mat2str(zeros(2, 0)), 'zeros(2,0)')",
    "isequal(mat2str(zeros(0, 3)), 'zeros(0,3)')",
    "isequal(mat2str(int8([1 2 3]), 'class'), 'int8([1 2 3])')",
    "isequal(mat2str(uint16(7), 'class'), 'uint16(7)')",
    "isequal(mat2str(int8([1 2 3])), '[1 2 3]')",
    "isequal(mat2str([true false], 'class'), '[true false]')",
    "isequal(mat2str(pi, 4, 'class'), 'double(3.142)')",
    "isequal(mat2str([1.5 2; 3 4]), '[1.5 2;3 4]')",
    "isequal(strrep('aaa', 'aa', 'b'), 'bb')",
    "isequal(strrep('abcabc', 'bc', 'X'), 'aXaX')",
    "isequal(strrep('abab', 'aba', 'x'), 'xb')",
    "isequal(strrep('abc', '', 'x'), 'abc')",
    "isequal(strrep({'aa', 'ba'}, 'a', 'cc'), {'cccc', 'bcc'})",
    "isequal(size(strfind('abc', 'x')), [0 0])",
    "isequal(size(strfind('abc', '')), [0 0])",
    "isequal(strfind('aaa', 'aa'), [1 2])",
    "isequal(strfind({'abc', 'x'}, 'b'), {2, []})",
    "isequal(isstrprop('1aF g', 'xdigit'), [true true true false false])",
    "isequal(isstrprop('a1 ', 'alpha'), [true false false])",
    "isequal(cast(65, 'char'), 'A')",
    "isequal(class(cast(5, 'like', int16(1))), 'int16')",
    "isStringScalar(\"a\") && ~isStringScalar('a') && ~isStringScalar([\"a\" \"b\"])",
    "isequal(convertStringsToChars(\"abc\"), 'abc') && ischar(convertStringsToChars(\"abc\"))",
    "isequal(convertStringsToChars([\"a\" \"bc\"]), {'a', 'bc'})",
    "isequal(convertStringsToChars(5), 5)",
    "isstring(convertCharsToStrings('abc')) && convertCharsToStrings('abc') == \"abc\"",
]

REGEXP = [
    "isequal(regexp({'a1', 'b22', 'c'}, '\\d+', 'once'), {2, 2, []})",
    "isequal(regexp({'a1', 'b22', 'c'}, '\\d+', 'match', 'once'), {'1', '22', ''})",
    "isequal(regexp({'a1', 'b22'}, '\\d+', 'match'), {{'1'}, {'22'}})",
    "isequal(regexp('a1', {'\\d', '[a-z]'}, 'match'), {{'1'}, {'a'}})",
    "isequal(regexp(sprintf('a\\nb'), 'a.b', 'match', 'once'), sprintf('a\\nb'))",
    "isempty(regexp(sprintf('a\\nb'), 'a.b', 'match', 'once', 'dotexceptnewline'))",
    "isequal(regexp('foo bar', '\\<\\w', 'match'), {'f', 'b'})",
    "isequal(regexp('foo bar', '\\w\\>', 'match'), {'o', 'r'})",
    "isequal(regexp('foo bar', 'o\\>'), 3)",
    "isequal(size(regexp('abc', '', 'match')), [1 0])",
    "isequal(size(regexp('abc', 'x*', 'match')), [1 0])",
    "isequal(numel(regexp('abc', 'x*', 'match', 'emptymatch')), 4)",
    "isequal(size(regexp('abc', '(x)', 'tokens', 'once')), [0 0])",
    "isequal(size(regexp('abc', 'x', 'match', 'once')), [0 0])",
    "isequal(regexp('ab12', '(\\d)(\\d)', 'tokenExtents', 'once'), [3 3; 4 4])",
    "isequal(regexp('ab12', '\\d', 'tokenExtents'), {[3 3], [4 4]})",
    "isequal(regexp('ab12cd345', '([a-z]+)(\\d+)', 'tokenExtents'), {[1 2; 3 4], [5 6; 7 9]})",
    "isequal(regexp('ab12cd345', '([a-z]+)(\\d+)', 'tokens'), {{'ab', '12'}, {'cd', '345'}})",
    "isequal(regexp('abc', 'b', 'split'), {'a', 'c'})",
    "isequal(regexp('abc', '[[:alpha:]]+', 'match'), {'abc'})",
    "isequal(regexp('aa bb', '(\\w)\\1', 'match'), {'aa', 'bb'})",
    "isempty(regexp(sprintf('ab\\n'), 'b$'))",
    "isequal(regexp(sprintf('ab\\ncd'), '^c', 'start', 'lineanchors'), 4)",
    "isequal(regexpi('ABC', 'b', 'match'), {'B'})",
    "isequal(regexprep('abc', '', '-'), 'abc')",
    "isequal(regexprep('abc', 'x*', '-'), 'abc')",
    "isequal(regexprep('abc', {'a', 'b'}, {'1', '2'}), '12c')",
    "isequal(regexprep('abc', {'a', 'b'}, 'Z'), 'ZZc')",
    "isequal(regexprep('abc', '(?<x>b)', '[$<x>]'), 'a[b]c')",
    "isequal(regexprep('a.b', '\\.', '\\\\'), 'a\\b')",
    "isequal(regexprep('hello', '^(.)', '${upper($1)}'), 'Hello')",
    "isequal(regexprep('hello world', '(\\w)(\\w*)', '$2$1'), 'elloh orldw')",
    "isequal(regexprep('abc', '(b)', '<$0>'), 'a<b>c')",
    "isequal(regexprep('ab', '(b)', '\\$1'), 'a$1')",
    "isequal(regexprep('a b', '\\s', '\\n'), sprintf('a\\nb'))",
    "isequal(regexprep('aaa', 'a', 'b', 'once'), 'baa')",
    "isequal(regexprep('Abc', 'a', 'x', 'ignorecase'), 'xbc')",
    "isequal(regexprep({'ab', 'cb'}, 'b', 'x'), {'ax', 'cx'})",
]

STRUCTS = [
    "isequal(getfield(struct('a', struct('b', 5)), 'a', 'b'), 5)",
    "isequal(getfield(setfield(struct('a', struct('b', 5)), 'a', 'b', 7), 'a', 'b'), 7)",
    "isequal(fieldnames(orderfields(struct('b', 1, 'c', 2, 'a', 3))), {'a'; 'b'; 'c'})",
    "isequal(fieldnames(orderfields(struct('b', 1, 'c', 2, 'a', 3), {'c', 'a', 'b'})), {'c'; 'a'; 'b'})",
    "isequal(fieldnames(orderfields(struct('b', 1, 'c', 2, 'a', 3), [3 1 2])), {'a'; 'b'; 'c'})",
    "isequal(fieldnames(orderfields(struct('b', 1, 'a', 3), struct('a', 0, 'b', 0))), {'a'; 'b'})",
    "isequal(getfield(orderfields(struct('b', 1, 'c', 2, 'a', 3), [3 1 2]), 'c'), 2)",
    "isequal(size(struct2cell(struct('a', {1, 2, 3}, 'b', {'x', 'y', 'z'}))), [2 1 3])",
    "isequal(struct2cell(struct('a', 1, 'b', 'q')), {1; 'q'})",
    "isequal(size(cell2struct({1 2; 3 4}, {'a', 'b'}, 2)), [2 1])",
    "isequal(size(cell2struct({1 2; 3 4}, {'a', 'b'}, 1)), [2 1])",
    "isequal(getfield(cell2struct({1; 'two'}, {'a', 'b'}, 1), 'b'), 'two')",
    "isequal(getfield(cell2struct({1, 'two'}, {'a', 'b'}, 2), 'b'), 'two')",
    "isequal(numfields(struct('a', 1, 'b', 2)), 2)",
    "isequal(numfields(5), 0)",
    "isequal(size(struct([])), [0 0]) && isstruct(struct([]))",
]

CELLS = [
    "isequal(class(cellfun(@(v) int8(v), {1, 2})), 'int8')",
    "isequal(cellfun(@(v) int8(v), {1, 2}), int8([1 2]))",
    "isequal(class(arrayfun(@(v) single(v), [1 2])), 'single')",
    "isequal(class(arrayfun(@(v) v > 1, [1 2])), 'logical')",
    "isequal(cellfun(@(c) c(1), {'ab', 'cd'}), 'ac')",
    "isequal(cellfun(@numel, {'ab', 'c'; '', 'def'}), [2 1; 0 3])",
    "isequal(cellfun('prodofsize', {'ab', [1 2; 3 4], []}), [2 4 0])",
    "isequal(cellfun('isclass', {'ab', 1}, 'char'), [true false])",
    "isequal(cellfun('size', {'ab', [1 2; 3 4]}, 2), [2 2])",
    "isequal(cellfun('length', {'ab', [1 2; 3 4], []}), [2 2 0])",
    "isequal(cellfun('isempty', {'ab', []}), [false true])",
    "~isreal({}) && ~isreal(struct()) && ~isreal(@sin) && isreal('a') && isreal(true) && ~isreal(1i)",
    "isequal(num2cell([1 2; 3 4], 1), {[1; 3], [2; 4]})",
    "isequal(num2cell([1 2; 3 4], 2), {[1 2]; [3 4]})",
    "isequal(num2cell([1 2; 3 4]), {1 2; 3 4})",
    "isequal(cell2mat({[1 2], 3; [4 5], 6}), [1 2 3; 4 5 6])",
    "isequal(height([1 2; 3 4; 5 6]), 3) && isequal(width([1 2; 3 4; 5 6]), 2)",
    "anynan([1 NaN]) && ~anynan([1 2]) && allfinite([1 2]) && ~allfinite([1 Inf])",
    "isequal(isoutlier([1 1 1 1 100]), [false false false false true])",
    "isequaln(standardizeMissing([1 -99 3], -99), [1 NaN 3])",
    "isequal(uniquetol([1 1+1e-14 2]), [1 2])",
]


@pytest.mark.parametrize("expr", NUM2STR + SPRINTF + BASES + TEXT + REGEXP + STRUCTS + CELLS)
def test_expression_is_true(m, expr):
    assert bool(m.num(expr).all()), expr


def test_orderfields_returns_permutation(m):
    m.run("[o, p] = orderfields(struct('b', 1, 'c', 2, 'a', 3));")
    assert m.num("p").tolist() == [[3.0], [1.0], [2.0]]
    assert m.num("isequal(fieldnames(o), {'a'; 'b'; 'c'})").all()


def test_regexp_all_outputs_in_order(m):
    m.run("[s, e, te, mt, tk, nm, sp] = regexp('ab12cd', '(?<w>[a-z]+)(?<n>\\d+)');")
    assert m.num("isequal(s, 1) && isequal(e, 4)").all()
    assert m.num("isequal(te, {[1 2; 3 4]})").all()
    assert m.num("isequal(mt, {'ab12'}) && isequal(tk, {{'ab', '12'}})").all()
    assert m.num("isequal(nm.w, 'ab') && isequal(nm.n, '12')").all()
    assert m.num("isequal(sp, {'', 'cd'})").all()


def test_regexp_cell_input_returns_every_output(m):
    m.run("[tok, mat] = regexp({'a1', 'b'}, '(\\d)', 'tokens', 'match');")
    assert m.num("isequal(mat, {{'1'}, cell(1, 0)})").all()
    assert m.num("isequal(tok{1}, {{'1'}})").all()


def test_cell2mat_rejects_mixed_classes(m):
    with pytest.raises(MatlabError):
        m.run("x = cell2mat({1, 'a'});")


def test_cellfun_uniform_output_rejects_mixed_integer_classes(m):
    with pytest.raises(MatlabError):
        m.run("x = cellfun(@(v) v, {int8(1), 'a'});")


def test_lasterr_inside_catch(m):
    m.run("try, error('L:e', 'last'); catch, [msg, id] = lasterr; end")
    assert m.str("msg") == "last"
    assert m.str("id") == "L:e"
    m.run("le = lasterror;")
    assert m.str("le.message") == "last"
    assert m.str("class(le)") == "struct"


def test_invalid_regular_expression_is_a_matlab_error(m):
    with pytest.raises(MatlabError):
        m.run("regexp('abc', '(a');")


def test_map_count_is_uint64(m):
    m.run("mp = containers.Map(); mp('a') = 1;")
    assert m.str("class(mp.Count)") == "uint64"
    assert m.str("class(mp.length())") == "double"
