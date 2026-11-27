"""Text functions on char vectors, cell arrays of char vectors and string arrays.

Each line is MATLAB code that must run without an error; the expected values in the
assertions were worked out by hand from the documented behaviour of each function.
"""

import pytest

MODERN_TEXT = [
    'assert(isequal(join(["a" "b" "c"]), "a b c"));',
    'assert(isequal(join(["a" "b"; "c" "d"]), ["a b"; "c d"]));',
    'assert(isequal(join(["a" "b"; "c" "d"], "-", 1), ["a-c" "b-d"]));',
    "assert(isequal(join({'a', 'b', 'c'}, ', '), {'a, b, c'}));",
    'assert(isequal(join(["a" "b" "c"], ["-" "+"]), "a-b+c"));',
    'assert(isequal(split("a,b;c", [",", ";"]), ["a"; "b"; "c"]));',
    "assert(isequal(split('a b c'), {'a'; 'b'; 'c'}));",
    'assert(isequal(split(["a b"; "c d"]), ["a" "b"; "c" "d"]));',
    'assert(isequal(split("a  b"), ["a"; "b"]));',  # consecutive delimiters count as one by default
    'assert(isequal(split("a  b", " ", "CollapseDelimiters", false), ["a"; ""; "b"]));',
    'assert(isequal(split("a,,b", ","), ["a"; "b"]));',
    '[parts, delims] = split("a,b;c", [",", ";"]); assert(isequal(delims, [","; ";"]));',
    'assert(isequal(split("a-b-c", "-", 2), ["a" "b" "c"]));',
    'assert(isequal(splitlines("a" + newline + "b"), ["a"; "b"]));',
    "assert(isequal(splitlines(sprintf('a\\r\\nb\\nc')), {'a'; 'b'; 'c'}));",
    'assert(isequal(compose("%d-%s", 5, "x"), "5-x"));',
    'assert(isequal(compose("%d", [1 2; 3 4]), ["1" "2"; "3" "4"]));',
    "assert(isequal(compose('%d', [1; 2]), {'1'; '2'}));",
    'assert(isequal(compose("%s is %d", ["a"; "b"], [1; 2]), ["a is 1"; "b is 2"]));',
    'assert(isequal(compose("a\\tb"), "a" + sprintf(\'\\t\') + "b"));',
    'assert(isequal(compose(["%d" "%5.2f"], 1.5), ["1.500000e+00" " 1.50"]));',
    'assert(count("banana", "an") == 2);',
    'assert(isequal(count(["aa" "b"], "a"), [2 0]));',
    'assert(count("Banana", "b", "IgnoreCase", true) == 1);',
    'assert(count("a1b22", digitsPattern) == 2);',
    "assert(count('abcabc', {'a', 'bc'}) == 4);",
    "assert(strcmp(strjoin({'a', 'b', 'c'}), 'a b c'));",
    "assert(strcmp(strjoin({'a', 'b', 'c'}, '\\n'), sprintf('a\\nb\\nc')));",
    'assert(strjoin(["x" "y"], ", ") == "x, y" && isstring(strjoin(["x" "y"], ", ")));',
    "assert(strcmp(strjoin({'a', 'b', 'c'}, {'1', '2'}), 'a1b2c'));",
    "assert(isequal(strsplit('a,b,,c', ',', 'CollapseDelimiters', false), {'a', 'b', '', 'c'}));",
    "assert(isequal(strsplit('a  b'), {'a', 'b'}));",
    "assert(isequal(strsplit('a1b22c', {'1', '22'}), {'a', 'b', 'c'}));",
    "[c, matches] = strsplit('a1b2c', {'1', '2'}); assert(isequal(matches, {'1', '2'}));",
    'assert(isequal(strsplit("a b"), ["a" "b"]));',
    "assert(isequal(strsplit('a.b', '.'), {'a', 'b'}));",
    "assert(isequal(strsplit('a\\tb', '\\t'), {'a\\tb'}) || isequal(strsplit(sprintf('a\\tb'), '\\t'), {'a', 'b'}));",
    "assert(isequal(strsplit('a|b', '\\|', 'DelimiterType', 'RegularExpression'), {'a', 'b'}));",
    "assert(strcmp(strcat('a ', 'b'), 'ab'));",
    'assert(isequal(strcat("a ", "b"), "a b"));',
    "assert(isequal(strcat({'a '}, 'b'), {'a b'}));",
    'assert(isequal(strcat(["a" "b"], "c"), ["ac" "bc"]));',
    "assert(isequal(strcat('a', {'b', 'c'}), {'ab', 'ac'}));",
    'assert(isequal(strcat("a", \'b\', "c"), "abc"));',
    'assert(isequal(append("a", "b", "c"), "abc"));',
    "assert(isequal(append('a ', 'b'), 'a b'));",
    'assert(isequal(append(["a" "b"], "c"), ["ac" "bc"]));',
    "assert(isequal(append({'a', 'b'}, 'c'), {'ac', 'bc'}));",
    ('assert("a" + 1 == "a1"); assert(1.5 + "a" == "1.5a"); assert("x" + true == "xtrue"); '
     'assert("a" + \'bc\' == "abc");'),
    'assert(isequal("a" + [1 2], ["a1" "a2"])); assert(ismissing("a" + missing));',
    'assert("v" + 0.1 == "v0.1"); assert("n" + 1e10 == "n10000000000"); assert("p" + pi == "p3.1416");',
    'assert("z" + (-0.5) == "z-0.5"); assert("i" + int8(5) == "i5"); assert("e" + 1e-5 == "e1e-05");',
    'assert(isequal(extract("abc123def45", digitsPattern), ["123"; "45"]));',
    "assert(isequal(extract('abc123def45', digitsPattern), {'123'; '45'}));",
    'assert(extractBefore("a=b", "=") == "a"); assert(extractAfter("a=b", "=") == "b");',
    'assert(extractBefore("hello", 3) == "he"); assert(extractAfter("hello", 3) == "lo");',
    "assert(strcmp(extractBefore('hello', 3), 'he')); assert(strcmp(extractAfter('hello', 'l'), 'lo'));",
    'assert(isequal(extractBefore(["a=b" "c=d"], "="), ["a" "c"]));',
    'assert(ismissing(extractBefore("abc", "x")));',
    'assert(isequal(extractBetween("f(x)+g(y)", "(", ")"), ["x"; "y"]));',
    "assert(isequal(extractBetween('f(x)', '(', ')'), {'x'}));",
    'assert(extractBetween("hello", 2, 4) == "ell");',
    'assert(extractBetween("f(x)", "(", ")", "Boundaries", "inclusive") == "(x)");',
    'assert(isequal(extractBetween(["<a>" "<b>"], "<", ">"), ["a" "b"]));',
    'assert(extractAfter("a1b", digitsPattern) == "b"); assert(extractBefore("a1b", digitsPattern) == "a");',
    'assert(extract("x=12", digitsPattern) == "12");',
    'assert(insertAfter("ac", "a", "b") == "abc"); assert(insertBefore("ac", "c", "b") == "abc");',
    'assert(insertAfter("hello", 2, "XX") == "heXXllo"); assert(insertBefore("hello", 2, "XX") == "hXXello");',
    "assert(strcmp(insertAfter('aXbX', 'X', '-'), 'aX-bX-'));",
    'assert(replaceBetween("f(x)", "(", ")", "y") == "f(y)");',
    'assert(replaceBetween("hello", 2, 4, "EY") == "hEYo");',
    'assert(replaceBetween("f(x)", "(", ")", "y", "Boundaries", "inclusive") == "fy");',
    'assert(eraseBetween("f(x)", "(", ")") == "f()");',
    'assert(eraseBetween("hello", 2, 4) == "ho");',
    'assert(eraseBetween("f(x)", "(", ")", "Boundaries", "inclusive") == "f");',
    "assert(strcmp(eraseBetween('a[b]c', '[', ']'), 'a[]c'));",
    'assert(strip("  a  ") == "a"); assert(strip("  a  ", "left") == "a  "); assert(strip("  a  ", "right") == "  a");',
    ('assert(strip("xxaxx", "x") == "a"); assert(strip("xxaxx", "left", "x") == "axx"); '
     'assert(strip("0012300", "both", "0") == "123");'),
    "assert(strcmp(strip('  a  '), 'a')); assert(isequal(strip({' a ', 'b '}), {'a', 'b'}));",
    'assert(pad("a", 3) == "a  "); assert(pad("a", 3, "left") == "  a"); assert(pad("a", 4, "both") == " a  ");',
    'assert(pad("a", 3, "right", "*") == "a**"); assert(isequal(pad(["a" "bbb"]), ["a  " "bbb"]));',
    "assert(isequal(pad({'a', 'bbb'}), {'a  ', 'bbb'})); assert(strcmp(pad('ab', 5, 'left', '0'), '000ab'));",
    'assert(pad("abc", 2) == "abc"); assert(isequal(pad(["a" "bb"], "left"), [" a" "bb"]));',
    'assert(contains("hello", ["x" "ell"])); assert(~contains("hello", ["x" "y"]));',
    ("assert(contains('Hello', 'hell', 'IgnoreCase', true)); assert(isequal(contains({'ab', 'cd'},"
     " 'a'), [true false]));"),
    'assert(isequal(contains(["ab"; "cd"], ["a" "d"]), [true; true]));',
    'assert(matches("abc", "abc")); assert(~matches("abc", "ab")); assert(matches("abc", ["x" "abc"]));',
    'assert(matches("ABC", "abc", "IgnoreCase", true)); assert(isequal(matches(["a" "b"], "a"), [true false]));',
    ('assert(matches("abc123", lettersPattern + digitsPattern)); assert(matches(\'xyyz\', "x" + '
     'wildcardPattern + "z"));'),
    'assert(startsWith("hello", ["x" "he"])); assert(startsWith("Hello", "he", "IgnoreCase", true));',
    'assert(endsWith("hello", "lo")); assert(endsWith("hello", ["x" "LO"], "IgnoreCase", true));',
    ('assert(startsWith("123abc", digitsPattern)); assert(endsWith("abc123", digitsPattern)); '
     'assert(~startsWith("abc", digitsPattern));'),
    "assert(isequal(startsWith({'ab', 'cd'}, 'a'), [true false]));",
    'assert(~contains(string(missing), "a"));',
    'assert(isequal(strfind("abcabc", "bc"), [2 5])); assert(isequal(strfind(\'abcabc\', \'bc\'), [2 5]));',
    ('assert(isequal(strfind({\'abc\', \'bcb\'}, \'b\'), {2, [1 3]})); assert(isequal(strfind(["abc" '
     '"bcb"], "b"), {2, [1 3]}));'),
    "assert(isequal(strfind('aaa', 'aa'), [1 2])); assert(isempty(strfind('abc', '')));",
    'assert(strrep("hello", "l", "L") == "heLLo"); assert(isequal(strrep(["ab" "cb"], "b", "x"), ["ax" "cx"]));',
    "assert(isequal(strrep({'ab', 'cb'}, 'b', 'x'), {'ax', 'cx'})); assert(strcmp(strrep('abc', 'b', ''), 'ac'));",
    'assert(isequal(strfind("a1b22", digitsPattern), [2 4]));',
    'assert(replace("abc", "b", "x") == "axc"); assert(replace("abc", ["a" "c"], ["1" "2"]) == "1b2");',
    'assert(replace("abc", ["a" "c"], "-") == "-b-"); assert(strcmp(replace(\'abc\', \'b\', \'x\'), \'axc\'));',
    ('assert(isequal(replace({\'ab\', \'cb\'}, \'b\', \'x\'), {\'ax\', \'cx\'})); assert(replace("a1b22", '
     'digitsPattern, "#") == "a#b#");'),
    ('assert(erase("abc", "b") == "ac"); assert(erase("abcabc", ["a" "c"]) == "bb"); '
     "assert(strcmp(erase('abc', 'b'), 'ac'));"),
    'assert(erase("a1b22", digitsPattern) == "ab");',
    ('assert(reverse("abc") == "cba"); assert(isequal(reverse(["ab" "cd"]), ["ba" "dc"])); '
     "assert(isequal(reverse({'ab'}), {'ba'}));"),
    ('assert(strlength("abc") == 3); assert(isequal(strlength(["a" "bb"]), [1 2])); '
     "assert(strlength('abc') == 3); assert(isequal(strlength({'a', 'bb'}), [1 2]));"),
    ('assert(upper("abc") == "ABC"); assert(isequal(lower(["AB" "Cd"]), ["ab" "cd"])); '
     "assert(isequal(upper({'ab'}), {'AB'}));"),
    ("assert(isequal(newline, char(10))); assert(strcmp(blanks(3), '   ')); "
     'assert(isStringScalar("a") && ~isStringScalar(\'a\') && ~isStringScalar(["a" "b"]));'),
    ('assert(strtrim("  a ") == "a"); assert(isequal(strtrim(["  a " " b"]), ["a" "b"])); '
     "assert(isequal(strtrim({' a ', 'b '}), {'a', 'b'}));"),
    'assert(isequal(deblank("a  "), "a")); assert(isequal(deblank({\'a  \', \' b \'}), {\'a\', \' b\'}));',
    ("assert(isequal(strjust(['a  '; 'bb '], 'right'), ['  a'; ' bb'])); assert(isequal(strjust(['"
     "  a'; ' bb'], 'left'), ['a  '; 'bb '])); assert(isequal(strjust('a   ', 'center'), ' a  '));"),
    ('assert(strncmp("hello", "help", 3)); assert(strncmpi("HELlo", "help", 3)); '
     'assert(isequal(strcmp(["a" "b"], "a"), [true false])); assert(isequal(strcmpi(["A" "b"], '
     '"a"), [true false]));'),
    ('assert(strcmp("abc", \'abc\')); assert(~strcmp("abc", "abd")); assert(isequal(strncmp({\'abc\', '
     "'abd', 'xbc'}, 'ab', 2), [true true false]));"),
    ("assert(isequal(validatestring('app', {'apple', 'pear'}), 'apple')); "
     'assert(isequal(validatestring("pe", ["apple" "pear"]), \'pear\'));'),
    ('assert(isequal(convertStringsToChars("abc"), \'abc\')); '
     'assert(isequal(convertStringsToChars(["a" "b"]), {\'a\', \'b\'})); '
     'assert(isequal(convertCharsToStrings(\'abc\'), "abc")); '
     'assert(isequal(convertCharsToStrings({\'a\', \'b\'}), ["a" "b"]));'),
    ("assert(isequal(isstrprop('a1 B', 'alpha'), [true false false true])); "
     "assert(isequal(isstrprop('a1 B', 'alphanum'), [true true false true])); "
     "assert(isequal(isstrprop('a1 B', 'wspace'), [false false true false]));"),
    ("assert(isequal(isstrprop('a1 B', 'upper'), [false false false true])); "
     "assert(isequal(isstrprop('a1 B', 'lower'), [true false false false])); "
     "assert(isequal(isstrprop('a.1', 'punct'), [false true false])); "
     "assert(isequal(isstrprop('aF9g', 'xdigit'), [true true true false]));"),
    ('assert(isequal(isstrprop("a1", \'digit\'), [false true])); assert(isequal(isstrprop({\'a1\', '
     "'2'}, 'digit'), {[false true], true}));"),
    ("assert(isequal(isstrprop(sprintf('a\\tb'), 'cntrl'), [false true false])); "
     "assert(isequal(isstrprop('a b', 'graphic'), [true false true])); assert(isequal(isstrprop('a"
     " b', 'print'), [true true true]));"),
    ("assert(isequal(double('aé€'), [97 233 8364])); assert(isequal(unicode2native('aé', 'UTF-8'),"
     " uint8([97 195 169]))); assert(strcmp(native2unicode(uint8([97 195 169]), 'UTF-8'), 'aé'));"),
    ("assert(isequal(unicode2native('a€', 'UTF-16LE'), uint8([97 0 172 32]))); "
     "assert(isequal(unicode2native('é', 'ISO-8859-1'), uint8(233))); "
     "assert(strcmp(native2unicode(uint8(233), 'latin1'), 'é'));"),
    ("assert(strcmp(upper('straße'), 'STRASSE') || strcmp(upper('straße'), 'STRAßE')); "
     "assert(strcmp(lower('ÀÉ'), 'àé')); assert(length('日本語') == 3); assert(strcmp(fliplr('日本'), "
     "'本日'));"),
]


REGEXP_AND_CONVERSIONS = [
    ("[tok, mat] = regexp('ab12cd345', '(\\d+)', 'tokens', 'match'); assert(isequal(mat, {'12', "
     "'345'}) && strcmp(tok{2}{1}, '345'));"),
    "n = regexp('key=val', '(?<k>\\w+)=(?<v>\\w+)', 'names'); assert(strcmp(n.k, 'key') && strcmp(n.v, 'val'));",
    "n = regexp('a=1,b=2', '(?<k>\\w)=(?<v>\\d)', 'names'); assert(isequal(size(n), [1 2]) && strcmp(n(2).k, 'b'));",
    "assert(strcmp(regexp('abc', 'b.', 'match', 'once'), 'bc')); assert(isequal(regexp('a1b2', '\\d'), [2 4]));",
    "assert(isequal(regexp('a,b;c', '[,;]', 'split'), {'a', 'b', 'c'}));",
    "te = regexp('ab12', '(\\w)(\\d)', 'tokenExtents'); assert(isequal(te{1}, [2 2; 3 3]));",
    "te = regexp('ab12', '(\\w)(\\d)', 'tokenExtents', 'once'); assert(isequal(te, [2 2; 3 3]));",
    "assert(strcmp(regexp('price: $42', '(?<=\\$)\\d+', 'match', 'once'), '42'));",
    ("assert(strcmp(regexp('foobar', 'foo(?=bar)', 'match', 'once'), 'foo')); "
     "assert(isempty(regexp('foobaz', 'foo(?=bar)', 'match', 'once')));"),
    ("assert(strcmp(regexp('foobaz', 'foo(?!bar)', 'match', 'once'), 'foo')); "
     "assert(isempty(regexp('xbar', '(?<!x)bar', 'match')));"),
    ("assert(isequal(regexp(sprintf('ab\\ncd'), '^\\w', 'match', 'lineanchors'), {'a', 'c'})); "
     "assert(isequal(regexp(sprintf('ab\\ncd'), '^\\w', 'match'), {'a'}));"),
    "assert(isequal(regexp(sprintf('ab\\ncd'), '\\w$', 'match', 'lineanchors'), {'b', 'd'}));",
    ("assert(isequal(regexp('abc', 'a b c', 'match', 'freespacing'), {'abc'})); "
     "assert(isempty(regexp('abc', 'a b c', 'match')));"),
    ("assert(isequal(regexp('abc', 'x*', 'match', 'emptymatch'), {'', '', '', ''})); "
     "assert(isempty(regexp('abc', 'x*', 'match')));"),
    ("assert(isequal(regexp('ABC', 'b', 'match', 'ignorecase'), {'B'})); "
     "assert(isequal(regexpi('ABC', 'b', 'match'), {'B'}));"),
    ("assert(isequal(regexp(sprintf('a\\nb'), 'a.b', 'match'), {sprintf('a\\nb')})); "
     "assert(isempty(regexp(sprintf('a\\nb'), 'a.b', 'match', 'dotexceptnewline')));"),
    ("[s, e, te, m, t, nm, sp] = regexp('x1y2', '(\\d)'); assert(isequal(s, [2 4]) && isequal(e, [2"
     " 4]) && isequal(m, {'1', '2'}) && isequal(sp, {'x', 'y', ''}));"),
    ("assert(isequal(regexp({'a1', 'b22'}, '\\d+', 'match'), {{'1'}, {'22'}})); "
     "assert(isequal(regexp({'a1', 'b22'}, '\\d+', 'match', 'once'), {'1', '22'}));"),
    ('assert(isequal(regexp("a1b22", "\\d+", \'match\'), ["1" "22"])); assert(isequal(regexp("a1b22",'
     ' "\\d+", \'match\', \'once\'), "1"));'),
    'assert(isequal(regexp(["a1" "b22"], "\\d+", \'match\'), {"1", "22"}));',
    ('assert(isequal(regexp("a,b", ",", \'split\'), ["a" "b"])); t = regexp("k=v", "(\\w)=(\\w)", '
     '\'tokens\'); assert(isequal(t{1}, ["k" "v"]));'),
    "assert(isequal(regexp('aXbXc', 'X', 'split', 'once'), {'a', 'bXc'}));",
    ("assert(strcmp(regexprep('abc', '(a)(b)', '$2$1'), 'bac')); assert(strcmp(regexprep('aaa', "
     "'a', 'b', 'once'), 'baa'));"),
    "assert(strcmp(regexprep('john smith', '(\\w)(\\w*)', '${upper($1)}$2'), 'John Smith'));",
    ("assert(strcmp(regexprep('abc', 'B', 'x', 'ignorecase'), 'axc')); "
     "assert(strcmp(regexprep('Abc ABC', 'abc', 'xyz', 'preservecase'), 'Xyz XYZ'));"),
    ("assert(isequal(regexprep({'ab', 'cb'}, 'b', 'x'), {'ax', 'cx'})); "
     'assert(isequal(regexprep(["ab" "cb"], "b", "x"), ["ax" "cx"]));'),
    ("assert(strcmp(regexprep('abc', {'a', 'b'}, {'1', '2'}), '12c')); "
     "assert(strcmp(regexprep('a.b', '\\.', '\\\\n'), 'a\\nb') || strcmp(regexprep('a.b', '\\.', "
     "'\\\\n'), sprintf('a\\\\nb')));"),
    "assert(strcmp(regexprep('hello world', '(?<a>\\w+) (?<b>\\w+)', '$<b> $<a>'), 'world hello'));",
    "assert(strcmp(regexprep('abc', '(?<x>b)', '[$<x>]'), 'a[b]c'));",
    "assert(strcmp(regexprep('abc', '', 'x'), 'abc'));",
    ("assert(strcmp(regexprep('abcd', '(?<=b)c', 'X'), 'abXd')); "
     "assert(strcmp(regexprep(sprintf('a\\nb'), '^b', 'X', 'lineanchors'), sprintf('a\\nX')));"),
    ("assert(strcmp(regexptranslate('escape', 'a.b*c'), 'a\\.b\\*c')); "
     "assert(strcmp(regexptranslate('wildcard', '*.m'), '.*\\.m'));"),
    ("assert(strcmp(regexp('ab12', '[[:alpha:]]+', 'match', 'once'), 'ab')); "
     "assert(strcmp(regexp('ab12', '[[:digit:]]+', 'match', 'once'), '12'));"),
    ("assert(strcmp(regexp('hello world', '\\<w\\w*', 'match', 'once'), 'world')); "
     "assert(strcmp(regexp('aXb', '(?i)x', 'match', 'once'), 'X'));"),
    ("assert(isequal(regexp('abab', '(ab)\\1', 'match'), {'abab'})); assert(isequal(regexp('aa bb',"
     " '(?<c>\\w)\\k<c>', 'match'), {'aa', 'bb'}));"),
    "tk = regexp('ab', '(a)|(b)', 'tokens'); assert(isequal(tk{1}, {'a'}) || isequal(tk{1}, {'a', ''}));",
    "assert(isequal(regexp('one two', '(\\w+)\\s(\\w+)', 'tokens', 'once'), {'one', 'two'}));",
    '[tok, rest] = strtok("one two"); assert(tok == "one" && rest == " two");',
    ("assert(strcmp(regexprep('abc', '(?<ch>b)', '${upper($<ch>)}'), 'aBc') || "
     "strcmp(regexprep('abc', '(b)', '${upper($1)}'), 'aBc'));"),
    "assert(strcmp(regexprep('3 apples', '(\\d+)', '${num2str(str2double($1) * 2)}'), '6 apples'));",
    "assert(strcmp(regexp('5XXXXX', '(\\d+)(??X{$1})', 'match', 'once'), '5XXXXX'));",
    ('assert(string(12) == "12"); assert(string(1.5) == "1.5"); assert(string(true) == "true"); '
     'assert(string(pi) == "3.1416");'),
    ('assert(isequal(string([1 2; 3 4]), ["1" "2"; "3" "4"])); assert(isequal(string({\'a\', \'b\'}), '
     '["a" "b"])); assert(isequal(string([\'ab\'; \'cd\']), ["ab"; "cd"]));'),
    ('assert(ismissing(string(NaN))); assert(isequal(string(int8(-5)), "-5")); assert(string(1e10)'
     ' == "10000000000"); assert(string(1.23456789) == "1.2346");'),
    ('assert(string(1e-5) == "1e-05"); assert(string(123456789) == "123456789"); assert(string(0.1'
     ' + 0.2) == "0.3"); assert(string(1 + 2i) == "1+2i");'),
    ('assert(isequal(string({1, \'a\', "b"}), ["1" "a" "b"])); assert(string(\'\') == ""); '
     'assert(isequal(size(string({})), [0 0]));'),
    ('assert(isequal(char("abc"), \'abc\')); assert(isequal(char(["a" "bcd"]), [\'a  \'; \'bcd\'])); '
     "assert(isequal(char({'a', 'bcd'}), ['a  '; 'bcd'])); assert(isequal(char(65:67), 'ABC'));"),
    ('assert(isequal(char(\'a\', "bc", {\'d\'}), [\'a \'; \'bc\'; \'d \']) || isequal(size(char(\'a\', "bc", '
     "{'d'})), [3 2]));"),
    ('assert(isequal(cellstr("abc"), {\'abc\'})); assert(isequal(cellstr(["a" "b"]), {\'a\', \'b\'})); '
     "assert(isequal(cellstr(['ab '; 'cd ']), {'ab'; 'cd'})); assert(isequal(cellstr('abc'), "
     "{'abc'}));"),
    "assert(isequal(cellstr({'a', 'b'}), {'a', 'b'})); assert(isequal(cellstr(''), {''}));",
    ('assert(double("3.5") == 3.5); assert(isnan(double("abc"))); assert(isequal(double(["1" '
     '"2.5"]), [1 2.5])); assert(isequal(double(\'ab\'), [97 98])); '
     'assert(isnan(double(string(missing))));'),
    ('assert(double("1e3") == 1000); assert(double("-.5") == -0.5); assert(double("Inf") == Inf); '
     'assert(double("1,000") == 1000 || isnan(double("1,000")));'),
    ('assert(str2double("1e3") == 1000); assert(isequaln(str2double(["1" "x"]), [1 NaN])); '
     "assert(isequaln(str2double({'2', 'y'}), [2 NaN])); assert(str2double('1,000') == 1000);"),
    ("assert(str2double('1+2i') == 1 + 2i); assert(isnan(str2double(5))); assert(str2double('  12"
     "  ') == 12); assert(str2double('1e-3') == 0.001); assert(isnan(str2double('')));"),
    ("assert(str2double('.5') == 0.5); assert(str2double('-Inf') == -Inf); "
     "assert(isnan(str2double('1 2'))); assert(isnan(str2double(string(missing)))); "
     "assert(str2double('1d3') == 1000);"),
    ('assert(isequal(ismissing(["a" missing ""]), [false true false])); '
     "assert(isequal(ismissing({'a', ''}), [false true])); assert(isequal(ismissing([1 NaN]), "
     '[false true]));'),
    ('assert(isequal(ismissing(["a" "b" "NA"], "NA"), [false false true])); '
     'assert(isequal(ismissing([1 -99 3], -99), [false true false]));'),
    ('s = standardizeMissing(["a" "N/A" "b"], "N/A"); assert(ismissing(s(2)) && s(1) == "a"); '
     'assert(isequaln(standardizeMissing([1 -99 3], -99), [1 NaN 3]));'),
    "assert(isequal(standardizeMissing({'a', 'NA'}, 'NA'), {'a', ''}));",
    ('m = string(missing); assert(ismissing(m)); assert(isequal(size(m), [1 1])); x = ["a" '
     'missing]; assert(ismissing(x(2))); assert(~(m == m)); assert(ismissing(missing));'),
    ('assert(isequal(fillmissing(["a" missing "c"], \'constant\', "z"), ["a" "z" "c"])); '
     'assert(isequal(rmmissing(["a" missing "c"]), ["a" "c"]));'),
    ('assert(isequal(sort(["b" "a" "C"]), ["C" "a" "b"])); [s, i] = sort(["b"; "a"]); '
     'assert(isequal(i, [2; 1])); assert(isequal(sort(["b" "a"], \'descend\'), ["b" "a"]));'),
    ('assert(isequal(unique(["b" "a" "b"]), ["a" "b"])); [u, i, j] = unique(["b" "a" "b"]); '
     "assert(isequal(i(:)', [2 1]) && isequal(j(:)', [2 1 2]));"),
    ('assert(isequal(unique(["b" "a" "b"], \'stable\'), ["b" "a"])); assert(isequal(unique({\'b\', '
     "'a', 'b'}), {'a', 'b'}));"),
    ('assert(ismember("a", ["a" "b"])); [tf, loc] = ismember(["a" "z"], ["b" "a"]); '
     'assert(isequal(tf, [true false]) && isequal(loc, [2 0]));'),
    ("assert(isequal(ismember({'a', 'z'}, {'b', 'a'}), [true false])); assert(ismember('a', {'a', "
     '\'b\'})); assert(ismember("a", {\'a\', \'b\'}));'),
    ('assert(isequal(intersect(["a" "b"], ["b" "c"]), "b")); assert(isequal(union(["b" "a"], ["c" '
     '"a"]), ["a" "b" "c"])); assert(isequal(setdiff(["a" "b" "c"], "b"), ["a" "c"]));'),
    ('assert(isequal(setxor(["a" "b"], ["b" "c"]), ["a" "c"])); assert(isequal(intersect({\'a\', '
     "'b'}, {'b', 'c'}), {'b'})); assert(isequal(union({'b'}, {'a'}), {'a', 'b'}));"),
    'assert(isequal(sort({\'b\', \'a\'}), {\'a\', \'b\'})); assert(issorted(["a" "b"]));',
    ('assert("abc" < "abd"); assert("b" > "a"); assert(isequal(["a" "b"] == ["a" "c"], [true '
     'false])); assert("a" ~= "b");'),
    ('assert(isequal(sort(["b10" "b9" "a"]), ["a" "b10" "b9"])); assert(isequal(fliplr(["a" "b"]),'
     ' ["b" "a"])); assert(isequal(numel(["a" "b"]), 2)); assert(isequal(size(["a"; "b"]), [2 '
     '1]));'),
    ('x = ["a" "b"]; x(4) = "d"; assert(ismissing(x(3))); x(2) = []; assert(numel(x) == 3); '
     # x(3) is a missing string: equal to itself for isequaln only
     'assert(isequaln([x "e"], [x, "e"])); assert(~isequal([x "e"], [x, "e"])); '
     'assert(isequal(["a" \'b\' {\'c\'}], ["a" "b" "c"]));'),
    ('assert(isequal(["a" 1], ["a" "1"])); assert(isequal([string(1) "b"], ["1" "b"])); '
     'assert(isequal(repmat("ab", 1, 2), ["ab" "ab"])); assert(isequal(strings(2, 1), [""; ""]));'),
    ("assert(isequal(cellfun(@numel, {'ab', 'c'}), [2 1])); assert(isequal(arrayfun(@strlength, "
     '["ab" "c"]), [2 1])); y = "abc"; assert(isequal(y{1}(2:3), \'bc\')); z = ["ab" "cd"]; '
     "assert(isequal(z{2}, 'cd'));"),
    ('assert(contains(lower("ABC"), "b")); assert(isequal(num2cell(["a" "b"]), {"a", "b"})); '
     'assert(isequal(horzcat("a", "b"), ["a" "b"])); assert(isequal(vertcat("a", "b"), ["a"; '
     '"b"]));'),
    ('assert(isequal(isspace("a b"), [false true false])); assert(isequal(isletter("a1"), [true '
     'false])); assert(isstring("a") && ~isstring(\'a\') && ischar(\'a\') && ~ischar("a")); '
     'assert(iscellstr({\'a\'}) && ~iscellstr({"a"}));'),
    ('assert(isequal(strtrim(sprintf(\' a\\t\')), \'a\')); assert(isequal(isempty(""), false)); '
     'assert(isempty(strings(0))); assert(strlength("") == 0); assert(isequal("" == "", true));'),
]


FORMATTING_AND_SCANNING = [
    ("assert(strcmp(sprintf('%bx', 1), '3ff0000000000000')); assert(strcmp(sprintf('%tx', "
     "single(1)), '3f800000')); assert(strcmp(sprintf('%lu', 5), '5')); "
     "assert(strcmp(sprintf('%ld', -5), '-5'));"),
    ("assert(strcmp(sprintf('%*d', 5, 3), '    3')); assert(strcmp(sprintf('%-*d|', 4, 3), '3   "
     "|')); assert(strcmp(sprintf('%.*f', 2, pi), '3.14')); assert(strcmp(sprintf('%*.*f', 8, 3, "
     "pi), '   3.142'));"),
    ("assert(strcmp(sprintf('%d %d\\n', [1 2; 3 4]), sprintf('1 3\\n2 4\\n'))); "
     "assert(strcmp(sprintf('%d,', 1:5), '1,2,3,4,5,')); assert(strcmp(sprintf('%d-%d ', 1, 2, 3),"
     " '1-2 3-'));"),
    ("assert(strcmp(sprintf('%s=%d;', 'a', 1, 'b', 2), 'a=1;b=2;')); assert(strcmp(sprintf('%s', "
     "'abc', 'def'), 'abcdef')); assert(strcmp(sprintf('%c', 'abc'), 'abc')); "
     "assert(strcmp(sprintf('%5.1s|', 'abc'), '    a|'));"),
    ("assert(strcmp(sprintf('a\\tb'), ['a' char(9) 'b'])); assert(strcmp(sprintf('%s', 'a\\tb'), "
     "'a\\tb')); assert(strcmp(sprintf('a\\\\b'), 'a\\b')); assert(strcmp(sprintf('100%%'), '100%')); "
     "assert(strcmp(sprintf('%%d'), '%d'));"),
    ("assert(strcmp(sprintf('\\x41\\101'), 'AA')); assert(strcmp(sprintf('%d', []), '')); "
     "assert(strcmp(sprintf('x%dy', []), 'xy')); assert(strcmp(sprintf('%s', ''), '')); "
     "assert(isempty(sprintf('')));"),
    ("assert(strcmp(sprintf('%d', 1.5), '1.500000e+00')); assert(strcmp(sprintf('%i', 3), '3')); "
     "assert(strcmp(sprintf('%u', 3), '3')); assert(strcmp(sprintf('%x', 255), 'ff')); "
     "assert(strcmp(sprintf('%X', 255), 'FF')); assert(strcmp(sprintf('%o', 8), '10'));"),
    ("assert(strcmp(sprintf('%#x', 255), '0xff')); assert(strcmp(sprintf('%#o', 8), '010')); "
     "assert(strcmp(sprintf('%+d', 5), '+5')); assert(strcmp(sprintf('% d', 5), ' 5')); "
     "assert(strcmp(sprintf('%05d', 42), '00042')); assert(strcmp(sprintf('%-5d|', 42), '42   "
     "|'));"),
    ("assert(strcmp(sprintf('%e', 12345.678), '1.234568e+04')); assert(strcmp(sprintf('%E', "
     "0.00012), '1.200000E-04')); assert(strcmp(sprintf('%g', 1e-5), '1e-05')); "
     "assert(strcmp(sprintf('%g', 100000), '100000')); assert(strcmp(sprintf('%g', 1e6), "
     "'1e+06'));"),
    ("assert(strcmp(sprintf('%G', 1e-10), '1E-10')); assert(strcmp(sprintf('%.3g', pi), '3.14')); "
     "assert(strcmp(sprintf('%10.4f|', pi), '    3.1416|')); assert(strcmp(sprintf('%.0f', 2.5), "
     "'2')); assert(strcmp(sprintf('%.0f', 3.5), '4'));"),
    ("assert(strcmp(sprintf('%f', Inf), 'Inf')); assert(strcmp(sprintf('%d', -Inf), '-Inf')); "
     "assert(strcmp(sprintf('%f', NaN), 'NaN')); assert(strcmp(sprintf('%5.1f|', NaN), '  NaN|'));"
     " assert(strcmp(sprintf('%d', NaN), 'NaN'));"),
    ("assert(strcmp(sprintf('%d', true), '1')); assert(strcmp(sprintf('%d', 'a'), '97')); "
     "assert(strcmp(sprintf('%s', 65), 'A')); assert(strcmp(sprintf('%d', int8(-5)), '-5')); "
     "assert(strcmp(sprintf('%d', uint64(18446744073709551615)), '18446744073709551615'));"),
    ('assert(strcmp(sprintf(\'%s\', "str"), \'str\')); assert(sprintf("%d", 3) == "3"); '
     'assert(isstring(sprintf("%s", \'a\'))); assert(strcmp(sprintf(\'%s and %s\', "a", "b"), \'a and '
     'b\')); assert(strcmp(sprintf(\'%d%s\', 1, "ab"), \'1ab\'));'),
    ('assert(strcmp(sprintf(\'%s\', ["a" "bc"]), \'abc\')); assert(strcmp(sprintf(\'%d %s|\', 1, \'ab\', '
     "2, 'cd'), '1 ab|2 cd|')); assert(strcmp(sprintf('%c%c%c', 72, 105, 33), 'Hi!')); "
     "assert(strcmp(sprintf('%5s|%-5s|', 'ab', 'cd'), '   ab|cd   |'));"),
    ("assert(strcmp(sprintf('%d', 1e10), '10000000000')); assert(strcmp(sprintf('%d', 2^53), "
     "'9007199254740992')); assert(strcmp(sprintf('%x', 2^32), '100000000'));"),
    ("assert(strcmp(sprintf('%.10g', 0.1), '0.1')); assert(strcmp(sprintf('%.15g', 0.1 + 0.2), "
     "'0.3')); assert(strcmp(sprintf('%.17g', 0.1), '0.10000000000000001')); "
     "assert(strcmp(sprintf('%5.3d', 7), '  007')); assert(strcmp(sprintf('%3$s %1$s %2$s', 'a', "
     "'b', 'c'), 'c a b'));"),
    "[s, err] = sprintf('%d', 5); assert(strcmp(s, '5') && isempty(err));",
    ("assert(strcmp(num2str(pi), '3.1416')); assert(strcmp(num2str(pi, 8), '3.1415927')); "
     "assert(strcmp(num2str(123456789), '123456789')); assert(strcmp(num2str(1.23456789), "
     "'1.2346')); assert(strcmp(num2str(-0.5), '-0.5'));"),
    ("assert(strcmp(num2str([1 2 3]), '1  2  3')); assert(strcmp(num2str([1.5 2.5]), '1.5         "
     "2.5')); assert(strcmp(num2str(1e10), '10000000000')); assert(strcmp(num2str(1e-5), "
     "'1e-05')); assert(strcmp(num2str(123.456), '123.456'));"),
    ("assert(strcmp(num2str(1234.5678), '1234.5678')); assert(strcmp(num2str(0.000123456), "
     "'0.00012346')); assert(strcmp(num2str(100), '100')); assert(strcmp(num2str(-7), '-7')); "
     "assert(strcmp(num2str(true), '1')); assert(strcmp(num2str('abc'), 'abc'));"),
    ("assert(strcmp(num2str([1 2], '%d'), '12')); assert(strcmp(num2str([1 2], '%d '), '1 2')); "
     "assert(strcmp(num2str(3, '%05d'), '00003')); assert(strcmp(num2str(1+2i), '1+2i')); "
     "assert(strcmp(num2str(1.5-2.5i), '1.5-2.5i'));"),
    ("assert(isequal(num2str([1 2; 3 4]), ['1  2'; '3  4'])); assert(isequal(num2str([1; 10; "
     "100]), ['  1'; ' 10'; '100'])); assert(strcmp(num2str(int8(-5)), '-5')); "
     "assert(strcmp(num2str(NaN), 'NaN')); assert(strcmp(num2str(Inf), 'Inf')); "
     "assert(strcmp(num2str(-Inf), '-Inf'));"),
    ("assert(strcmp(num2str([1 NaN 3]), '1  NaN    3')); assert(strcmp(num2str(123456789012), "
     "'123456789012')); assert(strcmp(num2str(0.1), '0.1')); assert(strcmp(num2str(99999.5), "
     "'99999.5')); assert(strcmp(num2str(1e5), '100000'));"),
    ("assert(strcmp(num2str(3.14159, 4), '3.142')); assert(strcmp(num2str(1234.5678, 2), "
     "'1.2e+03')); assert(strcmp(num2str([0.1 0.22]), '0.1        0.22')); "
     "assert(strcmp(num2str(-[1 2]), '-1 -2')); assert(strcmp(num2str([10 -2]), '10  -2')); "
     'assert(isempty(num2str([])));'),
    ("assert(strcmp(num2str(single(pi)), '3.1416')); assert(strcmp(num2str(uint8(200)), '200')); "
     'assert(strcmp(num2str("str"), \'str\')); assert(strcmp(num2str(1.0), \'1\')); '
     "assert(strcmp(num2str(12345.678), '12345.678')); assert(strcmp(num2str(0.5), '0.5'));"),
    ("assert(strcmp(mat2str([1 2; 3 4]), '[1 2;3 4]')); assert(strcmp(mat2str([0.1 1/3]), '[0.1 "
     "0.333333333333333]')); assert(strcmp(mat2str(pi, 4), '3.142')); assert(strcmp(mat2str([true "
     "false]), '[true false]')); assert(strcmp(mat2str('abc'), '''abc'''));"),
    ("assert(strcmp(mat2str(int8([1 2])), '[1 2]')); assert(strcmp(mat2str(int8([1 2]), 'class'), "
     "'int8([1 2])')); assert(strcmp(mat2str(single(1.5), 'class'), 'single(1.5)')); "
     "assert(strcmp(mat2str([]), 'zeros(0,0)')); assert(strcmp(mat2str(5), '5'));"),
    ("assert(strcmp(mat2str([1+2i 3]), '[1+2i 3+0i]')); assert(strcmp(mat2str([1;2]), '[1;2]')); "
     "assert(strcmp(mat2str([NaN Inf -Inf]), '[NaN Inf -Inf]')); assert(strcmp(mat2str(zeros(2, "
     "0)), 'zeros(2,0)')); assert(strcmp(mat2str(true), 'true')); assert(strcmp(mat2str(1e10), "
     "'10000000000'));"),
    ("assert(strcmp(mat2str(['ab'; 'cd']), '[''ab'';''cd'']')); assert(strcmp(mat2str(1e-5), "
     "'1e-05')); assert(strcmp(mat2str(123456789.123), '123456789.123')); "
     "assert(strcmp(mat2str(-0.5), '-0.5'));"),
    ("assert(strcmp(int2str(2.7), '3')); assert(strcmp(int2str(-2.5), '-3')); "
     "assert(strcmp(int2str([1.2 2.7]), '1  3')); assert(strcmp(int2str(2.5), '3')); "
     "assert(strcmp(int2str(1e10), '10000000000')); assert(isequal(int2str([1.4; 20.6]), [' 1'; "
     "'21']));"),
    ("assert(isequal(sscanf('1 2 3', '%d'), [1; 2; 3])); assert(sscanf('3.14', '%f') == 3.14); "
     "assert(isequal(sscanf('1,2,3', '%d,'), [1; 2; 3])); assert(isequal(sscanf('a1b2', 'a%db%d'),"
     ' [1; 2]));'),
    ("assert(strcmp(sscanf('hello world', '%s'), 'helloworld')); assert(isequal(sscanf('1 2 3 4', "
     "'%d', [2 2]), [1 3; 2 4])); assert(isequal(sscanf('1 2 3 4', '%d', 2), [1; 2]));"),
    ("[a, n] = sscanf('1 2 x', '%d'); assert(isequal(a, [1; 2]) && n == 2); "
     "assert(isequal(sscanf('12abc', '%d%s'), [12; 97; 98; 99])); assert(isequal(sscanf('3 4', '%f"
     " %f'), [3; 4])); assert(isempty(sscanf('abc', '%d')));"),
    ("assert(isequal(sscanf('1.5e3', '%g'), 1500)); assert(isequal(sscanf('ff', '%x'), 255)); "
     "assert(isequal(sscanf('17', '%o'), 15)); assert(strcmp(sscanf('abc', '%c'), 'abc')); "
     "assert(isequal(sscanf('12345', '%2d'), [12; 34; 5])); assert(isequal(sscanf('a b', '%c %c'),"
     " 'ab'));"),
    ("assert(isequal(sscanf('1 2 3', '%*d %d'), [2]) || isequal(sscanf('1 2 3', '%*d %d'), 2)); "
     'assert(isequal(sscanf(\'temp=25.5C\', \'temp=%fC\'), 25.5)); assert(isequal(sscanf("7", "%d"), '
     "7)); assert(isequal(sscanf('-3 +4', '%d'), [-3; 4]));"),
    ("assert(isequal(sscanf('1;2;3', '%d;'), [1; 2; 3])); assert(isequal(sscanf('x=1 y=2', 'x=%d "
     "y=%d')', [1 2])); assert(isequal(sscanf('3 apples', '%d %s')', [3 double('apples')]));"),
    ("c = textscan(sprintf('1,2\\n3,4'), '%f %f', 'Delimiter', ','); assert(isequal(c{1}, [1; 3]) "
     '&& isequal(c{2}, [2; 4]));'),
    "c = textscan('a 1 b 2', '%s %d'); assert(isequal(c{1}, {'a'; 'b'}) && isequal(c{2}, int32([1; 2])));",
    "c = textscan(sprintf('x y\\n1 2\\n3 4'), '%f %f', 'HeaderLines', 1); assert(isequal(c{2}, [2; 4]));",
    ("c = textscan('1 2 3 4', '%f'); assert(isequal(c{1}, [1; 2; 3; 4])); c = textscan('1;2', "
     "'%f', 'Delimiter', ';'); assert(isequal(c{1}, [1; 2]));"),
    ("c = textscan('a,b,c', '%s', 'Delimiter', ','); assert(isequal(c{1}, {'a'; 'b'; 'c'})); c = "
     "textscan('1 2 3 4', '%f %f', 'CollectOutput', true); assert(isequal(c{1}, [1 2; 3 4]));"),
    ("c = textscan('1,NA,3', '%f', 'Delimiter', ',', 'TreatAsEmpty', 'NA'); assert(isequaln(c{1}, "
     "[1; NaN; 3])); c = textscan('1 2 3', '%d %*d %d'); assert(isequal(c{2}, int32(3)));"),
    ('c = textscan("5 6", "%f"); assert(isequal(c{1}, [5; 6])); c = textscan(\'1.5 x\', \'%f %q\'); '
     "assert(isequal(c{2}, {'x'}));"),
    ("c = textscan('3 4', '%n %n'); assert(c{1} == 3 && c{2} == 4); c = textscan('1 2 3 4 5 6', "
     "'%f %f', 2); assert(isequal(c{1}, [1; 3])); c = textscan('7 8', '%u8 %f32'); "
     "assert(isa(c{1}, 'uint8') && isa(c{2}, 'single'));"),
]


@pytest.mark.parametrize("code", MODERN_TEXT)
def test_modern_text(m, code):
    m.run(code)


@pytest.mark.parametrize("code", REGEXP_AND_CONVERSIONS)
def test_regexp_and_conversions(m, code):
    m.run(code)


@pytest.mark.parametrize("code", FORMATTING_AND_SCANNING)
def test_formatting_and_scanning(m, code):
    m.run(code)


