"""Pattern objects and the text functions that search with them.

Each line is MATLAB code that must run without an error; the expected values in the
assertions were worked out by hand from the documented behaviour of each function.
"""

import pytest

PATTERNS = [
    'r = extract("abc123def45", digitsPattern); assert(numel(r) == 2 && r(1) == "123" && r(2) == "45");',
    'assert(contains("a1", lettersPattern + digitsPattern)); assert(replace("a1b22", digitsPattern, "#") == "a#b#");',
    'p = "x" + wildcardPattern + "z"; assert(matches("xyyz", p)); assert(isa(p, \'pattern\'));',
    ('assert(isequal(extract("a1 b22 c333", digitsPattern(2)), ["22"; "33"])); '
     'assert(isequal(extract("a1 b22 c333", digitsPattern(2, 3)), ["22"; "333"]));'),
    ('assert(isequal(extract("ab1cd", lettersPattern), ["ab"; "cd"])); '
     'assert(isequal(extract("ab1cd", lettersPattern(1)), ["a"; "b"; "c"; "d"]));'),
    ('assert(isequal(extract("a_1 b2", alphanumericsPattern), ["a"; "1"; "b2"])); '
     'assert(isequal(extract("a  b", whitespacePattern), "  "));'),
    'assert(extract("<a><b>", "<" + wildcardPattern + ">") == ["<a>"; "<b>"]);',
    ('assert(extract("key: value", lettersPattern + ":") == "key:"); assert(isequal(extract("abc",'
     ' characterListPattern("ac")), ["a"; "c"]));'),
    ('assert(isequal(extract("a1b2", characterListPattern("a", "c")), ["a"; "b"])); '
     'assert(matches("color", "colo" + optionalPattern("u") + "r")); assert(matches("colour", '
     '"colo" + optionalPattern("u") + "r"));'),
    ('assert(extract("aaab", asManyOfPattern("a")) == "aaa"); assert(extract("aaab", '
     'asManyOfPattern("a", 1, 2)) == ["aa"; "a"]); assert(extract("abab", asManyOfPattern("ab", '
     '2)) == "abab");'),
    ('assert(extract("xaaay", "x" + asFewOfPattern("a") + "a") == "xa"); assert(extract("xaaay", '
     '"x" + asFewOfPattern("a", 2) + "a") == "xaaa");'),
    ('assert(matches("ABC", caseInsensitivePattern("abc"))); assert(~matches("ABC", "abc")); '
     'assert(matches("aBC", "a" + caseInsensitivePattern("bc"))); assert(~matches("ABC", "a" + '
     'caseInsensitivePattern("bc")));'),
    ('assert(contains("ABC", caseSensitivePattern("B"), \'IgnoreCase\', true)); '
     'assert(~contains("AbC", caseSensitivePattern("B"), \'IgnoreCase\', true));'),
    ('assert(isequal(extract("a1b22", regexpPattern("\\d+")), ["1"; "22"])); assert(matches("abc", '
     'regexpPattern("[a-c]+"))); assert(matches("ABC", regexpPattern("abc", "IgnoreCase", true)));'),
    ('assert(replace("abc abc", textBoundary + "abc", "X") == "X abc"); assert(replace("abc abc", '
     '"abc" + textBoundary, "X") == "abc X"); assert(count("abc", textBoundary("start") + "a") == '
     '1);'),
    ('assert(isequal(extract("a" + newline + "b", lineBoundary("start") + lettersPattern), ["a"; '
     '"b"])); assert(count("a" + newline + "b", lettersPattern + lineBoundary("end")) == 2);'),
    ('assert(isequal(extract("ab12cd", letterBoundary + digitsPattern), "12")); '
     'assert(isequal(extract("ab 12", digitBoundary("start") + digitsPattern(1)), "1"));'),
    'assert(isequal(extract("ab1 cd", alphanumericBoundary("start") + lettersPattern), ["ab"; "cd"]));',
    'assert(isequal(extract("a1b2", lettersPattern + lookAheadBoundary("2")), "b"));',
    ('assert(isequal(extract("$12 15", lookBehindBoundary("$") + digitsPattern), "12")); '
     'assert(isequal(extract("x12 y15", lookBehindBoundary(lettersPattern) + digitsPattern), '
     '["12"; "15"]));'),
    ('assert(isequal(extract("a1 b2", ~lookBehindBoundary("a") + digitsPattern), "2")); '
     'assert(isequal(extract("a1 b2", lettersPattern + ~lookAheadBoundary("1")), "b"));'),
    ('assert(matches("cat", "cat" | "dog")); assert(matches("dog", pattern("cat") | "dog")); '
     'assert(~matches("cow", "cat" | "dog")); assert(isequal(extract("cat dog", "cat" | "dog"), '
     '["cat"; "dog"]));'),
    ('assert(matches("ab12", (lettersPattern | "_") + digitsPattern)); q = digitsPattern + "px"; '
     'assert(extract("12px 5em", q) == "12px");'),
    ('assert(isequal(extract("2024-01-05", digitsPattern(4) + "-" + digitsPattern(2) + "-" + '
     'digitsPattern(2)), "2024-01-05"));'),
    ('assert(isequal(extract("x=1, y=22", namedPattern(lettersPattern, "name") + "=" + '
     'digitsPattern), ["x=1"; "y=22"]));'),
    ('assert(extract("a.b", wildcardPattern(1) + ".") == "a."); assert(extract("<<a>>", "<" + '
     'wildcardPattern("Except", "<") + ">") == "<a>");'),
    ('assert(isequal(split("a1b22c", digitsPattern), ["a"; "b"; "c"])); '
     'assert(isequal(strip("123abc45", digitsPattern), "abc")); assert(strip("123abc45", "left", '
     'digitsPattern) == "abc45");'),
    ('assert(erase("a1b2", digitsPattern) == "ab"); assert(extractBefore("abc123", digitsPattern) '
     '== "abc"); assert(extractAfter("abc123x", digitsPattern) == "x");'),
    ('assert(isequal(extractBetween("a1b22c", lettersPattern(1), lettersPattern(1)), "1") || '
     'isequal(extractBetween("a1b22c", lettersPattern(1), lettersPattern(1)), ["1"; "22"]));'),
    ('assert(insertAfter("a1b2", digitsPattern, "!") == "a1!b2!"); assert(insertBefore("a1b2", '
     'digitsPattern, "#") == "a#1b#2"); assert(replaceBetween("a(1)b", "(", ")", "X") == "a(X)b");'),
    ('assert(startsWith("12ab", digitsPattern(2))); assert(~startsWith("1ab", digitsPattern(2))); '
     'assert(endsWith("ab12", digitsPattern(2))); assert(count("a1b22c333", digitsPattern) == 3);'),
    ('assert(isequal(contains(["a1" "bc"], digitsPattern), [true false])); '
     "assert(isequal(contains({'a1', 'bc'}, digitsPattern), [true false])); assert(contains('a1', "
     'digitsPattern));'),
    'assert(isequal(strfind("a1b22", digitsPattern), [2 4])); assert(isequal(strfind(\'ab12\', lettersPattern), 1));',
    'x = evalc(\'p = digitsPattern\'); assert(contains(x, "pattern") && contains(x, "digitsPattern"));',
    "assert(isequal(size(digitsPattern), [1 1])); assert(isequal(class(lettersPattern), 'pattern'));",
    ('assert(matches("a-b", "a" + characterListPattern("-_") + "b")); assert(matches("a]b", "a" + '
     'characterListPattern("]") + "b")); assert(matches("a^b", "a" + characterListPattern("^") + '
     '"b"));'),
    ('assert(matches("a.b", "a.b")); assert(~matches("axb", "a.b")); assert(matches("a*b", '
     '"a*b")); assert(contains("a(b", "(")); assert(contains("a\\b", "\\"));'),
    ('assert(isequal(extract(["a1"; "b2"], digitsPattern), ["1"; "2"])); '
     'assert(isequal(extract(["a1" "b2"], digitsPattern), ["1" "2"]));'),
    ('assert(isequal(extract("no digits", digitsPattern), strings(0, 1))); '
     "assert(isequal(size(extract('none', digitsPattern)), [0 1]));"),
    'assert(replace("a1", digitsPattern, "") == "a");',
]


@pytest.mark.parametrize("code", PATTERNS)
def test_patterns(m, code):
    m.run(code)


