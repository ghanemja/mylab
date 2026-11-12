% String arrays do not exist in Octave; every expectation is in overrides.json.
s = "hello";
try, c1 = class(s); catch, end
try, z1 = size(s); catch, end
try, n1 = strlength(s); catch, end
try, t1 = isstring(s); catch, end
try, t2 = ischar(s); catch, end
try, t3 = isstring('hello'); catch, end
try, c2 = char(s); catch, end
try, z2 = size(["a", "bc", "def"]); catch, end
try, n2 = strlength(["a", "bc", "def"]); catch, end
try, j1 = char("abc" + "def"); catch, end
try, j2 = char("n=" + 5); catch, end
try, j3 = char(strcat("a", "b")); catch, end
try, u1 = char(upper("abc")); catch, end
try, k1 = class(upper("abc")); catch, end
try, e1 = "abc" == "abc"; catch, end
try, e2 = "abc" == 'abc'; catch, end
try, e3 = ["a", "b"] == "a"; catch, end
try, d1 = double("3.5"); catch, end
try, d2 = double("abc"); catch, end
try, x1 = char(string(42)); catch, end
try, x2 = char(string(3.5)); catch, end
try, x3 = char(string(true)); catch, end
try, x4 = cellstr(["a", "bc"]); catch, end
try, x5 = char(string('text')); catch, end
try, x6 = size(string({'a', 'b'; 'c', 'd'})); catch, end
try, l1 = numel("hello"); catch, end
try, l2 = length("hello"); catch, end
try, r1 = char(replace("hello world", "o", "0")); catch, end
try, r2 = char(strrep("hello", "l", "L")); catch, end
try, q1 = contains("hello world", "wor"); catch, end
try, q2 = contains(["apple", "pear"], "pp"); catch, end
try, q3 = startsWith("hello", "he"); catch, end
try, q4 = endsWith("hello", "lo"); catch, end
try, p1 = cellstr(split("a,b,c", ",")); catch, end
try, zp1 = size(split("a,b,c", ",")); catch, end
try, p2 = char(join(["a", "b", "c"], "-")); catch, end
try, p3 = char(strtrim("  x  ")); catch, end
try, p4 = char(strip("  x  ")); catch, end
try, p5 = char(extractBefore("hello world", " ")); catch, end
try, p6 = char(extractAfter("hello world", " ")); catch, end
try, p7 = char(reverse("abc")); catch, end
try, p8 = char(lower("ABC")); catch, end
try, p9 = char(erase("hello world", "o")); catch, end
try, p10 = str2double("2.5"); catch, end
try, p11 = char(sprintf("%d items", 3)); catch, end
try, k2 = class(sprintf("%d items", 3)); catch, end
try, p12 = char(pad("ab", 5)); catch, end
try, p13 = strcmp("abc", 'abc'); catch, end
try, p14 = isempty(""); catch, end
try, p15 = strlength(""); catch, end
try, p16 = isempty(strings(0)); catch, end
try, z3 = size(strings(2, 3)); catch, end
try, p17 = char(num2str(5) + "x"); catch, end
try, p18 = ismissing(string(missing)); catch, end
try, p19 = char(compose("%d-%d", 1, 2)); catch, end
