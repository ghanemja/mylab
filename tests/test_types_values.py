"""Data types: dictionary, containers.Map, datetime and duration, categorical, struct and cell functions, class rules.

Each line is MATLAB code that must run without an error; the expected values in the
assertions were worked out by hand from the documented behaviour of each function.
"""

import pytest

DICTIONARY_AND_MAP = [
    ('d = dictionary(["a", "b"], [1 2]); assert(d("b") == 2); d("c") = 3; assert(numEntries(d) == '
     '3); assert(isKey(d, "a")); k = keys(d); assert(numel(k) == 3); d = remove(d, "a"); '
     'assert(~isKey(d, "a"));'),
    ('d = dictionary; assert(~isConfigured(d)); d("x") = 5; assert(isConfigured(d)); '
     'assert(isequal(types(d), ["string" "double"])); assert(d("x") == 5);'),
    ('d = dictionary("a", 1, "b", 2); assert(isequal(d(["b" "a"]), [2 1])); '
     'assert(isequal(keys(d), ["a"; "b"])); assert(isequal(values(d), [1; 2]));'),
    ('d = dictionary([1 2 3], ["one" "two" "three"]); assert(d(2) == "two"); assert(isequal(d([3 '
     '1]), ["three" "one"])); assert(isequal(isKey(d, [1 5]), [true false]));'),
    ('d = dictionary(["a" "b"], [1 2]); d(["c" "d"]) = [3 4]; assert(numEntries(d) == 4); d(["a" '
     '"b"]) = 0; assert(isequal(d(["a" "b" "c"]), [0 0 3]));'),
    'd = dictionary("a", 1); d2 = d; d2("b") = 2; assert(numEntries(d) == 1 && numEntries(d2) == 2);',
    ('d = dictionary("a", 1); threw = false; try, d("zz"); catch e, threw = true; end; '
     "assert(threw && strcmp(e.identifier, 'MATLAB:dictionary:ScalarKeyNotFound'));"),
    ('d = dictionary("a", 1); assert(lookup(d, "zz", "FallbackValue", -1) == -1); '
     'assert(isequal(lookup(d, ["a" "q"], FallbackValue=0), [1 0]));'),
    'd = dictionary("a", 1); d = insert(d, "b", 2); assert(d("b") == 2);',
    'd = dictionary("a", 1); d("a") = []; assert(numEntries(d) == 0); assert(isConfigured(d));',
    ('d = configureDictionary("string", "double"); assert(isConfigured(d) && numEntries(d) == 0); '
     'd("k") = 2; assert(d("k") == 2); assert(isequal(size(keys(configureDictionary("string", '
     '"double"))), [0 1]));'),
    ('d = dictionary("a", {[1 2 3]}); assert(isequal(d{"a"}, [1 2 3])); d{"b"} = \'text\'; '
     'assert(ischar(d{"b"})); v = d("a"); assert(iscell(v));'),
    ('d = dictionary(["x" "y"], {1, \'two\'}); assert(isequal(values(d), {1; \'two\'})); c = '
     'values(dictionary("a", 5), "cell"); assert(iscell(c) && c{1} == 5);'),
    ('d = dictionary(["a" "b"], [1 2]); e = entries(d); assert(istable(e) && height(e) == 2 && '
     'isequal(e.Value, [1; 2]) && isequal(e.Key, ["a"; "b"]));'),
    ('d = dictionary(["a" "b"], [1 2]); s = entries(d, "struct"); assert(isstruct(s) && numel(s) '
     '== 2 && s(2).Value == 2); c = entries(d, "cell"); assert(iscell(c) && isequal(size(c), [2 '
     '2]));'),
    ('d = dictionary(\'abc\', 1); assert(isKey(d, "abc")); assert(isequal(types(d), ["string" '
     '"double"])); assert(d(\'abc\') == 1);'),
    ('d = dictionary(["a" "b"], ["x" "y"]); assert(d("a") == "x"); d = dictionary("n", \'chars\'); '
     'assert(isstring(d("n")));'),
    ('d = dictionary([true false], ["yes" "no"]); assert(d(true) == "yes"); d = dictionary(int8([1'
     ' 2]), [10 20]); assert(d(int8(2)) == 20); assert(d(2) == 20);'),
    ('d = dictionary("a", 1); assert(isequal(d, dictionary("a", 1))); assert(~isequal(d, '
     'dictionary("a", 2))); assert(isa(d, \'dictionary\') && strcmp(class(d), \'dictionary\'));'),
    ('d = dictionary(["b" "a"], [2 1]); assert(isequal(keys(d), ["b"; "a"])); d("b") = 9; '
     'assert(isequal(keys(d), ["b"; "a"])); assert(isequal(values(d), [9; 1]));'),
    ('s = evalc(\'d = dictionary(["a" "b"], [1 2])\'); assert(contains(s, \'dictionary (string\') && '
     'contains(s, \'with 2 entries\') && contains(s, \'"a"\') && contains(s, \'1\'));'),
    "s = evalc('d = dictionary'); assert(contains(s, 'dictionary with unset key and value types'));",
    ('d = dictionary(datetime(2024, 1, 1), "new year"); assert(d(datetime(2024, 1, 1)) == "new '
     'year"); assert(~isKey(d, datetime(2024, 1, 2)));'),
    ('d = dictionary("a", struct(\'x\', 1)); assert(d("a").x == 1); d = dictionary(1:3, 4:6); '
     'assert(isequal(d.keys, [1; 2; 3]) && d.numEntries == 3);'),
    'd = dictionary("a", 1); n = 0; for k = keys(d)\', n = n + d(k); end; assert(n == 1);',
    ("m = containers.Map(); m('a') = 1; m('b') = 2; assert(m.isKey('a') && isKey(m, 'b')); "
     "assert(m.Count == 2); assert(isequal(keys(m), {'a', 'b'})); assert(isequal(values(m), {1, "
     "2})); assert(isequal(values(m, {'b'}), {2}));"),
    ("m = containers.Map({'x', 'y'}, [10 20]); assert(m('y') == 20); remove(m, 'x'); "
     "assert(m.Count == 1); assert(strcmp(m.KeyType, 'char') && strcmp(m.ValueType, 'double'));"),
    ("m = containers.Map('KeyType', 'double', 'ValueType', 'any'); m(3) = 'q'; m4 = m; m4(4) = 1; "
     'assert(m.Count == 2); assert(isequal(keys(m), {3, 4})); assert(length(m) == 2);'),
    ("m = containers.Map([1 2 3], {'a', 'b', 'c'}); assert(strcmp(m(2), 'b')); "
     "assert(isequal(cell2mat(keys(m)), [1 2 3])); assert(strcmp(m.KeyType, 'double'));"),
    ("m = containers.Map('k', 5); assert(m('k') == 5); m = containers.Map({'a', 'b'}, {1, 'two'});"
     " assert(strcmp(m('b'), 'two')); assert(strcmp(m.ValueType, 'any'));"),
    "m = containers.Map('UniformValues', false); m('k') = {1, 2}; assert(iscell(m('k')));",
    ("m = containers.Map({'b', 'a', 'c'}, [2 1 3]); assert(isequal(keys(m), {'a', 'b', 'c'})); "
     'assert(isequal(values(m), {1, 2, 3}));'),
    ("m = containers.Map('KeyType', 'int32', 'ValueType', 'double'); m(int32(5)) = 1.5; m(7) = 2; "
     'assert(m(5) == 1.5 && m.Count == 2);'),
    ("m = containers.Map('KeyType', 'char', 'ValueType', 'double'); threw = false; try, m('a') = "
     "'text'; catch, threw = true; end; assert(threw);"),
    ('m = containers.Map("s", 1); assert(m("s") == 1 && m(\'s\') == 1); assert(isKey(m, "s")); '
     "assert(isequal(isKey(m, {'s', 'zz'}), [true false]));"),
    ("m = containers.Map({'a', 'b'}, [1 2]); m2 = containers.Map({'c'}, 3); m3 = [m; m2]; "
     "assert(m3.Count == 3 && m3('c') == 3);"),
    ("m = containers.Map({'a', 'b'}, [1 2]); v = values(m, {'b', 'a'}); assert(isequal(v, {2, "
     "1})); threw = false; try, values(m, {'zz'}); catch, threw = true; end; assert(threw);"),
    ("m = containers.Map({'a'}, {1}); assert(isobject(m)); assert(isa(m, 'containers.Map') && "
     "isa(m, 'handle')); assert(strcmp(class(m), 'containers.Map'));"),
    ("m = containers.Map(); assert(m.Count == 0 && isempty(m)); assert(isempty(keys(m))); m('x') ="
     ' []; assert(m.Count == 1);'),
    ("m = containers.Map('KeyType', 'double', 'ValueType', 'double'); for k = 1:5, m(k) = k^2; "
     'end; assert(m(4) == 16); ks = cell2mat(keys(m)); assert(isequal(ks, 1:5)); '
     'assert(isequal(cell2mat(values(m)), [1 4 9 16 25]));'),
    ("m = containers.Map({'a', 'b'}, [1 2]); m.remove({'a', 'b'}); assert(m.Count == 0); threw = "
     "false; try, m.remove('a'); catch e, threw = true; end; assert(threw);"),
    ("m = containers.Map({'one'}, {struct('x', 1)}); assert(m('one').x == 1); m = "
     "containers.Map('KeyType', 'char', 'ValueType', 'any'); m('f') = @sin; f = m('f'); "
     'assert(f(0) == 0);'),
    ("m = containers.Map(1, 'x'); assert(strcmp(m(1), 'x')); m = containers.Map(single(1.5), 2); "
     'assert(m(1.5) == 2); m = containers.Map(true, 1) ;'),
    ("m = containers.Map({'a', 'b'}, {1, 2}); n = 0; for k = keys(m), n = n + m(k{1}); end; "
     "assert(n == 3); c = m.values; assert(isequal(c, {1, 2})); assert(isequal(m.keys, {'a', "
     "'b'}));"),
]


DATES_AND_DURATIONS = [
    ("t = datetime(2024, 1, 1, 12, 0, 0, 'TimeZone', 'America/New_York'); u = t; u.TimeZone = "
     "'Europe/London'; assert(hour(u) == 17); assert(strcmp(u.TimeZone, 'Europe/London'));"),
    ("z = datetime(2024, 7, 1, 12, 0, 0, 'TimeZone', 'UTC'); z.TimeZone = 'Asia/Tokyo'; "
     'assert(hour(z) == 21); assert(day(z) == 1);'),
    ("t = datetime(2024, 7, 1, 12, 0, 0, 'TimeZone', 'America/New_York'); assert(tzoffset(t) == "
     "hours(-4)); assert(isdst(t)); w = datetime(2024, 1, 1, 'TimeZone', 'America/New_York'); "
     'assert(tzoffset(w) == hours(-5) && ~isdst(w));'),
    ("a = datetime(2024, 1, 1, 12, 0, 0, 'TimeZone', 'America/New_York'); b = datetime(2024, 1, 1,"
     " 17, 0, 0, 'TimeZone', 'Europe/London'); assert(a == b); assert(seconds(b - a) == 0); "
     'assert(~(a < b));'),
    ("a = datetime(2024, 3, 9, 12, 0, 0, 'TimeZone', 'America/New_York'); b = a + hours(24); "
     'assert(hour(b) == 13); c = a + caldays(1); assert(hour(c) == 12); assert(hours(c - a) == '
     '23);'),
    ("t = datetime(2024, 1, 1, 0, 0, 0, 'TimeZone', 'UTC'); assert(posixtime(t) == 1704067200); u "
     "= datetime(1704067200, 'ConvertFrom', 'posixtime', 'TimeZone', 'Asia/Tokyo'); assert(hour(u)"
     ' == 9);'),
    ("t = datetime(2024, 1, 1, 12, 0, 0, 'TimeZone', '+05:30'); t.TimeZone = 'UTC'; assert(hour(t)"
     ' == 6 && minute(t) == 30);'),
    ("t = datetime(2024, 6, 1, 12, 0, 0, 'TimeZone', 'Europe/Paris', 'Format', 'yyyy-MM-dd HH:mm "
     "Z'); assert(strcmp(char(t), '2024-06-01 12:00 +0200')); t.Format = 'HH:mm z'; "
     "assert(strcmp(char(t), '12:00 CEST'));"),
    ("t = datetime(2024, 6, 1, 12, 0, 0, 'TimeZone', 'Europe/Paris', 'Format', 'HH:mm xxx'); "
     "assert(strcmp(char(t), '12:00 +02:00')); t.Format = 'HH:mm ZZZZZ'; assert(strcmp(char(t), "
     "'12:00 +02:00'));"),
    ("tz = timezones; assert(istable(tz) && any(strcmp(tz.Name, 'Europe/London'))); tz2 = "
     "timezones('Europe'); assert(all(startsWith(tz2.Name, 'Europe')));"),
    ("threw = false; try, datetime(2024, 1, 1, 'TimeZone', 'UTC') == datetime(2024, 1, 1); catch, "
     'threw = true; end; assert(threw);'),
    ("t = datetime('now', 'TimeZone', 'Asia/Kolkata'); assert(strcmp(t.TimeZone, 'Asia/Kolkata'));"
     " u = datetime('today', 'TimeZone', 'local'); assert(~isempty(u.TimeZone));"),
    "t = datetime(2024, 3, 10, 2, 30, 0, 'TimeZone', 'America/New_York'); assert(hour(t) == 3);",
    ("s = sort([datetime(2024, 1, 1, 12, 0, 0, 'TimeZone', 'Asia/Tokyo'), datetime(2024, 1, 1, 1, "
     "0, 0, 'TimeZone', 'UTC')]); assert(hour(s(1)) == 10 || hour(s(1)) == 1);"),
    ("t = datetime(2024, 3, 5, 14, 7, 9); t.Format = 'yyyy-MM-dd''T''HH:mm:ss'; "
     "assert(strcmp(char(t), '2024-03-05T14:07:09')); t.Format = 'MMM d, y'; "
     "assert(strcmp(char(t), 'Mar 5, 2024'));"),
    ("t = datetime(2024, 3, 5, 14, 7, 9.123); t.Format = 'hh:mm:ss.SSS a'; assert(strcmp(char(t), "
     "'02:07:09.123 PM')); t.Format = 'eee, MMMM dd'; assert(strcmp(char(t), 'Tue, March 05')); "
     "t.Format = 'EEEE'; assert(strcmp(char(t), 'Tuesday'));"),
    ("t = datetime(2024, 3, 5); t.Format = 'DDD'; assert(strcmp(char(t), '065')); t.Format = 'QQQ "
     "yyyy'; assert(strcmp(char(t), 'Q1 2024')); t.Format = 'yy/M/d'; assert(strcmp(char(t), "
     "'24/3/5')); t.Format = 'uuuu'; assert(strcmp(char(t), '2024'));"),
    ("t = datetime(2024, 3, 5); t.Format = 'w'; assert(strcmp(char(t), '10')); t.Format = 'MMMMM';"
     " assert(strcmp(char(t), 'M')); t.Format = 'e'; assert(strcmp(char(t), '3'));"),
    ("p = datetime('05/03/2024 14:07', 'InputFormat', 'dd/MM/yyyy HH:mm'); assert(p.Month == 3 && "
     "p.Hour == 14); p = datetime('Mar 5, 2024', 'InputFormat', 'MMM d, yyyy'); assert(p.Day == "
     '5);'),
    ("p = datetime('2024-03-05T14:07:09Z', 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssZ', 'TimeZone', "
     "'UTC'); assert(p.Hour == 14) ; p = datetime('2:07 PM', 'InputFormat', 'h:mm a'); "
     'assert(p.Hour == 14);'),
    ("p = datetime({'2024-01-02', '2024-02-03'}); assert(isequal(size(p), [1 2]) && p(2).Month == "
     '2); p = datetime(["2024-01-02"; "2024-02-03"]); assert(isequal(size(p), [2 1]));'),
    ("assert(datetime(738000, 'ConvertFrom', 'datenum') == datetime(2020, 7, 28)); "
     "assert(datetime(86400, 'ConvertFrom', 'posixtime') == datetime(1970, 1, 2)); "
     "assert(datetime(45366, 'ConvertFrom', 'excel') == datetime(2024, 3, 15));"),
    ("assert(datetime(2451545, 'ConvertFrom', 'juliandate') == datetime(2000, 1, 1, 12, 0, 0)); "
     "assert(datetime(20240315, 'ConvertFrom', 'yyyymmdd') == datetime(2024, 3, 15)); "
     'assert(yyyymmdd(datetime(2024, 3, 15)) == 20240315);'),
    ('assert(datenum(datetime(2020, 7, 28)) == 738000); assert(juliandate(datetime(2000, 1, 1, 12,'
     ' 0, 0)) == 2451545); assert(exceltime(datetime(2024, 3, 15)) == 45366); '
     "assert(convertTo(datetime(1970, 1, 2), 'posixtime') == 86400);"),
    ("t = datetime(2024, 5, 17, 13, 45, 10); assert(dateshift(t, 'start', 'month') == "
     "datetime(2024, 5, 1)); assert(dateshift(t, 'end', 'month') == datetime(2024, 5, 31)); "
     "assert(dateshift(t, 'start', 'year') == datetime(2024, 1, 1));"),
    ("t = datetime(2024, 5, 17, 13, 45, 10); assert(dateshift(t, 'start', 'day') == datetime(2024,"
     " 5, 17)); assert(dateshift(t, 'start', 'hour') == datetime(2024, 5, 17, 13, 0, 0)); "
     "assert(dateshift(t, 'end', 'year') == datetime(2024, 12, 31));"),
    ("t = datetime(2024, 5, 17); assert(dateshift(t, 'start', 'week') == datetime(2024, 5, 12)); "
     "assert(dateshift(t, 'dayofweek', 'Monday') == datetime(2024, 5, 20)); assert(dateshift(t, "
     "'dayofweek', 'Friday') == t);"),
    ("t = datetime(2024, 5, 17); assert(dateshift(t, 'start', 'month', 'next') == datetime(2024, "
     "6, 1)); assert(dateshift(t, 'start', 'month', 'previous') == datetime(2024, 4, 1)); "
     "assert(dateshift(t, 'start', 'quarter') == datetime(2024, 4, 1));"),
    ("t = datetime(2024, 5, 17); assert(dateshift(t, 'dayofweek', 'Monday', 'previous') == "
     "datetime(2024, 5, 13)); assert(dateshift(t, 'start', 'month', 2) == datetime(2024, 7, 1)); "
     "assert(dateshift(t, 'dayofweek', 'weekend') == datetime(2024, 5, 18));"),
    ("d = between(datetime(2024, 1, 31), datetime(2024, 3, 15)); [y, mo, dd] = split(d, {'years', "
     "'months', 'days'}); assert(y == 0 && mo == 1 && dd == 15);"),
    ("d = between(datetime(2023, 1, 1), datetime(2024, 3, 2), 'days'); assert(caldays(d) == 426); "
     "d = between(datetime(2023, 1, 1), datetime(2024, 3, 2), {'months', 'days'}); [mo, dd] = "
     "split(d, {'months', 'days'}); assert(mo == 14 && dd == 1);"),
    ('d = caldiff([datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 4, 1)]); '
     'assert(isequal(calmonths(d), [1 2])); d = caldiff([datetime(2024, 1, 1), datetime(2024, 1, '
     "15)], 'weeks'); assert(calweeks(d) == 2);"),
    ('c = calmonths(1) + caldays(10); t = datetime(2024, 1, 31) + c; assert(t == datetime(2024, 3,'
     " 10)); c2 = c * 2; [mo, dd] = split(c2, {'months', 'days'}); assert(mo == 2 && dd == 20);"),
    ("c = calyears(1) + calmonths(14); [y, mo] = split(c, {'years', 'months'}); assert(y == 2 && "
     'mo == 2); assert(iscalendarduration(c)); c = calquarters(2); assert(calmonths(c) == 6); c = '
     'calweeks(2); assert(caldays(c) == 14);'),
    ('c = caldays(1) + hours(5); t = datetime(2024, 1, 1) + c; assert(t == datetime(2024, 1, 2, 5,'
     ' 0, 0)); c = -calmonths(1); assert(datetime(2024, 3, 31) + c == datetime(2024, 2, 29)); '
     'assert(datetime(2024, 3, 31) - calmonths(1) == datetime(2024, 2, 29));'),
    ('c = [calmonths(1) caldays(3)]; assert(isequal(size(c), [1 2])); assert(isequal(calmonths(c),'
     ' [1 0])); s = string(calmonths(14) + caldays(3)); assert(s == "1y 2mo 3d"); '
     "assert(strcmp(char(caldays(3)), '3d'));"),
    ('assert(isequal(isbetween(datetime(2024, 1, 5:7), datetime(2024, 1, 6), datetime(2024, 1, '
     '10)), [false true true])); assert(isbetween(datetime(2024, 1, 6), datetime(2024, 1, 6), '
     "datetime(2024, 1, 10), 'open') == false);"),
    ('assert(isequal(isweekend(datetime(2024, 1, 5:8)), [false true true false])); '
     'assert(isnat(NaT)); n = NaT(2, 2); assert(isequal(size(n), [2 2]) && all(isnat(n(:)))); '
     'assert(isequal(isnat([datetime(2024, 1, 1) NaT]), [false true]));'),
    ('assert(isnat(datetime(NaN, NaN, NaN))); t = [datetime(2024, 1, 1) NaT]; '
     'assert(isequal(ismissing(t), [false true])); assert(isequal(rmmissing(t), datetime(2024, 1, '
     '1)));'),
    ("d = duration(1, 30, 0); assert(minutes(d) == 90); assert(strcmp(char(d), '01:30:00')); "
     "d.Format = 'm'; assert(strcmp(char(d), '90 min')); d.Format = 'hh:mm'; "
     "assert(strcmp(char(d), '01:30')); d.Format = 's'; assert(strcmp(char(d), '5400 sec'));"),
    ("d = duration(1, 30, 15.5); d.Format = 'hh:mm:ss.SSS'; assert(strcmp(char(d), "
     "'01:30:15.500')); d = duration(1, 30, 15); d.Format = 'mm:ss'; assert(strcmp(char(d), "
     "'90:15')); d.Format = 'dd:hh:mm:ss'; assert(strcmp(char(d), '00:01:30:15'));"),
    ("d = hours(36); d.Format = 'd'; assert(strcmp(char(d), '1.5 days')); d.Format = 'h'; "
     "assert(strcmp(char(d), '36 hr')); d = years(2); assert(strcmp(char(d), '2 yrs')); "
     'assert(days(years(1)) == 365.2425);'),
    ('[h, m, s] = hms(duration(1, 30, 15)); assert(h == 1 && m == 30 && s == 15); t = '
     'datetime(2024, 1, 1, 13, 45, 10); assert(timeofday(t) == duration(13, 45, 10)); [h, m, s] = '
     'hms(t); assert(h == 13 && m == 45 && s == 10);'),
    ('d = duration(\'01:30:00\'); assert(hours(d) == 1.5); d = duration("12:34:56.5"); '
     "assert(seconds(d) == 45296.5); d = duration('1:2:3:4', 'InputFormat', 'dd:hh:mm:ss'); "
     'assert(hours(d) == 26 + 3 / 60 + 4 / 3600);'),
    ('d = duration([1 2; 3 4], 0, 0); assert(isequal(hours(d), [1 2; 3 4])); d = duration(1, 0, 0,'
     ' 500); assert(seconds(d) == 3600.5); d = duration([1 30 0]); assert(minutes(d) == 90); '
     'assert(milliseconds(seconds(1.5)) == 1500);'),
    ('d = [hours(1) minutes(30)]; assert(isequal(sort(d), [minutes(30) hours(1)])); assert(max(d) '
     '== hours(1)); assert(seconds(mean(d)) == 2700); assert(isequal(cumsum(d), [hours(1) '
     'minutes(90)])); assert(seconds(diff(d)) == -1800);'),
    ('d = hours(1); assert(d * 2 == hours(2)); assert(d / 2 == minutes(30)); assert(d / '
     'minutes(30) == 2); assert(-d == hours(-1)); assert(abs(-d) == d); assert(d + 1 == '
     'hours(25)); assert(isduration(d) && ~isduration(1));'),
    ("d = minutes(90); assert(floor(hours(d)) == 1); assert(isequal(round(hours(1.4), 'hours'), "
     "hours(1))); assert(isequal(ceil(minutes(61), 'hours'), hours(2)));"),
    ('assert(isequal(string(hours(1.5)), "1.5 hr")); assert(isequal(cellstr([hours(1) hours(2)]), '
     "{'1 hr', '2 hr'})); assert(isnan(hours(NaN))); assert(isinf(hours(Inf))); "
     'assert(isequal(seconds([1 2]) > seconds(1.5), [false true]));'),
    ("t = datetime(2024, 5, 17); assert(week(t) == 20); assert(week(t, 'weekofmonth') == 3); "
     "assert(quarter(t) == 2); assert(day(t) == 17); assert(day(t, 'dayofyear') == 138); "
     "assert(day(t, 'dayofweek') == 6);"),
    ("t = datetime(2024, 5, 17); assert(strcmp(day(t, 'name'), 'Friday')); assert(strcmp(day(t, "
     "'shortname'), 'Fri')); assert(strcmp(month(t, 'name'), 'May')); assert(strcmp(month(t, "
     "'shortname'), 'May')); assert(year(t) == 2024);"),
    ('t = datetime(2024, 5, 17, 13, 45, 10.5); assert(hour(t) == 13 && minute(t) == 45 && '
     "second(t) == 10.5); assert(second(t, 'secondofday') == 13 * 3600 + 45 * 60 + 10.5); [y, m, "
     'd] = ymd(t); assert(d == 17);'),
    ("t = datetime(2024, 5, 17); [n, s] = weekday(t); assert(n == 6 && strcmp(s, 'Fri')); "
     'assert(eomday(2024, 2) == 29); assert(isequal(year([t t + calyears(1)]), [2024 2025])); '
     'assert(week(datetime(2024, 1, 1)) == 1);'),
    ("t = datetime(2024, 1, 15, 10, 30, 45); assert(dateshift(t, 'start', 'minute') == "
     "datetime(2024, 1, 15, 10, 30, 0)); assert(dateshift(t, 'end', 'day') == datetime(2024, 1, "
     "16)); assert(dateshift(t, 'start', 'hour', 'nearest') == datetime(2024, 1, 15, 11, 0, 0));"),
    ('t = datetime(2024, 1, 31):calmonths(1):datetime(2024, 4, 30); assert(numel(t) == 4 && t(2) '
     '== datetime(2024, 2, 29)); t = datetime(2024, 1, 1):hours(12):datetime(2024, 1, 2); '
     'assert(numel(t) == 3); l = linspace(datetime(2024, 1, 1), datetime(2024, 1, 3), 3); '
     'assert(l(2) == datetime(2024, 1, 2));'),
    ('t = datetime(2024, 1, 1) + days(0:2); assert(isequal(diff(t), [days(1) days(1)])); '
     'assert(max(t) == t(3)); [m, i] = min(t); assert(i == 1); assert(mean(t) == t(2)); '
     'assert(isequal(unique([t t]), t)); assert(ismember(t(2), t));'),
    ("t = datetime(2024, 1, 1, 10, 0, 0); assert(strcmp(datestr(t), '01-Jan-2024 10:00:00')); "
     "assert(strcmp(datestr(t, 'HH:MM PM'), '10:00 AM')); assert(strcmp(datestr(t, 1), "
     "'01-Jan-2024')); assert(strcmp(datestr(t, 29), '2024-01-01')); assert(strcmp(datestr(t, 'mmm"
     " dd, yyyy'), 'Jan 01, 2024'));"),
]


CATEGORICAL_STRUCT_CELL_CLASSES = [
    ("c = categorical({'a', 'b', 'a', 'c'}); assert(isequal(categories(c), {'a'; 'b'; 'c'})); "
     "assert(isequal(countcats(c), [2 1 1])); assert(isequal(c == 'a', [true false true false])); "
     'assert(isequal(double(c), [1 2 1 3]));'),
    ("o = categorical({'lo', 'hi', 'mid'}, {'lo', 'mid', 'hi'}, 'Ordinal', true); assert(max(o) =="
     " 'hi'); assert(min(o) == 'lo'); assert(median(o) == 'mid'); assert(o(1) < o(2)); "
     'assert(isordinal(o));'),
    ("o = categorical({'lo', 'hi', 'mid', 'lo'}, {'lo', 'mid', 'hi'}, 'Ordinal', true); s = "
     "sort(o); assert(isequal(cellstr(s), {'lo', 'lo', 'mid', 'hi'})); [m, i] = max(o); assert(i "
     "== 2); assert(isequal(o >= 'mid', [false true true false]));"),
    ("o = categorical({'lo', 'hi', 'mid', 'lo'}, {'lo', 'mid', 'hi'}, 'Ordinal', true); s = "
     "sort(o, 'descend'); assert(isequal(cellstr(s), {'hi', 'mid', 'lo', 'lo'})); assert(isequal(o"
     " > 'lo', [false true true false])); assert(isequal(o ~= 'lo', [false true true false]));"),
    ("o = categorical([1 3 2], 1:3, {'S', 'M', 'L'}, 'Ordinal', true); assert(o(2) == 'L'); "
     "assert(isequal(cellstr(max(o, categorical({'M'}, {'S', 'M', 'L'}, 'Ordinal', true))), {'M', "
     "'L', 'M'})); assert(isequal(o < 'L', [true false true]));"),
    ("c = categorical({'a', 'b'}); threw = false; try, c(1) < c(2); catch, threw = true; end; "
     'assert(threw); threw = false; try, max(c); catch, threw = true; end; assert(threw);'),
    ("c = categorical({'a', 'b', 'a'}); c2 = setcats(c, {'b', 'z'}); "
     "assert(isequal(categories(c2), {'b'; 'z'})); assert(isundefined(c2(1)) && c2(2) == 'b');"),
    ("c = categorical({'a', 'b', 'a'}); c2 = reordercats(c, {'b', 'a'}); "
     "assert(isequal(categories(c2), {'b'; 'a'})); assert(isequal(double(c2), [2 1 2])); c3 = "
     "reordercats(c2); assert(isequal(categories(c3), {'a'; 'b'}));"),
    ("c = categorical({'a', 'b'}); c2 = addcats(c, {'x', 'y'}); assert(numel(categories(c2)) == "
     "4); c3 = addcats(c, 'z', 'Before', 'a'); assert(isequal(categories(c3), {'z'; 'a'; 'b'})); "
     "c4 = addcats(c, 'm', 'After', 'a'); assert(isequal(categories(c4), {'a'; 'm'; 'b'}));"),
    ("c = categorical({'a', 'b'}, 'Protected', true); assert(isprotected(c)); threw = false; try, "
     "c(3) = 'zzz'; catch, threw = true; end; assert(threw); c(3) = 'a'; assert(numel(c) == 3);"),
    ("c = categorical({'a', 'b'}); assert(~isprotected(c)); c(3) = 'new'; "
     "assert(numel(categories(c)) == 3); o = categorical({'x'}, {'x', 'y'}, 'Ordinal', true); "
     'assert(isprotected(o));'),
    ("c = categorical({'a', 'b', 'a', 'c'}); c2 = mergecats(c, {'a', 'b'}, 'ab'); "
     "assert(isequal(categories(c2), {'ab'; 'c'})); c3 = renamecats(c, 'c', 'z'); assert(any(c3 =="
     " 'z')); c4 = removecats(c, 'c'); assert(isundefined(c4(4)));"),
    ("c = categorical({'a', 'b', 'a', 'c'}); c5 = removecats(addcats(c, 'unused')); "
     "assert(numel(categories(c5)) == 3); assert(isequal(iscategory(c, {'a', 'q'}), [true "
     "false])); assert(isequal(cellstr(c(1:2)), {'a', 'b'}));"),
    ("c = categorical([1 2 1 3]); assert(isequal(categories(c), {'1'; '2'; '3'})); c = "
     "categorical([1 2 1 3], [1 2 3], {'one', 'two', 'three'}); assert(c(4) == 'three'); c = "
     'categorical(["x" "y" "x"]); assert(isequal(countcats(c), [2 1]));'),
    ("c = categorical([true false true]); assert(isequal(categories(c), {'false'; 'true'})); c = "
     "categorical({'a', '', 'b'}); assert(isundefined(c(2))); assert(isequal(ismissing(c), [false "
     'true false]));'),
    ("c = categorical({'b', 'a', 'b'}); [u, i, j] = unique(c); assert(isequal(cellstr(u), {'a', "
     "'b'}) && isequal(j(:)', [2 1 2])); assert(isequal(ismember(c, {'a'}), [false true false])); "
     'assert(isequal(string(c), ["b" "a" "b"]));'),
    ("c = categorical({'b', 'a', 'b'}); assert(mode(c) == 'b'); s = evalc('summary(c)'); "
     "assert(contains(s, 'a') && contains(s, '2')); assert(isequal(size(c'), [3 1])); c2 = [c "
     "categorical({'z'})]; assert(numel(c2) == 4 && numel(categories(c2)) == 3);"),
    ("c = categorical({'b', 'a', 'b'}); assert(isequal(c == categorical({'b', 'b', 'b'}), [true "
     'false true])); assert(isequal(c == ["b" "a" "x"], [true true false])); assert(nnz(c == \'b\') '
     "== 2); assert(isequal(find(c == 'a'), 2));"),
    ('d = discretize([1 5 10], [0 4 8 12]); assert(isequal(d, [1 2 3])); d = discretize([1 5 10], '
     "[0 4 8 12], 'categorical'); assert(iscategorical(d) && isequal(cellstr(d), {'[0, 4)', '[4, "
     "8)', '[8, 12]'}));"),
    ("d = discretize([1 5 10], [0 4 8 12], 'categorical', {'low', 'mid', 'high'}); assert(d(2) == "
     "'mid'); d = discretize([1 5 10], [0 4 8 12], [10 20 30]); assert(isequal(d, [10 20 30])); d "
     '= discretize([1 5 20], [0 4 8 12]); assert(isnan(d(3)));'),
    ("d = discretize([4 8], [0 4 8], 'IncludedEdge', 'right'); assert(isequal(d, [1 2])); d = "
     'discretize([1 2 3 4 5 6], 3); assert(isequal(d, [1 1 2 2 3 3]) || numel(unique(d)) == 3); '
     '[d, e] = discretize([0.5 1.5], [0 1 2]); assert(isequal(e, [0 1 2]));'),
    ("s = evalc('c = categorical({''a'', ''b''})'); assert(contains(s, '1×2 categorical array') &&"
     " contains(s, 'a') && contains(s, 'b'));"),
    ("s = struct('a', 1, 'b', 'x'); assert(isequal(fieldnames(s), {'a'; 'b'})); "
     "assert(isequal(isfield(s, {'a', 'z'}), [true false])); s = rmfield(s, 'b'); "
     "assert(~isfield(s, 'b')); f = 'dyn'; s.(f) = 5; assert(s.dyn == 5);"),
    ("s = struct('a', 1, 'b', 2, 'c', 3); s2 = rmfield(s, {'a', 'c'}); "
     'assert(isequal(fieldnames(s2), {\'b\'})); s3 = rmfield(s, ["a" "b"]); '
     "assert(isequal(fieldnames(s3), {'c'})); threw = false; try, rmfield(s, 'zz'); catch, threw ="
     ' true; end; assert(threw);'),
    ("s = struct('z', 1, 'a', 2); [o, p] = orderfields(s); assert(isequal(fieldnames(o), {'a'; "
     "'z'}) && isequal(p, [2; 1])); o = orderfields(s, {'z', 'a'}); assert(isequal(fieldnames(o), "
     "{'z'; 'a'})); o = orderfields(s, [2 1]); assert(isequal(fieldnames(o), {'a'; 'z'}));"),
    ("s = struct('z', 1, 'a', 2); t = struct('a', 0, 'z', 0); o = orderfields(s, t); "
     "assert(isequal(fieldnames(o), {'a'; 'z'}) && o.a == 2);"),
    ("n.a.b.c = 4; assert(getfield(n, 'a', 'b', 'c') == 4); n = setfield(n, 'a', 'b', 'd', 9); "
     "assert(n.a.b.d == 9); t = struct('v', {10, 20, 30}); assert(getfield(t, {2}, 'v') == 20); t "
     "= setfield(t, {3}, 'v', 99); assert(t(3).v == 99);"),
    ("s.m = magic(3); assert(getfield(s, 'm', {2, 2}) == 5); s = setfield(s, 'm', {1, 1}, 100); "
     "assert(s.m(1, 1) == 100); assert(isequal(isfield(struct('a', {1, 2}), 'a'), true)); "
     'assert(~isfield(5, \'a\')); assert(isfield(struct(\'a\', 1), "a"));'),
    ("t = struct('v', {1, 2, 3}); assert(isequal(size(t), [1 3])); assert(isequal(arrayfun(@(e) "
     "e.v * 2, t), [2 4 6])); u = struct('c', {{1, 2}}); assert(iscell(u.c) && numel(u) == 1); "
     "assert(isempty(struct([]))); assert(isequal(size(struct('a', {})), [0 0]));"),
    ("c = struct2cell(struct('p', 1, 'q', 'w')); assert(isequal(c, {1; 'w'})); r = cell2struct({1;"
     " 2}, {'x', 'y'}, 1); assert(r.y == 2); r = cell2struct({1, 2; 3, 4}, {'x', 'y'}, 2); "
     'assert(numel(r) == 2 && r(2).x == 3);'),
    ("assert(isequal(structfun(@(v) v * 2, struct('a', 1, 'b', 2)), [2; 4])); r = structfun(@(v) "
     "[v v], struct('a', 1, 'b', 2), 'UniformOutput', false); assert(isequal(r.b, [2 2])); r = "
     "structfun(@(v) error('x'), struct('a', 1), 'ErrorHandler', @(e, v) -1); assert(r == -1);"),
    ("s = struct('a', {1, 2, 3}); t = [s.a]; assert(isequal(t, [1 2 3])); [s.a] = deal(0); "
     'assert(all([s.a] == 0)); s(2) = []; assert(numel(s) == 2); q = arrayfun(@(e) e.a, s); '
     'assert(isequal(q, [0 0]));'),
    ("s = struct('name', {'x', 'y'}, 'val', {1, 2}); names = {s.name}; assert(isequal(names, {'x',"
     " 'y'})); [s.val] = deal(5, 6); assert(s(2).val == 6); s(3).name = 'z'; "
     'assert(isempty(s(3).val));'),
    ("s = struct('a', 1); s(2).a = 2; s2 = [s struct('a', 3)]; assert(numel(s2) == 3); w = "
     "repmat(struct('m', 0), 2, 2); assert(isequal(size(w), [2 2])); x = struct('a', {{}}); "
     'assert(iscell(x.a) && isempty(x.a));'),
    ("s = struct(); k = 'key'; s.([k '1']) = 7; assert(s.key1 == 7); s.(sprintf('f%d', 2)) = 8; "
     "assert(s.f2 == 8); v = s.(['key' '1']); assert(v == 7);"),
    "s = evalc('t = struct(''a'', {1, 2})'); assert(contains(s, '1×2 struct array with fields:') && contains(s, 'a'));",
    ("c = {1, 'a', [1 2 3]}; assert(isequal(cellfun(@numel, c), [1 1 3])); "
     "assert(isequal(cellfun('isempty', {[], 1}), [true false])); assert(isequal(cellfun('length',"
     " {[1 2], 'abc'}), [2 3])); assert(isequal(cellfun(@isempty, {[], 'a'}), [true false]));"),
    ("assert(isequal(cellfun('ndims', {1, ones(2, 2, 2)}), [2 3])); "
     "assert(isequal(cellfun('prodofsize', {ones(2, 3)}), 6)); assert(isequal(cellfun('isclass', "
     "{1, 'a'}, 'char'), [false true])); assert(isequal(cellfun('size', {ones(2, 3)}, 2), 3));"),
    ("assert(isequal(cellfun('isreal', {1, 1i}), [true false])); "
     "assert(isequal(cellfun('islogical', {true, 1}), [true false])); "
     "assert(isequal(cellfun(@upper, {'a', 'b'}, 'UniformOutput', false), {'A', 'B'}));"),
    ("d = cellfun(@(x) x * 2, {1, 2}, 'UniformOutput', false); assert(isequal(d, {2, 4})); [mx, "
     'ix] = cellfun(@max, {[1 5], [7 2]}); assert(isequal(mx, [5 7]) && isequal(ix, [2 1])); '
     'assert(isequal(cellfun(@plus, {1, 2}, {10, 20}), [11 22]));'),
    ("r = cellfun(@(x) error('bad'), {1, 2}, 'ErrorHandler', @(err, x) -1); assert(isequal(r, [-1 "
     "-1])); r = cellfun(@(x) error('my:id', 'msg'), {5}, 'ErrorHandler', @(err, x) "
     "strcmp(err.identifier, 'my:id') && err.index == 1 && x == 5); assert(r);"),
    ("r = cellfun(@(x) x, {1, 'a'}, 'UniformOutput', 0); assert(iscell(r)); r = cellfun(@(x, y) [x"
     " y], {'a', 'b'}, {'c', 'd'}, 'uniformoutput', false); assert(isequal(r, {'ac', 'bd'})); "
     'assert(isempty(cellfun(@numel, {})));'),
    ("a = arrayfun(@(k) 1:k, 1:2, 'UniformOutput', false); assert(isequal(a{2}, [1 2])); "
     "assert(isequal(arrayfun(@(k) k^2, [1 2; 3 4]), [1 4; 9 16])); r = arrayfun(@(x) error('e'), "
     "1:2, 'ErrorHandler', @(s, x) x * 10); assert(isequal(r, [10 20]));"),
    ('[a, b] = arrayfun(@(x) deal(x, x * 2), 1:3); assert(isequal(b, [2 4 6])); r = arrayfun(@(s) '
     "numel(s.v), struct('v', {1, [1 2]})); assert(isequal(r, [1 2]));"),
    ('[p, q, r] = deal(1, 2, 3); assert(r == 3); [p, q] = deal(7); assert(p == 7 && q == 7); c = '
     'cell(1, 2); [c{:}] = deal(0); assert(isequal(c, {0, 0})); threw = false; try, [p, q] = '
     'deal(1, 2, 3); catch, threw = true; end; assert(threw);'),
    ('assert(isequal(cell2mat({[1 2], 3; [4 5], 6}), [1 2 3; 4 5 6])); assert(isequal(num2cell([1 '
     '2; 3 4]), {1, 2; 3, 4})); m = mat2cell(magic(4), [2 2], [3 1]); assert(isequal(size(m{1, '
     "1}), [2 3])); assert(iscellstr({'a', 'b'}) && ~iscellstr({'a', 1}));"),
    ("c = {3, 'a', [1 2]}; c(2) = []; assert(numel(c) == 2); e = cell(2, 3); assert(isempty(e{2, "
     "3})); assert(isequal(sort({'b', 'a', 'c'}), {'a', 'b', 'c'})); "
     "assert(isequal(cellfun(@class, {1, 'a', {}}, 'UniformOutput', false), {'double', 'char', "
     "'cell'}));"),
    ("c = {1, 2; 3, 4}; assert(isequal(c(:, 1), {1; 3})); assert(isequal(size(c'), [2 2])); "
     'assert(isequal(vertcat(c{:, 1}), [1; 3])); c{3, 3} = 9; assert(isequal(size(c), [3 3]) && '
     'isempty(c{3, 1})); assert(isequal(cellfun(@isempty, c), logical([0 0 1; 0 0 1; 1 1 0])));'),
    ('c = num2cell([1 2 3]); [a, b, d] = c{:}; assert(d == 3); x = cell(1, 3); x(:) = {5}; '
     'assert(isequal(x, {5, 5, 5})); y = [c, {4}]; assert(numel(y) == 4); z = [c; c]; '
     'assert(isequal(size(z), [2 3])); assert(isequal(iscell(c), true));'),
    ("assert(isequal(typecast(single(1), 'uint32'), uint32(1065353216))); "
     "assert(isequal(typecast(uint8([1 0 0 0]), 'uint32'), uint32(1))); "
     "assert(isequal(typecast(int16(-1), 'uint16'), uint16(65535))); assert(isequal(typecast(1, "
     "'uint64'), uint64(4607182418800017408)));"),
    ("assert(isequal(typecast(uint8([0 0 128 63]), 'single'), single(1))); "
     "assert(isequal(size(typecast(uint32([1 2]), 'uint8')), [1 8])); "
     "assert(isequal(typecast(uint16([1; 2]), 'uint8'), uint8([1; 0; 2; 0]))); "
     "assert(isa(typecast(int8(1), 'uint8'), 'uint8'));"),
    ("assert(isa(cast(1.7, 'int8'), 'int8') && cast(1.7, 'int8') == 2); assert(isa(cast(int8(5), "
     "'double'), 'double')); assert(isa(cast(1, 'like', single(2)), 'single')); "
     "assert(isa(cast(300, 'uint8'), 'uint8') && cast(300, 'uint8') == 255); "
     "assert(islogical(cast(1, 'logical')));"),
    ("assert(isa(cast('a', 'double'), 'double') && cast('a', 'double') == 97); "
     "assert(ischar(cast(97, 'char'))); assert(isa(cast(1, 'like', 1i), 'double')); "
     "assert(isa(cast(5, 'like', int16(1)), 'int16')); assert(isa(cast(true, 'single'), "
     "'single'));"),
    ("assert(strcmp(class([int8(1) 2.7]), 'int8')); assert(strcmp(class([single(1) 2]), "
     "'single')); assert(strcmp(class([true 2]), 'double')); assert(strcmp(class(['a' 66]), "
     "'char')); assert(strcmp(class([true 'a']), 'char')); assert(strcmp(class([int8(1) "
     "int16(2)]), 'int8'));"),
    ("assert(strcmp(class([single(1) int8(2)]), 'int8')); assert(strcmp(class({1, 'a'}), 'cell'));"
     ' assert(strcmp(class([{1}, 2]), \'cell\')); assert(strcmp(class(["a" 1]), \'string\')); '
     'assert(strcmp(class([\'a\' "b"]), \'string\')); assert(strcmp(class([1 true]), \'double\'));'),
    ("assert(strcmp(class([int8(100) int16(1000)]), 'int8') && isequal([int8(100) int16(1000)], "
     "int8([100 127]))); assert(strcmp(class([uint8(1) 'a']), 'char')); assert(strcmp(class([true "
     "false]), 'logical')); assert(strcmp(class([[] int8(1)]), 'int8'));"),
    ("assert(strcmp(class(1:int8(3)), 'int8')); assert(strcmp(class(single(1):3), 'single')); "
     "assert(strcmp(class(true + true), 'double')); assert(strcmp(class(int8(1) * 2.5), 'int8')); "
     "assert(strcmp(class(single(1) + int8(1)), 'int8')); assert(strcmp(class('a' + 'b'), "
     "'double'));"),
    ("assert(strcmp(class(1 == 1), 'logical')); assert(strcmp(class(~1), 'logical')); "
     "assert(strcmp(class(-true), 'double')); assert(strcmp(class(single(1) * 1i), 'single')); "
     "assert(strcmp(class(uint8(5) / uint8(2)), 'uint8') && uint8(5) / uint8(2) == 3);"),
    ("assert(strcmp(class([datetime(2024, 1, 1) NaT]), 'datetime')); assert(strcmp(class([hours(1)"
     " 2]), 'duration')); assert(strcmp(class([categorical({'a'}) {'b'}]), 'categorical')); "
     "assert(strcmp(class({}), 'cell')); assert(strcmp(class(@sin), 'function_handle'));"),
    ("assert(isa(1, 'numeric') && isa(1, 'float') && ~isa(1, 'integer')); assert(isa(int8(1), "
     "'integer') && isa(int8(1), 'numeric')); assert(isa(single(1), 'float')); assert(~isa(true, "
     '\'numeric\')); assert(isa("a", \'string\') && ~isa(\'a\', \'string\'));'),
    ("assert(isequal(double('a'), 97)); assert(isequal(char([72 105]), 'Hi')); "
     'assert(isequal(logical([2 0 -1]), [true false true])); assert(isequal(int8([1.5 -1.5 200]), '
     'int8([2 -2 127]))); assert(isequal(uint8(-5), uint8(0))); assert(isequal(single(1) == 1, '
     'true));'),
    ('assert(ismissing(missing)); m = [1 missing 3]; assert(isnan(m(2))); m = ["a" missing]; '
     'assert(ismissing(m(2))); m = {1, missing}; assert(ismissing(m{2})); '
     "assert(strcmp(class(missing), 'missing')); assert(isequal(size(missing), [1 1]));"),
    ("m = [datetime(2024, 1, 1) missing]; assert(isnat(m(2))); m = [categorical({'a'}) missing]; "
     'assert(isundefined(m(2))); x = [1 2 3]; x(2) = missing; assert(isnan(x(2))); s = ["a" "b"]; '
     's(1) = missing; assert(ismissing(s(1)));'),
]


@pytest.mark.parametrize("code", DICTIONARY_AND_MAP)
def test_dictionary_and_map(m, code):
    m.run(code)


@pytest.mark.parametrize("code", DATES_AND_DURATIONS)
def test_dates_and_durations(m, code):
    m.run(code)


@pytest.mark.parametrize("code", CATEGORICAL_STRUCT_CELL_CLASSES)
def test_categorical_struct_cell_classes(m, code):
    m.run(code)


