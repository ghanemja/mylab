"""datetime, duration and calendarDuration arrays, and the serial-date-number functions."""

import datetime as pydt

import numpy as np
import pytest

from mylab.builtins.datetimes import (
    CalendarDuration, DatetimeArray, DurationArray, civil_from_days, days_from_civil,
)
from mylab.errors import MatlabError
from mylab.values import Cell, to_str


def secs(m, expr):
    """Seconds since 1970 of a datetime expression (or the seconds of a duration)."""
    return np.asarray(m.interp.eval_expression(expr).data, dtype=float)


def stamp(*parts):
    return (pydt.datetime(*parts) - pydt.datetime(1970, 1, 1)).total_seconds()


def texts(m, expr):
    v = m.interp.eval_expression(f"cellstr({expr})")
    assert isinstance(v, Cell)
    return [to_str(x) for x in v.values_f()]


# -- calendar arithmetic ------------------------------------------------------------------------------

def test_civil_day_numbers_match_python():
    for y, mo, d in [(1970, 1, 1), (2000, 2, 29), (2024, 12, 31), (1900, 3, 1), (1600, 2, 29), (1, 1, 1)]:
        want = (pydt.date(y, mo, d) - pydt.date(1970, 1, 1)).days
        assert float(days_from_civil(y, mo, d)) == want
        assert tuple(float(x) for x in civil_from_days(want)) == (y, mo, d)


# -- construction ----------------------------------------------------------------------------------------

def test_from_components(m):
    assert isinstance(m.interp.eval_expression("datetime(2024,3,15)"), DatetimeArray)
    assert secs(m, "datetime(2024,3,15)").item() == stamp(2024, 3, 15)
    assert secs(m, "datetime(2024,3,15,10,30,45)").item() == stamp(2024, 3, 15, 10, 30, 45)
    np.testing.assert_array_equal(secs(m, "datetime(2024,1,1:3)"), [[stamp(2024, 1, d) for d in (1, 2, 3)]])
    np.testing.assert_array_equal(secs(m, "datetime([2020 1 5; 2021 12 25])"),
                                  [[stamp(2020, 1, 5)], [stamp(2021, 12, 25)]])
    # values roll over
    assert secs(m, "datetime(2024,14,1)").item() == stamp(2025, 2, 1)
    assert secs(m, "datetime(2024,1,0)").item() == stamp(2023, 12, 31)
    assert secs(m, "datetime(2024,1,1,25,0,0)").item() == stamp(2024, 1, 2, 1)
    assert secs(m, "datetime(2024,1,1,0,0,0,500)").item() == stamp(2024, 1, 1) + 0.5


def test_now_and_relative_days(m):
    now = pydt.datetime.now()
    got = secs(m, "datetime('now')").item()
    assert abs(got - (now - pydt.datetime(1970, 1, 1)).total_seconds()) < 5
    assert abs(secs(m, "datetime").item() - got) < 5
    today = secs(m, "datetime('today')").item()
    assert today % 86400 == 0 and 0 <= got - today < 86400 + 5
    assert secs(m, "datetime('tomorrow')").item() - today == 86400
    assert today - secs(m, "datetime('yesterday')").item() == 86400


@pytest.mark.parametrize("text, parts", [
    ("2024-03-15", (2024, 3, 15)), ("2024-03-15 10:30:45", (2024, 3, 15, 10, 30, 45)),
    ("2024-03-15T10:30:45", (2024, 3, 15, 10, 30, 45)), ("15-Mar-2024", (2024, 3, 15)),
    ("15-Mar-2024 08:05:00", (2024, 3, 15, 8, 5)), ("2024/03/15", (2024, 3, 15)), ("03/15/2024", (2024, 3, 15)),
    ("15.03.2024", (2024, 3, 15)), ("March 15, 2024", (2024, 3, 15)), ("Mar 15, 2024", (2024, 3, 15)),
    ("15 Mar 2024 14:22", (2024, 3, 15, 14, 22)), ("3/5/2021", (2021, 3, 5)),
    ("03/15/2024 5:06 PM", (2024, 3, 15, 17, 6)),
    ("25/12/2021", (2021, 12, 25)), ("20240315", (2024, 3, 15)),
])
def test_text_formats_are_detected(m, text, parts):
    assert secs(m, f"datetime('{text}')").item() == stamp(*parts)


def test_text_arrays_and_input_format(m):
    np.testing.assert_array_equal(secs(m, "datetime({'2020-01-05','2021-12-25'})"),
                                  [[stamp(2020, 1, 5), stamp(2021, 12, 25)]])
    np.testing.assert_array_equal(secs(m, 'datetime(["2020-01-05"; "2021-12-25"])'),
                                  [[stamp(2020, 1, 5)], [stamp(2021, 12, 25)]])
    v = secs(m, "datetime({'2020-01-05','','NaT'})")
    assert v[0, 0] == stamp(2020, 1, 5) and np.isnan(v[0, 1]) and np.isnan(v[0, 2])
    assert secs(m, "datetime('05/01/2020','InputFormat','dd/MM/yyyy')").item() == stamp(2020, 1, 5)
    assert secs(m, "datetime('2021.03.05 at 7h','InputFormat','yyyy.MM.dd ''at'' H''h''')").item() == \
        stamp(2021, 3, 5, 7)
    assert secs(m, "datetime('Tue, 5 Mar 2021 1:02:03.250 PM','InputFormat','eee, d MMM yyyy h:mm:ss.SSS a')").item() \
        == stamp(2021, 3, 5, 13, 2, 3) + 0.25
    # a mixed list: the layout that fits every entry wins (day first here)
    np.testing.assert_array_equal(secs(m, "datetime({'05/01/2020','25/12/2021'})"),
                                  [[stamp(2020, 1, 5), stamp(2021, 12, 25)]])


def test_text_errors(m):
    for code in ("datetime('garbage')", "datetime('2020-01-05','InputFormat','dd/MM/yyyy')",
                 "datetime(2024,1,1,'TimeZone','Europe/Paris')", "datetime(2024,1)", "datetime({1,2})",
                 "datetime(2024,1,1,'Format','bogus')", "datetime(5,'ConvertFrom','nonsense')"):
        with pytest.raises(MatlabError):
            m.run(code + ";")
    with pytest.raises(MatlabError, match="UTC"):
        m.run("datetime(2024,1,1,'TimeZone','Asia/Tokyo');")


def test_convert_from(m):
    assert secs(m, "datetime(738000.5,'ConvertFrom','datenum')").item() == stamp(2020, 7, 28, 12)
    assert secs(m, "datetime(86400,'ConvertFrom','posixtime')").item() == 86400
    assert secs(m, "datetime(45366,'ConvertFrom','excel')").item() == stamp(2024, 3, 15)
    assert secs(m, "datetime(2451545,'ConvertFrom','juliandate')").item() == stamp(2000, 1, 1, 12)
    assert secs(m, "datetime(20240315,'ConvertFrom','yyyymmdd')").item() == stamp(2024, 3, 15)
    assert secs(m, "datetime(86400000,'ConvertFrom','epochtime','TicksPerSecond',1000)").item() == 86400


def test_conversions_back(m):
    m.run("t = datetime(2024,3,15,6,0,0);")
    assert m.num("datenum(t)").item() == 739326.25
    assert m.num("posixtime(t)").item() == stamp(2024, 3, 15, 6)
    assert m.num("juliandate(datetime(2000,1,1,12,0,0))").item() == 2451545
    assert m.num("exceltime(datetime(2024,3,15))").item() == 45366
    assert m.num("yyyymmdd(t)").item() == 20240315
    assert m.num("convertTo(t,'posixtime')").item() == stamp(2024, 3, 15, 6)
    assert m.num("convertTo(t,'datenum')").item() == 739326.25


def test_time_zones(m):
    m.run("u = datetime(2024,1,1,12,0,0,'TimeZone','UTC'); z = u.TimeZone; p = posixtime(u);")
    assert m.str("z") == "UTC" and m.num("p").item() == stamp(2024, 1, 1, 12)
    assert secs(m, "datetime('2024-03-15T10:30:00Z','TimeZone','UTC')").item() == stamp(2024, 3, 15, 10, 30)
    m.run("v = datetime(2024,1,1,12,0,0,'TimeZone','local'); w = v; w.TimeZone = 'UTC'; "
          "d = posixtime(v) - posixtime(w);")
    assert m.num("d").item() == 0  # same instant, different wall clock
    assert m.str("datetime(2024,1,1).TimeZone") == ""
    with pytest.raises(MatlabError, match="time zone"):
        m.run("u - datetime(2024,1,1);")


# -- properties and formats ---------------------------------------------------------------------------------

def test_component_properties(m):
    m.run("t = datetime(2024,2,29,13,5,9.25); a = [t.Year t.Month t.Day t.Hour t.Minute t.Second];")
    np.testing.assert_array_equal(m.num("a"), [[2024, 2, 29, 13, 5, 9.25]])
    m.run("v = datetime(2024,1,1:3); v.Day = [5 6 7]; d = v.Day; v.Year = 2030; y = v.Year; e = v.Day(2);")
    np.testing.assert_array_equal(m.num("d"), [[5, 6, 7]])
    np.testing.assert_array_equal(m.num("y"), [[2030, 2030, 2030]])
    assert m.num("e").item() == 6
    m.run("t = datetime(2024,1,31); t.Month = 2; q = t.Day; t.Hour = 30; r = [t.Day t.Hour];")
    assert m.num("q").item() == 29  # clamped to the end of February
    np.testing.assert_array_equal(m.num("r"), [[1, 6]])  # 30 hours roll into March 1
    with pytest.raises(MatlabError):
        m.run("t.Nope")
    with pytest.raises(MatlabError):
        m.run("t.Nope = 1;")


@pytest.mark.parametrize("fmt, want", [
    ("yyyy-MM-dd HH:mm:ss.SSS", "2024-02-09 13:05:09.250"), ("dd/MM/yy", "09/02/24"), ("d MMM yyyy", "9 Feb 2024"),
    ("eeee, MMMM d, yyyy", "Friday, February 9, 2024"), ("eee h:mm a", "Fri 1:05 PM"), ("hh:mm:ss a", "01:05:09 PM"),
    ("yyyyMMdd'T'HHmmss", "20240209T130509"), ("'Q'Q yyyy", "Q1 2024"), ("QQQ", "Q1"), ("DDD", "040"),
    ("MMMMM", "F"), ("H 'o''clock'", "13 o'clock"), ("M/d/y", "2/9/2024"),
])
def test_format_letters(m, fmt, want):
    m.run("t = datetime(2024,2,9,13,5,9.25);")
    m.interp.set_var("f", fmt)
    m.run("t.Format = f; s = char(t);")
    assert m.str("s") == want
    assert m.str("t.Format") == fmt


def test_default_format(m):
    assert m.str("datetime(2024,1,5).Format") == "dd-MMM-yyyy"
    assert m.str("datetime(2024,1,5,1,2,3).Format") == "dd-MMM-yyyy HH:mm:ss"
    assert m.str("char(datetime(2024,1,5))") == "05-Jan-2024"
    assert m.str("char(datetime('2024-01-05 07:08:09'))") == "05-Jan-2024 07:08:09"
    assert m.str("char(datetime(2024,1,5,'Format','yyyy.MM.dd'))") == "2024.01.05"
    # seconds are cut, never rounded up
    assert m.str("char(datetime(2024,1,5,23,59,59.9))") == "05-Jan-2024 23:59:59"


def test_text_conversions(m):
    m.run("t = datetime(2024,1,[5 6]); s = string(t); c = cellstr(t); ch = char(t'); n = char(t(1), 'yyyy/MM/dd');")
    assert list(m.val("s").data.flat) == ["05-Jan-2024", "06-Jan-2024"]
    assert texts(m, "t") == ["05-Jan-2024", "06-Jan-2024"]
    assert m.val("ch").shape == (2, 11)
    assert m.str("n") == "2024/01/05"
    assert m.run("fprintf('%s\\n', string(t(1)));") == "05-Jan-2024\n"


def test_display(m):
    assert m.run("t = datetime(2024,1,5)") == "t = \n\n  datetime\n\n   05-Jan-2024\n\n"
    assert m.run("t = datetime(2024,1,5:6)") == \
        "t = \n\n  1×2 datetime array\n\n   05-Jan-2024   06-Jan-2024\n\n"
    assert m.run("t = [datetime(2024,1,5,1,2,3); NaT]") == \
        "t = \n\n  2×1 datetime array\n\n   05-Jan-2024 01:02:03\n   NaT\n\n"
    assert m.run("disp(datetime(2024,1,5))") == "   05-Jan-2024\n"
    assert "0×0 empty datetime array" in m.run("t = datetime.empty")
    assert "{[05-Jan-2024]}" in m.run("c = {datetime(2024,1,5)}")
    assert "when: 05-Jan-2024" in m.run("s.when = datetime(2024,1,5)")
    assert m.str("class(datetime(2024,1,5))") == "datetime"
    assert "1x2" in m.run("t = datetime(2024,1,5:6); whos t") and "datetime" in m.run("whos t")


# -- arithmetic and comparison --------------------------------------------------------------------------------

def test_datetime_arithmetic(m):
    m.run("a = datetime(2024,1,31); b = datetime(2024,3,1,12,0,0); d = b - a;")
    assert isinstance(m.val("d"), DurationArray)
    assert secs(m, "d").item() == 30.5 * 86400
    assert secs(m, "a + days(1)").item() == stamp(2024, 2, 1)
    assert secs(m, "hours(36) + a").item() == stamp(2024, 2, 1, 12)
    assert secs(m, "a - minutes(1)").item() == stamp(2024, 1, 30, 23, 59)
    assert secs(m, "a + 1").item() == stamp(2024, 2, 1)  # numbers count as days
    np.testing.assert_array_equal(secs(m, "a + hours(0:12:24)"),
                                  [[stamp(2024, 1, 31), stamp(2024, 1, 31, 12), stamp(2024, 2, 1)]])
    assert secs(m, "b - '2024-03-01'").item() == 12 * 3600
    assert m.num("seconds(datetime(2024,1,2) - datetime(2024,1,1))").item() == 86400
    for code in ("a + b", "5 - a", "a * 2", "sin(a)"):
        with pytest.raises(MatlabError):
            m.run(code + ";")


def test_calendar_durations(m):
    m.run("a = datetime(2024,1,31,8,0,0);")
    assert isinstance(m.interp.eval_expression("calmonths(1)"), CalendarDuration)
    np.testing.assert_array_equal(
        secs(m, "a + calmonths(1:3)"),
        [[stamp(2024, 2, 29, 8), stamp(2024, 3, 31, 8), stamp(2024, 4, 30, 8)]])
    assert secs(m, "a + calyears(1)").item() == stamp(2025, 1, 31, 8)
    assert secs(m, "a - calyears(1) + caldays(3)").item() == stamp(2023, 2, 3, 8)
    assert secs(m, "a + calweeks(2)").item() == stamp(2024, 2, 14, 8)
    assert secs(m, "a + calquarters(1)").item() == stamp(2024, 4, 30, 8)
    assert secs(m, "a + (calmonths(1) + caldays(1))").item() == stamp(2024, 3, 1, 8)
    assert secs(m, "a + 2*calmonths(1)").item() == stamp(2024, 3, 31, 8)
    assert "1y 2mo 2d" in m.run("c = calmonths(14) + caldays(2)")
    assert "-1mo" in m.run("c = -calmonths(1)")
    m.run("c = between(datetime(2020,1,15), datetime(2021,3,20)); e = datetime(2020,1,15) + c;")
    assert "1y 2mo 5d" in m.run("c")
    assert secs(m, "e").item() == stamp(2021, 3, 20)
    assert bool(m.num("iscalendarduration(c)").item())
    with pytest.raises(MatlabError):
        m.run("calmonths(1.5);")


def test_comparisons(m):
    m.run("t = datetime(2024,1,1:3);")
    np.testing.assert_array_equal(m.num("t > datetime(2024,1,1)"), [[False, True, True]])
    np.testing.assert_array_equal(m.num("t == '2024-01-02'"), [[False, True, False]])
    np.testing.assert_array_equal(m.num("'2024-01-02' <= t"), [[False, True, True]])
    np.testing.assert_array_equal(m.num("t ~= t(2)"), [[True, False, True]])
    np.testing.assert_array_equal(m.num("isbetween(t, '2024-01-02', datetime(2024,1,5))"), [[False, True, True]])
    np.testing.assert_array_equal(m.num("isbetween(t, t(1), t(3), 'open')"), [[False, True, False]])
    assert bool(m.num("isequal(t, datetime(2024,1,1:3))").item())
    assert not bool(m.num("isequal(t, datetime(2024,1,1:2))").item())
    assert not bool(m.num("isequal(t(1), 5)").item())
    assert m.run("if t(2) > t(1), disp('later'), end") == "later\n"
    with pytest.raises(MatlabError):
        m.run("t < 5;")


def test_nat(m):
    m.run("n = NaT; a = [n == n, n ~= n, n < datetime(2024,1,1), isnat(n)]; z = NaT(2,3);")
    np.testing.assert_array_equal(m.num("a"), [[False, True, False, True]])
    assert m.val("z").data.shape == (2, 3)
    np.testing.assert_array_equal(m.num("isnat([datetime(2024,1,1) NaT])"), [[False, True]])
    np.testing.assert_array_equal(m.num("ismissing([datetime(2024,1,1) NaT])"), [[False, True]])
    assert m.str("char(NaT)") == "NaT"
    assert np.isnan(m.num("year(NaT)").item())
    assert bool(m.num("isnat(NaT + days(1))").item())
    assert bool(m.num("isdatetime(NaT)").item()) and not bool(m.num("isdatetime(5)").item())


# -- indexing, concatenation, ranges ---------------------------------------------------------------------------

def test_indexing_and_concatenation(m):
    m.run("t = datetime(2024,1,1:3); a = t(2); b = t(end); c = t([true false true]); "
          "u = [t datetime(2025,1,1)]; v = [t'; datetime(2025,1,1)]; w = [t, '2030-06-07'];")
    assert secs(m, "a").item() == stamp(2024, 1, 2) and secs(m, "b").item() == stamp(2024, 1, 3)
    assert m.val("c").data.shape == (1, 2) and m.val("u").data.shape == (1, 4) and m.val("v").data.shape == (4, 1)
    assert secs(m, "w(end)").item() == stamp(2030, 6, 7)
    np.testing.assert_array_equal(m.num("[numel(t) size(t) isempty(t) isrow(t) length(t)]"), [[3, 1, 3, 0, 1, 3]])
    m.run("t(5) = datetime(2024,2,1); t(1) = []; t(1) = '2030-01-01'; t(2) = NaT;")
    v = secs(m, "t")
    assert v.shape == (1, 4) and v[0, 0] == stamp(2030, 1, 1) and np.isnan(v[0, 1]) and np.isnan(v[0, 2])
    assert v[0, 3] == stamp(2024, 2, 1)
    m.run("r = reshape(datetime(2024,1,1:4), 2, 2); p = r(2,1); q = r(:,2)'; s = repmat(r(1), 2, 3);")
    assert secs(m, "p").item() == stamp(2024, 1, 2)
    np.testing.assert_array_equal(secs(m, "q"), [[stamp(2024, 1, 3), stamp(2024, 1, 4)]])
    assert m.val("s").data.shape == (2, 3)
    for code in ("t(2) = 5;", "t{1}", "[t hours(1)]", "t(9)"):
        with pytest.raises(MatlabError):
            m.run(code)


def test_loops_over_elements(m):
    assert m.run("for k = datetime(2024,1,1:3), fprintf('%d ', k.Day); end") == "1 2 3 "
    assert m.run("for k = hours(1:2), fprintf('%g ', minutes(k)); end") == "60 120 "


def test_ranges(m):
    np.testing.assert_array_equal(secs(m, "colon(datetime(2024,1,1), datetime(2024,1,4))"),
                                  [[stamp(2024, 1, d) for d in (1, 2, 3, 4)]])
    np.testing.assert_array_equal(secs(m, "colon(datetime(2024,1,1), days(10), datetime(2024,2,1))"),
                                  [[stamp(2024, 1, 1), stamp(2024, 1, 11), stamp(2024, 1, 21), stamp(2024, 1, 31)]])
    np.testing.assert_array_equal(secs(m, "colon(datetime(2024,1,31), calmonths(1), datetime(2024,5,1))"),
                                  [[stamp(2024, 1, 31), stamp(2024, 2, 29), stamp(2024, 3, 31), stamp(2024, 4, 30)]])
    np.testing.assert_array_equal(secs(m, "colon(datetime(2024,1,2), -hours(12), datetime(2024,1,1))"),
                                  [[stamp(2024, 1, 2), stamp(2024, 1, 1, 12), stamp(2024, 1, 1)]])
    np.testing.assert_array_equal(secs(m, "colon(hours(1), minutes(30), hours(2))"), [[3600, 5400, 7200]])
    m.run("x = colon(datetime(2024,1,2), datetime(2024,1,1));")
    assert m.val("x").data.shape == (1, 0)
    np.testing.assert_array_equal(secs(m, "linspace(datetime(2024,1,1), datetime(2024,1,2), 3)"),
                                  [[stamp(2024, 1, 1), stamp(2024, 1, 1, 12), stamp(2024, 1, 2)]])
    np.testing.assert_array_equal(secs(m, "datetime(2024,1,1) + days(0:2)"),
                                  [[stamp(2024, 1, d) for d in (1, 2, 3)]])


def test_colon_operator_on_datetimes(m):
    """``t1:step:t2`` needs a hook in the interpreter's range evaluation; until then the error says what to do."""
    try:
        m.run("r = datetime(2024,1,1):days(1):datetime(2024,1,4);")
    except MatlabError as e:
        assert "colon(" in str(e)
    else:
        np.testing.assert_array_equal(secs(m, "r"), [[stamp(2024, 1, d) for d in (1, 2, 3, 4)]])


# -- array functions ----------------------------------------------------------------------------------------------

def test_sort_min_max_unique_diff_mean(m):
    m.run("t = datetime(2024,1,[5 2 9 2]); [s, i] = sort(t); d = sort(t, 'descend'); [lo, k] = min(t); hi = max(t); "
          "u = unique(t); df = diff(datetime(2024,1,[1 3 4])); mu = mean(datetime(2024,1,[1 3])); "
          "md = median(datetime(2024,1,[1 2 9]));")
    np.testing.assert_array_equal(secs(m, "s"), [[stamp(2024, 1, d) for d in (2, 2, 5, 9)]])
    np.testing.assert_array_equal(m.num("i"), [[2, 4, 1, 3]])
    np.testing.assert_array_equal(secs(m, "d"), [[stamp(2024, 1, d) for d in (9, 5, 2, 2)]])
    assert secs(m, "lo").item() == stamp(2024, 1, 2) and m.num("k").item() == 2
    assert secs(m, "hi").item() == stamp(2024, 1, 9)
    np.testing.assert_array_equal(secs(m, "u"), [[stamp(2024, 1, d) for d in (2, 5, 9)]])
    assert isinstance(m.val("df"), DurationArray)
    np.testing.assert_array_equal(secs(m, "df"), [[2 * 86400, 86400]])
    assert secs(m, "mu").item() == stamp(2024, 1, 2) and secs(m, "md").item() == stamp(2024, 1, 2)
    assert isinstance(m.val("s"), DatetimeArray) and isinstance(m.val("u"), DatetimeArray)
    np.testing.assert_array_equal(m.num("ismember(datetime(2024,1,[2 3]), t)"), [[True, False]])
    assert bool(m.num("issorted(s)").item())
    # NaT sorts last and is skipped by min / max
    m.run("w = sort([NaT datetime(2024,1,2) datetime(2024,1,1)]); x = max([NaT datetime(2024,1,2)]);")
    assert np.isnan(secs(m, "w")[0, 2]) and secs(m, "x").item() == stamp(2024, 1, 2)


def test_component_functions(m):
    m.run("t = datetime(2024,3,15,10,30,45.5);")
    for fn, want in (("year", 2024), ("month", 3), ("day", 15), ("hour", 10), ("minute", 30), ("second", 45.5),
                     ("quarter", 1), ("week", 11)):
        assert m.num(f"{fn}(t)").item() == want
    assert m.num("day(t,'dayofyear')").item() == 75
    assert m.num("day(t,'dayofweek')").item() == 6
    assert texts(m, "month(t,'name')") == ["March"] and texts(m, "month(t,'shortname')") == ["Mar"]
    assert texts(m, "day(t,'name')") == ["Friday"]
    m.run("[y, mo, d] = ymd(t); [h, mi, s] = hms(t); [n, name] = weekday(t); [~, long] = weekday(t, 'long');")
    got = [m.num(v).item() for v in ("y", "mo", "d", "h", "mi", "s")]
    np.testing.assert_array_equal(got, [2024, 3, 15, 10, 30, 45.5])
    assert m.num("n").item() == 6 and m.str("name") == "Fri" and m.str("long") == "Friday"
    assert secs(m, "timeofday(t)").item() == 10 * 3600 + 30 * 60 + 45.5
    np.testing.assert_array_equal(m.num("year(datetime(2020,1,1) + calyears(0:2))"), [[2020, 2021, 2022]])
    assert m.num("week(datetime(2024,1,7))").item() == 2 and m.num("week(datetime(2024,12,31))").item() == 53


@pytest.mark.parametrize("call, parts", [
    ("dateshift(t,'start','day')", (2024, 3, 15)), ("dateshift(t,'end','day')", (2024, 3, 16)),
    ("dateshift(t,'start','month')", (2024, 3, 1)), ("dateshift(t,'end','month')", (2024, 3, 31)),
    ("dateshift(t,'start','year')", (2024, 1, 1)), ("dateshift(t,'end','year')", (2024, 12, 31)),
    ("dateshift(t,'start','hour')", (2024, 3, 15, 10)), ("dateshift(t,'end','hour')", (2024, 3, 15, 11)),
    ("dateshift(t,'start','week')", (2024, 3, 10)), ("dateshift(t,'end','week')", (2024, 3, 16)),
    ("dateshift(t,'start','quarter')", (2024, 1, 1)), ("dateshift(t,'start','minute')", (2024, 3, 15, 10, 30)),
    ("dateshift(t,'start','month','next')", (2024, 4, 1)), ("dateshift(t,'start','month','previous')", (2024, 2, 1)),
    ("dateshift(t,'end','month',2)", (2024, 5, 31)), ("dateshift(t,'start','day','nearest')", (2024, 3, 15)),
    ("dateshift(t,'dayofweek','Monday')", (2024, 3, 18, 10, 30, 45)),
    ("dateshift(t,'dayofweek','Monday','previous')", (2024, 3, 11, 10, 30, 45)),
    ("dateshift(t,'dayofweek',6)", (2024, 3, 15, 10, 30, 45)),
])
def test_dateshift(m, call, parts):
    m.run("t = datetime(2024,3,15,10,30,45);")
    assert secs(m, call).item() == stamp(*parts)


# -- durations -----------------------------------------------------------------------------------------------------

def test_duration_units(m):
    for fn, unit in (("seconds", 1), ("minutes", 60), ("hours", 3600), ("days", 86400),
                     ("years", 365.2425 * 86400), ("milliseconds", 1e-3)):
        assert isinstance(m.interp.eval_expression(f"{fn}(2)"), DurationArray)
        assert secs(m, f"{fn}(2)").item() == pytest.approx(2 * unit)
        assert m.num(f"{fn}({fn}(2.5))").item() == pytest.approx(2.5)
    assert m.num("minutes(hours(1.5))").item() == 90
    assert m.num("hours(days(1) + minutes(30))").item() == 24.5
    np.testing.assert_array_equal(secs(m, "hours([1 2; 3 4])"), 3600 * np.array([[1, 2], [3, 4]]))
    assert secs(m, "duration(1,30,15)").item() == 5415
    assert secs(m, "duration(1,30,15,500)").item() == 5415.5
    np.testing.assert_array_equal(secs(m, "duration([1 2 3; 4 5 6])"), [[3723], [14706]])
    assert secs(m, "duration('01:30:15')").item() == 5415
    assert secs(m, "duration('2:12:00:00')").item() == 2.5 * 86400
    assert bool(m.num("isduration(hours(1))").item()) and not bool(m.num("isduration(1)").item())
    with pytest.raises(MatlabError):
        m.run("hours('x');")


def test_duration_arithmetic(m):
    m.run("h = hours(3);")
    assert secs(m, "h + minutes(30)").item() == 12600
    assert secs(m, "h - hours(5)").item() == -7200
    assert secs(m, "-h").item() == -10800
    assert secs(m, "2*h").item() == 21600 and secs(m, "h*2").item() == 21600 and secs(m, "h/2").item() == 5400
    np.testing.assert_array_equal(secs(m, "h .* [1 2]"), [[10800, 21600]])
    assert m.num("h / minutes(30)").item() == 6
    assert secs(m, "h + 1").item() == 10800 + 86400  # numbers count as days
    assert secs(m, "abs(-h)").item() == 10800
    assert secs(m, "sum([h h])").item() == 21600
    np.testing.assert_array_equal(secs(m, "cumsum(hours([1 2 3]))"), [[3600, 10800, 21600]])
    assert secs(m, "mean([hours(1) hours(3)])").item() == 7200
    assert secs(m, "max([hours(1) minutes(90)])").item() == 5400
    np.testing.assert_array_equal(secs(m, "sort([hours(3) hours(1) hours(2)])"), [[3600, 7200, 10800]])
    assert secs(m, "round(minutes(1.6), 'minutes')").item() == 120
    assert secs(m, "floor(seconds(2.7))").item() == 2
    np.testing.assert_array_equal(m.num("[hours(1) < minutes(61), hours(1) == minutes(60), hours(2) <= hours(1)]"),
                                  [[True, True, False]])
    np.testing.assert_array_equal(m.num("hours(1) == '01:00:00'"), [[True]])
    m.run("d = duration(0,0,0); d(3) = minutes(1); e = [hours(1); minutes(1)]; e(1) = [];")
    np.testing.assert_array_equal(secs(m, "d"), [[0, 0, 60]])
    assert secs(m, "e").item() == 60
    for code in ("h * h", "2 / h", "h + datetime(2024,1,1) + datetime(2024,1,1)"):
        with pytest.raises(MatlabError):
            m.run(code + ";")


@pytest.mark.parametrize("expr, want", [
    ("hours(1.5)", "1.5 hr"), ("minutes(90)", "90 min"), ("seconds(1)", "1 sec"), ("days(2)", "2 days"),
    ("days(1)", "1 day"), ("years(1)", "1 yr"), ("years(2)", "2 yrs"), ("milliseconds(1500)", "1.5 sec"),
    ("duration(1,30,15)", "01:30:15"), ("hours(30) + duration(0,0,0)", "30 hr"), ("duration(30,0,0)", "30:00:00"),
    ("-duration(1,2,3)", "-01:02:03"), ("duration(25,30,0,'Format','dd:hh:mm:ss')", "01:01:30:00"),
    ("duration(0,0,3.14159,'Format','hh:mm:ss.SSS')", "00:00:03.142"), ("duration(0,90,5,'Format','mm:ss')", "90:05"),
    ("duration(1,30,0,'Format','h')", "1.5 hr"), ("hours(1)/3", "0.33333 hr"), ("seconds(NaN)", "NaN sec"),
])
def test_duration_display(m, expr, want):
    assert m.str(f"char({expr})") == want


def test_duration_format_property(m):
    m.run("d = [hours(1) minutes(30)]; d.Format = 'hh:mm:ss'; f = d.Format;")
    assert m.str("f") == "hh:mm:ss"
    assert m.run("d") == "d = \n\n  1×2 duration array\n\n   01:00:00   00:30:00\n\n"
    assert m.run("x = hours(2)") == "x = \n\n  duration\n\n   2 hr\n\n"
    with pytest.raises(MatlabError):
        m.run("d.Format = 'bogus';")


# -- serial date numbers -----------------------------------------------------------------------------

def test_datenum(m):
    assert m.num("datenum(2024,3,15)").item() == 739326
    assert m.num("datenum(2024,3,15,12,0,0)").item() == 739326.5
    assert m.num("datenum([2020 2 29])").item() == 737850
    np.testing.assert_array_equal(m.num("datenum([2020 2 29 12 0 0; 2021 1 1 0 0 0])"), [[737850.5], [738157]])
    assert m.num("datenum('2024-03-15')").item() == 739326
    assert m.num("datenum('15-Mar-2024 12:00:00')").item() == 739326.5
    assert m.num("datenum('15/03/2024', 'dd/mm/yyyy')").item() == 739326
    assert m.num("datenum(datetime(2024,3,15))").item() == 739326
    assert m.num("datenum(0,1,1)").item() == 1 and m.num("datenum(1,1,1)").item() == 367
    assert m.num("datenum(2024,14,1)").item() == m.num("datenum(2025,2,1)").item()
    assert m.num("datenum(2024,1,1.5)").item() == m.num("datenum(2024,1,1)").item() + 0.5
    assert m.num("datenum(739326)").item() == 739326
    assert abs(m.num("now").item() - (pydt.datetime.now().toordinal() + 366)) < 1.01
    assert m.num("floor(now) == today").item() == 1
    c = m.num("clock").ravel()
    assert c.size == 6 and c[0] >= 2024 and 1 <= c[1] <= 12
    assert m.num("floor(now) == datenum(date)").item() == 1


@pytest.mark.parametrize("fmt, want", [
    ("0", "28-Jul-2020 18:05:09"), ("1", "28-Jul-2020"), ("2", "07/28/20"), ("3", "Jul"), ("5", "07"), ("6", "07/28"),
    ("7", "28"), ("8", "Tue"), ("10", "2020"), ("11", "20"), ("12", "Jul20"), ("13", "18:05:09"),
    ("14", " 6:05:09 PM"), ("15", "18:05"), ("16", " 6:05 PM"), ("17", "Q3-20"), ("19", "28/07"), ("20", "28/07/20"),
    ("21", "Jul.28,2020 18:05:09"), ("22", "Jul.28,2020"), ("23", "07/28/2020"), ("24", "28/07/2020"),
    ("25", "20/07/28"), ("26", "2020/07/28"), ("28", "Jul2020"), ("29", "2020-07-28"), ("30", "20200728T180509"),
    ("31", "2020-07-28 18:05:09"), ("'yyyy/mm/dd HH:MM:SS'", "2020/07/28 18:05:09"),
    ("'dddd, mmmm dd, yyyy'", "Tuesday, July 28, 2020"), ("'HH:MM PM'", " 6:05 PM"), ("'HH:MM:SS.FFF'", "18:05:09.000"),
])
def test_datestr_formats(m, fmt, want):
    assert m.str(f"datestr(datenum(2020,7,28,18,5,9), {fmt})") == want


def test_datestr_inputs(m):
    assert m.str("datestr(738000)") == "28-Jul-2020"
    assert m.str("datestr(738000.5)") == "28-Jul-2020 12:00:00"
    assert m.str("datestr([2021 5 6 7 8 9])") == "06-May-2021 07:08:09"
    assert m.str("datestr(datetime(2024,3,15,10,30,0))") == "15-Mar-2024 10:30:00"
    assert m.str("datestr('2024-03-15', 1)") == "15-Mar-2024"
    m.run("r = datestr([738000; 738001.5]);")
    rows = m.val("r")
    assert rows.shape == (2, 20) and "".join(rows[1]) == "29-Jul-2020 12:00:00"
    assert len(m.str("datestr(now)")) == 20
    assert m.str("datestr(datenum(2024,12,31,23,59,59.9))") == "31-Dec-2024 23:59:59"
    with pytest.raises(MatlabError):
        m.run("datestr(738000, 99);")


def test_datevec_weekday_eomday_addtodate(m):
    np.testing.assert_array_equal(m.num("datevec(738000.75)"), [[2020, 7, 28, 18, 0, 0]])
    np.testing.assert_array_equal(m.num("datevec('2024-03-15 10:30:45')"), [[2024, 3, 15, 10, 30, 45]])
    np.testing.assert_array_equal(m.num("datevec(datetime(2024,3,15))"), [[2024, 3, 15, 0, 0, 0]])
    m.run("[y, mo, d, h, mi, s] = datevec(datenum(2024,3,15,10,30,45));")
    np.testing.assert_allclose([m.num(v).item() for v in ("y", "mo", "d", "h", "mi", "s")], [2024, 3, 15, 10, 30, 45])
    np.testing.assert_array_equal(m.num("datevec([738000; 738001])")[:, :3], [[2020, 7, 28], [2020, 7, 29]])
    np.testing.assert_array_equal(m.num("weekday(738001:738007)"), [[4, 5, 6, 7, 1, 2, 3]])
    assert m.num("weekday('2024-03-15')").item() == 6
    m.run("[n, s] = weekday(738000);")
    assert m.num("n").item() == 3 and m.str("s") == "Tue"
    np.testing.assert_array_equal(m.num("eomday(2023, 1:12)"), [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]])
    np.testing.assert_array_equal(m.num("[eomday(2024,2) eomday(1900,2) eomday(2000,2)]"), [[29, 28, 29]])
    assert m.num("addtodate(738000, 1, 'month')").item() == 738031
    assert m.num("addtodate(738000, -2, 'year')").item() == 737269
    assert m.num("addtodate(738000, 36, 'hour')").item() == 738001.5
    assert m.num("addtodate(datenum(2024,1,31), 1, 'month')").item() == m.num("datenum(2024,2,29)").item()
    np.testing.assert_array_equal(m.num("[year(738000) month(738000) day(738000)]"), [[2020, 7, 28]])
    with pytest.raises(MatlabError):
        m.run("addtodate(738000, 1, 'fortnight');")


def test_calendar(m):
    cal = m.num("calendar(2024, 2)")
    assert cal.shape == (6, 7)
    np.testing.assert_array_equal(cal[0], [0, 0, 0, 0, 1, 2, 3])
    assert cal.max() == 29 and np.count_nonzero(cal) == 29
    out = m.run("calendar(2024, 2)")
    assert "Feb 2024" in out and "Tu" in out
    np.testing.assert_array_equal(m.num("calendar(datetime(2024,2,10))"), cal)


def test_elapsed_time_helpers_still_work(m):
    m.run("t0 = clock; e = etime(clock, t0); tic; x = toc;")
    assert 0 <= m.num("e").item() < 5 and 0 <= m.num("x").item() < 5


# -- tables and plotting ----------------------------------------------------------------------------------

def test_table_with_datetime_and_duration_columns(m):
    out = m.run("d = datetime(2024,1,[3;1;2]); v = [10;20;30]; el = hours([1;2;3]); T = table(d, v, el)")
    assert "03-Jan-2024    10    1 hr" in out
    m.run("S = sortrows(T, 'd'); a = S.v'; b = T(T.d > datetime(2024,1,1), :); h = height(b); c = T.d(2); "
          "dd = T.d.Day'; T.d(2) = datetime(2030,5,5); e = T.d(2); U = [T; S]; hu = height(U);")
    np.testing.assert_array_equal(m.num("a"), [[20, 30, 10]])
    assert m.num("h").item() == 2 and m.num("hu").item() == 6
    assert secs(m, "c").item() == stamp(2024, 1, 1) and secs(m, "e").item() == stamp(2030, 5, 5)
    np.testing.assert_array_equal(m.num("dd"), [[3, 1, 2]])
    m.run("d = datetime(2024,1,[1;1;2]); v = [1;2;3]; G = groupsummary(table(d, v), 'd', 'sum', 'v'); g = G.sum_v';")
    np.testing.assert_array_equal(m.num("g"), [[3, 3]])


def test_plotting_accepts_datetimes(m):
    m.run("plot(datetime(2024,1,1:3), [1 2 3]); ax = gca; n = numel(ax.Children); close all")
    assert m.num("n").item() == 1
    m.run("plot(hours(1:3), [1 2 3]); close all")
