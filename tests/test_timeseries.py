"""timeseries objects: construction, properties, methods, statistics, arithmetic, plots and MAT-files.

The expected values below were worked out by hand from small inputs.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def col(m, expr):
    return np.asarray(m.num(expr), dtype=float).reshape(-1)


# -- construction ------------------------------------------------------------------------------------------

def test_default_times_and_properties(m):
    m.run("ts = timeseries([4; 8; 15; 16]);")
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 1, 2, 3])
    np.testing.assert_array_equal(m.num("size(ts.Time)"), [[4, 1]])
    np.testing.assert_array_equal(col(m, "ts.Data"), [4, 8, 15, 16])
    assert m.num("ts.Length") == 4
    assert m.str("ts.Name") == "unnamed"
    assert m.num("ts.IsTimeFirst").item() is True or m.num("ts.IsTimeFirst").item() == 1
    assert m.str("ts.TimeInfo.Units") == "seconds"
    assert m.str("class(ts)") == "timeseries"
    assert m.num("isobject(ts) && isa(ts, 'timeseries')")


def test_empty_and_named_empty(m):
    m.run("a = timeseries(); b = timeseries('speed');")
    assert m.num("isempty(a) && isempty(b)")
    assert m.num("a.Length") == 0 and m.num("length(b)") == 0
    assert m.str("b.Name") == "speed" and m.str("a.Name") == "unnamed"


def test_name_as_option_or_last_argument(m):
    m.run("a = timeseries([1; 2], [0 1], 'Name', 'alpha'); b = timeseries([1; 2], [0 1], 'beta');"
          "c = timeseries([1; 2], 'Name', 'gamma');")
    assert m.str("a.Name") == "alpha" and m.str("b.Name") == "beta" and m.str("c.Name") == "gamma"
    np.testing.assert_array_equal(col(m, "c.Time"), [0, 1])


def test_samples_are_sorted_by_time(m):
    m.run("ts = timeseries([30; 10; 20], [2 0 1], [3; 1; 2]);")
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 1, 2])
    np.testing.assert_array_equal(col(m, "ts.Data"), [10, 20, 30])
    np.testing.assert_array_equal(col(m, "double(ts.Quality)"), [1, 2, 3])
    assert m.str("class(ts.Quality)") == "int8"


def test_row_vector_runs_along_the_last_dimension(m):
    m.run("ts = timeseries([4 5 6]);")
    np.testing.assert_array_equal(m.num("size(ts.Data)"), [[1, 1, 3]])
    assert not m.num("ts.IsTimeFirst")
    np.testing.assert_array_equal(col(m, "squeeze(ts.Data)"), [4, 5, 6])
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 1, 2])


def test_matrix_whose_columns_match_the_times(m):
    m.run("ts = timeseries([1 2 3; 4 5 6], [0 1 2]);")
    np.testing.assert_array_equal(m.num("size(ts.Data)"), [[2, 1, 3]])
    np.testing.assert_array_equal(col(m, "ts.Data(:, :, 2)"), [2, 5])
    np.testing.assert_array_equal(m.num("getdatasamplesize(ts)"), [[2, 1]])


def test_matrix_rows_are_samples(m):
    m.run("ts = timeseries([1 2; 3 4; 5 6], [0 1 2]);")
    assert m.num("ts.IsTimeFirst") and m.num("ts.Length") == 3
    np.testing.assert_array_equal(m.num("getdatasamplesize(ts)"), [[1, 2]])


def test_three_dimensional_data(m):
    m.run("x = reshape(1:24, 2, 3, 4); ts = timeseries(x);")
    assert m.num("ts.Length") == 4 and not m.num("ts.IsTimeFirst")
    np.testing.assert_array_equal(m.num("size(ts.Data)"), [[2, 3, 4]])
    np.testing.assert_array_equal(m.num("mean(ts)"), np.mean(np.arange(1, 25).reshape(2, 3, 4, order="F"), axis=2))


def test_data_and_times_that_do_not_fit(m):
    with pytest.raises(MatlabError):
        m.run("timeseries(ones(3, 3, 2), 0:2)")
    with pytest.raises(MatlabError):
        m.run("timeseries([1; 2; 3], [0 1])")
    with pytest.raises(MatlabError):
        m.run("timeseries({1, 2})")


def test_quality_per_value(m):
    m.run("ts = timeseries([1 2; 3 4], [0 1], [0 1; 2 3]);")
    np.testing.assert_array_equal(m.num("double(ts.Quality)"), [[0, 1], [2, 3]])
    with pytest.raises(MatlabError):
        m.run("timeseries([1; 2], [0 1], [1 2 3])")
    with pytest.raises(MatlabError):
        m.run("timeseries([1; 2], [0 1], [500; 0])")


def test_times_given_as_dates(m):
    m.run("ts = timeseries([10; 20; 30], {'01-Jan-2020', '02-Jan-2020', '04-Jan-2020'});")
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 1, 3])
    assert m.str("ts.TimeInfo.Units") == "days"
    assert m.str("ts.TimeInfo.StartDate") == "01-Jan-2020 00:00:00"
    m.run("d = getabstime(ts);")
    assert m.str("d{3}") == "04-Jan-2020 00:00:00"
    m.run("w = timeseries([1; 2], datetime(2021, 5, 1) + hours([0 6]));")
    np.testing.assert_allclose(col(m, "w.Time"), [0, 0.25])
    assert m.str("w.TimeInfo.StartDate") == "01-May-2021 00:00:00"


# -- properties ---------------------------------------------------------------------------------------------

def test_property_names_ignore_case(m):
    m.run("ts = timeseries([1; 2; 3], [0 2 4]);")
    np.testing.assert_array_equal(col(m, "ts.data"), [1, 2, 3])
    np.testing.assert_array_equal(col(m, "ts.time"), [0, 2, 4])
    m.run("ts.name = 'lower';")
    assert m.str("ts.Name") == "lower"


def test_setting_data_and_time(m):
    m.run("ts = timeseries([1; 2; 3], [0 1 2]); ts.Data = [7; 8; 9]; ts.Time = [0 5 30];")
    np.testing.assert_array_equal(col(m, "ts.Data"), [7, 8, 9])
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 5, 30])
    m.run("ts.Data(2) = -1;")
    np.testing.assert_array_equal(col(m, "ts.Data"), [7, -1, 9])
    with pytest.raises(MatlabError):
        m.run("ts.Data = [1; 2];")
    with pytest.raises(MatlabError):
        m.run("ts.Time = [1 2];")
    with pytest.raises(MatlabError):
        m.run("ts.Time = [3 2 1];")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("ts.Length = 3;")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("ts.IsTimeFirst = false;")


def test_setting_data_on_an_empty_timeseries(m):
    m.run("ts = timeseries(); ts.Data = [5; 6];")
    np.testing.assert_array_equal(col(m, "ts.Time"), [0, 1])


def test_time_info(m):
    m.run("ts = timeseries([1; 2; 3], [0 10 20]); ti = ts.TimeInfo;")
    assert m.num("ti.Start") == 0 and m.num("ti.End") == 20 and m.num("ti.Increment") == 10
    assert m.num("ti.Length") == 3 and m.str("class(ti)") == "tsdata.timemetadata"
    m.run("ts.TimeInfo.Units = 'minutes'; ts.TimeInfo.Start = 100;")
    assert m.str("ts.TimeInfo.Units") == "minutes"
    np.testing.assert_array_equal(col(m, "ts.Time"), [100, 110, 120])
    m.run("ts.TimeInfo.Increment = 2;")
    np.testing.assert_array_equal(col(m, "ts.Time"), [100, 102, 104])
    m.run("u = timeseries([1; 2; 3], [0 1 3]);")
    assert np.isnan(m.num("u.TimeInfo.Increment"))
    with pytest.raises(MatlabError):
        m.run("ts.TimeInfo.Units = 'fortnights';")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("ts.TimeInfo.End = 3;")


def test_start_date_property(m):
    m.run("ts = timeseries([2; 4; 6], [0 10 20]); ts.TimeInfo.StartDate = '15-Mar-2021 08:00:00';"
          "d = getabstime(ts);")
    assert m.str("d{2}") == "15-Mar-2021 08:00:10"
    m.run("g = getsampleusingtime(ts, '15-Mar-2021 08:00:20');")
    np.testing.assert_array_equal(col(m, "g.Data"), [6])


def test_data_info_and_interpolation(m):
    m.run("ts = timeseries([1; 2]); ts.DataInfo.Units = 'm/s';")
    assert m.str("ts.DataInfo.Units") == "m/s" and m.str("getinterpmethod(ts)") == "linear"
    m.run("ts.DataInfo.Interpolation = tsdata.interpolation('zoh');")
    assert m.str("getinterpmethod(ts)") == "zoh"
    assert m.str("ts.DataInfo.Interpolation.Name") == "zoh"
    assert m.str("class(ts.DataInfo)") == "tsdata.datametadata"


def test_user_data_and_treat_nan(m):
    m.run("ts = timeseries([1; 2]); ts.UserData = struct('a', 3);")
    assert m.num("ts.UserData.a") == 3
    assert m.num("ts.TreatNaNasMissing")


def test_get_and_set(m):
    m.run("ts = timeseries([1; 2], 'Name', 'one'); g = get(ts); p = properties(ts);")
    assert m.num("isstruct(g) && isequal(fieldnames(g), p)")
    assert m.str("get(ts, 'Name')") == "one"
    m.run("set(ts, 'Name', 'two');")
    assert m.str("ts.Name") == "two"
    m.run("t3 = set(ts, 'Name', 'three');")
    assert m.str("t3.Name") == "three" and m.str("ts.Name") == "two"


def test_value_semantics(m):
    m.run("a = timeseries([1; 2]); b = a; b.Name = 'changed';")
    assert m.str("a.Name") == "unnamed"


def test_size_length_isempty(m):
    m.run("ts = timeseries((1:6)');")
    np.testing.assert_array_equal(m.num("size(ts)"), [[6, 1]])
    assert m.num("size(ts, 1)") == 6 and m.num("length(ts)") == 6 and not m.num("isempty(ts)")
    m.run("[r, c] = size(ts);")
    assert m.num("r") == 6 and m.num("c") == 1


def test_properties_fieldnames_isprop(m):
    m.run("ts = timeseries([1; 2]); p = properties(ts); f = fieldnames(ts);")
    expected = ["Events", "Name", "UserData", "Data", "DataInfo", "Time", "TimeInfo", "Quality", "QualityInfo",
                "IsTimeFirst", "TreatNaNasMissing", "Length"]
    assert m.run("disp(strjoin(p', ','))").strip() == ",".join(expected)
    assert m.num("isequal(p, f)")
    assert m.num("isprop(ts, 'TimeInfo')") and not m.num("isprop(ts, 'Foo')")
    assert m.num("isequal(properties('timeseries'), p)")


def test_display(m):
    out = m.run("ts = timeseries([1; 2; 3], 'Name', 'pos')")
    assert "timeseries" in out and "Common Properties:" in out
    assert "Name: 'pos'" in out and "Data: [3×1 double]" in out
    out = m.run("disp(ts)")
    assert "Common Properties:" in out and "pos" in out
    out = m.run("c = {ts}")
    assert "1×1 timeseries" in out


# -- resampling and selecting ----------------------------------------------------------------------------

def test_resample_linear(m):
    m.run("ts = timeseries([0; 10; 20; 30], 0:3); r = resample(ts, [0.5 2.25]);")
    np.testing.assert_allclose(col(m, "r.Data"), [5, 22.5])
    np.testing.assert_allclose(col(m, "r.Time"), [0.5, 2.25])


def test_resample_outside_the_range_gives_nan_and_warns(m):
    out = m.run("ts = timeseries([0; 10], [0 1]); r = resample(ts, [-1 0.5 2]);")
    assert "Warning" in out and m.interp.last_warning_id == "interpolation:interpolation:noextrap"
    v = col(m, "r.Data")
    assert np.isnan(v[0]) and v[1] == 5 and np.isnan(v[2])
    out = m.run("warning('off', 'interpolation:interpolation:noextrap'); r = resample(ts, [2 3]);")
    assert "Warning" not in out


def test_resample_zero_order_hold(m):
    m.run("ts = timeseries([0; 10; 20; 30], 0:3); z = setinterpmethod(ts, 'zoh'); a = resample(z, [0.5 2.25 3]);"
          "b = resample(ts, [0.5 2.25], 'zoh');")
    np.testing.assert_array_equal(col(m, "a.Data"), [0, 20, 30])
    np.testing.assert_array_equal(col(m, "b.Data"), [0, 20])


def test_resample_matrix_data_and_quality_code(m):
    m.run("ts = timeseries([0 100; 2 80], [0 2]); r = resample(ts, [1 2], 'linear', 5);")
    np.testing.assert_allclose(m.num("r.Data"), [[1, 90], [2, 80]])
    np.testing.assert_array_equal(col(m, "double(r.Quality)"), [5, 5])


def test_resample_with_a_function_handle(m):
    m.run("ts = timeseries([1; 4; 9], [0 1 2]);"
          "ts = setinterpmethod(ts, @(tn, t, d) interp1(t, d, tn, 'nearest')); r = resample(ts, [0.2 1.7]);")
    np.testing.assert_array_equal(col(m, "r.Data"), [1, 9])


def test_resample_with_dates(m):
    m.run("ts = timeseries([0; 24], {'01-Jan-2020', '02-Jan-2020'}); r = resample(ts, {'01-Jan-2020 06:00:00'});")
    np.testing.assert_allclose(col(m, "r.Data"), [6])
    np.testing.assert_allclose(col(m, "r.Time"), [0.25])


def test_get_samples(m):
    m.run("ts = timeseries([5; 6; 7; 8], [0 1 2 3], [1; 2; 3; 4]); s = getsamples(ts, [2 4]);"
          "d = getdatasamples(ts, logical([1 0 1 0]));")
    np.testing.assert_array_equal(col(m, "s.Time"), [1, 3])
    np.testing.assert_array_equal(col(m, "s.Data"), [6, 8])
    np.testing.assert_array_equal(col(m, "double(s.Quality)"), [2, 4])
    np.testing.assert_array_equal(col(m, "d"), [5, 7])
    m.run("a = getsampleusingtime(ts, 2); b = getsampleusingtime(ts, 0.5, 3); c = getsampleusingtime(ts, 9);")
    np.testing.assert_array_equal(col(m, "a.Data"), [7])
    np.testing.assert_array_equal(col(m, "b.Time"), [1, 2, 3])
    assert m.num("c.Length") == 0


def test_append(m):
    m.run("a = timeseries([1; 2; 3], [0 1 2]); b = timeseries([4; 5], [3 4]); c = append(a, b); d = [a; b];")
    np.testing.assert_array_equal(col(m, "c.Time"), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(col(m, "c.Data"), [1, 2, 3, 4, 5])
    assert m.num("isequal(c, d)")
    with pytest.raises(MatlabError, match="overlap"):
        m.run("append(b, a)")


def test_add_and_delete_samples(m):
    m.run("a = timeseries([1; 2; 3], [0 1 2]); b = addsample(a, 'Data', 9, 'Time', 1.5);")
    np.testing.assert_array_equal(col(m, "b.Time"), [0, 1, 1.5, 2])
    np.testing.assert_array_equal(col(m, "b.Data"), [1, 2, 9, 3])
    m.run("c = addsample(a, 'Data', 7, 'Time', 1, 'OverwriteFlag', true);")
    np.testing.assert_array_equal(col(m, "c.Data"), [1, 7, 3])
    m.run("s.Data = [10; 11]; s.Time = [3 4]; e = addsample(a, s);")
    np.testing.assert_array_equal(col(m, "e.Data"), [1, 2, 3, 10, 11])
    m.run("d1 = delsample(b, 'Index', [1 3]); d2 = delsample(b, 'Value', 2);")
    np.testing.assert_array_equal(col(m, "d1.Time"), [1, 2])
    np.testing.assert_array_equal(col(m, "d2.Time"), [0, 1, 1.5])
    m.run("m2 = timeseries([1 2; 3 4], [0 1]); m3 = addsample(m2, 'Data', [5 6], 'Time', 2);")
    np.testing.assert_array_equal(m.num("m3.Data"), [[1, 2], [3, 4], [5, 6]])


def test_synchronize(m):
    m.run("s1 = timeseries((0:4)' * 2, 0:4); s2 = timeseries((0:2:8)', 0:2:8);"
          "[u1, u2] = synchronize(s1, s2, 'union'); [i1, i2] = synchronize(s1, s2, 'Intersection');"
          "[f1, f2] = synchronize(s1, s2, 'Uniform', 'Interval', 1.5);")
    np.testing.assert_array_equal(col(m, "u1.Time"), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(col(m, "u1.Data"), [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(col(m, "u2.Data"), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(col(m, "i1.Time"), [0, 2, 4])
    np.testing.assert_array_equal(col(m, "i2.Data"), [0, 2, 4])
    np.testing.assert_allclose(col(m, "f1.Time"), [0, 1.5, 3])
    np.testing.assert_allclose(col(m, "f1.Data"), [0, 3, 6])
    with pytest.raises(MatlabError, match="overlap"):
        m.run("synchronize(timeseries([1; 2], [0 1]), timeseries([1; 2], [5 6]), 'union')")


def test_synchronize_converts_units(m):
    m.run("a = timeseries([0; 60], [0 1]); a.TimeInfo.Units = 'minutes';"
          "b = timeseries([0; 30; 60], [0 30 60]); [x, y] = synchronize(a, b, 'union');")
    assert m.str("x.TimeInfo.Units") == "seconds"
    np.testing.assert_array_equal(col(m, "x.Time"), [0, 30, 60])
    np.testing.assert_array_equal(col(m, "x.Data"), [0, 30, 60])


def test_synchronize_with_start_dates(m):
    m.run("a = timeseries([0; 10; 20], [0 1 2]); a.TimeInfo.Units = 'days'; a.TimeInfo.StartDate = '01-Jan-2020';"
          "b = timeseries([5; 6], [0 1]); b.TimeInfo.Units = 'days'; b.TimeInfo.StartDate = '02-Jan-2020';"
          "[x, y] = synchronize(a, b, 'union'); [p, q] = synchronize(a, b, 'union', 'KeepOriginalTimes', true);")
    np.testing.assert_allclose(col(m, "x.Time"), [1, 2])
    np.testing.assert_allclose(col(m, "x.Data"), [10, 20])
    np.testing.assert_allclose(col(m, "y.Data"), [5, 6])
    assert m.str("y.TimeInfo.StartDate") == "01-Jan-2020 00:00:00"
    np.testing.assert_allclose(col(m, "q.Time"), [0, 1])
    assert m.str("q.TimeInfo.StartDate") == "02-Jan-2020 00:00:00"


# -- events -----------------------------------------------------------------------------------------------

def test_events(m):
    m.run("ts = timeseries((1:5)', 0:4); ts = addevent(ts, 'start', 1); ts = addevent(ts, tsdata.event('end', 3));")
    np.testing.assert_array_equal(m.num("size(ts.Events)"), [[1, 2]])
    assert m.str("ts.Events(2).Name") == "end" and m.num("ts.Events(1).Time") == 1
    m.run("a = gettsafterevent(ts, 'start'); b = gettsbeforeatevent(ts, 'end'); c = gettsbetweenevents(ts, "
          "'start', 'end'); d = gettsatevent(ts, 'end');")
    np.testing.assert_array_equal(col(m, "a.Time"), [2, 3, 4])
    np.testing.assert_array_equal(col(m, "b.Time"), [0, 1, 2, 3])
    np.testing.assert_array_equal(col(m, "c.Time"), [1, 2, 3])
    np.testing.assert_array_equal(col(m, "d.Data"), [4])
    m.run("ts = delevent(ts, 'start');")
    np.testing.assert_array_equal(m.num("size(ts.Events)"), [[1, 1]])


# -- statistics -----------------------------------------------------------------------------------------

def test_statistics(m):
    m.run("ts = timeseries([2; 4; 4; 4; 5; 5; 7; 9]);")
    assert m.num("mean(ts)") == 5 and m.num("median(ts)") == 4.5
    assert m.num("max(ts)") == 9 and m.num("min(ts)") == 2 and m.num("sum(ts)") == 40
    np.testing.assert_allclose(m.num("var(ts)"), 32 / 7)
    np.testing.assert_allclose(m.num("std(ts)"), np.sqrt(32 / 7))
    assert m.num("iqr(ts)") == 2


def test_statistics_of_matrix_data(m):
    m.run("ts = timeseries([1 2; 3 4; 5 9], 0:2);")
    np.testing.assert_array_equal(m.num("mean(ts)"), [[3, 5]])
    np.testing.assert_array_equal(m.num("max(ts)"), [[5, 9]])
    np.testing.assert_array_equal(m.num("sum(ts)"), [[9, 15]])


def test_missing_data(m):
    m.run("ts = timeseries([1; NaN; 3; 4], 0:3);")
    np.testing.assert_allclose(m.num("mean(ts)"), 8 / 3)
    np.testing.assert_allclose(m.num("mean(ts, 'MissingData', 'interpolate')"), 2.5)
    m.run("ts.TreatNaNasMissing = false;")
    assert np.isnan(m.num("mean(ts)"))
    m.run("q = timeseries([1; 2; 3; 4], 0:3, [0; 0; -1; 0]);")
    np.testing.assert_allclose(m.num("mean(q, 'Quality', -1)"), 7 / 3)


def test_time_weighting(m):
    m.run("w = timeseries([1; 2; 3], [0 1 3]); u = timeseries([1; 2; 6], [0 1 2]);")
    # weights 1, 1.5 and 2, divided by their mean 1.5
    np.testing.assert_allclose(m.num("mean(w, 'Weighting', 'time')"), (2 / 3 + 2 + 4) / 3)
    np.testing.assert_allclose(m.num("mean(u, 'Weighting', 'time')"), 3)


# -- arithmetic -------------------------------------------------------------------------------------------

def test_arithmetic(m):
    m.run("a = timeseries([1; 2; 3], 0:2); b = timeseries([10; 20; 30], 0:2);")
    np.testing.assert_array_equal(col(m, "getfield(a + b, 'Data')"), [11, 22, 33])
    np.testing.assert_array_equal(col(m, "getfield(b - a, 'Data')"), [9, 18, 27])
    np.testing.assert_array_equal(col(m, "getfield(a .* b, 'Data')"), [10, 40, 90])
    np.testing.assert_array_equal(col(m, "getfield(b ./ a, 'Data')"), [10, 10, 10])
    np.testing.assert_array_equal(col(m, "getfield(2 * a, 'Data')"), [2, 4, 6])
    np.testing.assert_array_equal(col(m, "getfield(a / 2, 'Data')"), [0.5, 1, 1.5])
    np.testing.assert_array_equal(col(m, "getfield(1 - a, 'Data')"), [0, -1, -2])
    np.testing.assert_array_equal(col(m, "getfield(-a, 'Data')"), [-1, -2, -3])
    np.testing.assert_array_equal(col(m, "getfield(a .^ 2, 'Data')"), [1, 4, 9])
    np.testing.assert_array_equal(col(m, "getfield(a + b, 'Time')"), [0, 1, 2])
    with pytest.raises(MatlabError, match="same times"):
        m.run("a + timeseries([1; 1; 1], 1:3)")


def test_arithmetic_with_samples_and_matrices(m):
    m.run("v = timeseries([1 2; 3 4; 5 6], 0:2); r = v .* [10 100]; s = v * [1; 1];")
    np.testing.assert_array_equal(m.num("r.Data"), [[10, 200], [30, 400], [50, 600]])
    np.testing.assert_array_equal(m.num("s.Data"), [[3], [7], [11]])


# -- filtering --------------------------------------------------------------------------------------------

def test_detrend_and_filter(m):
    m.run("a = detrend(timeseries([1; 3; 5; 8], 0:3), 'constant'); b = detrend(timeseries([1; 3; 5; 7], [0 1 2 3]),"
          " 'linear'); f = filter(timeseries([1; 1; 1; 1], 0:3), 1, [1 -0.5]);")
    np.testing.assert_allclose(col(m, "a.Data"), [-3.25, -1.25, 0.75, 3.75])
    np.testing.assert_allclose(col(m, "b.Data"), [0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(col(m, "f.Data"), [1, 1.5, 1.75, 1.875])


def test_ideal_filter_keeps_the_mean_with_notch(m):
    m.run("t = 0:7; ts = timeseries((3 + cos(pi * t))', t); n = idealfilter(ts, [0.4 0.6], 'notch');")
    np.testing.assert_allclose(col(m, "n.Data"), np.full(8, 3.0), atol=1e-12)


# -- plots, conversion and files --------------------------------------------------------------------------

def test_plot(m):
    m.run("figure('Visible', 'off'); ts = timeseries([3; 1; 2], [0 5 10], 'Name', 'level'); h = plot(ts);")
    np.testing.assert_array_equal(col(m, "h.XData"), [0, 5, 10])
    np.testing.assert_array_equal(col(m, "h.YData"), [3, 1, 2])
    assert m.str("get(get(gca, 'Title'), 'String')") == "Time Series Plot:level"
    assert m.str("get(get(gca, 'XLabel'), 'String')") == "Time (seconds)"
    m.run("close all")


def test_timetable_conversion(m):
    m.run("ts = timeseries([1 2; 3 4; 5 6], [0 1 2], 'Name', 'pos'); ts.DataInfo.Units = 'm';"
          "TT = timeseries2timetable(ts);")
    assert m.str("class(TT)") == "timetable"
    assert m.str("TT.Properties.VariableNames{1}") == "pos"
    np.testing.assert_array_equal(m.num("TT.pos"), [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(col(m, "seconds(TT.Time)"), [0, 1, 2])
    m.run("U = timeseries2timetable(timeseries([1; 2]), timeseries([3; 4]));")
    assert m.str("strjoin(U.Properties.VariableNames, ',')") == "Data_1,Data_2"
    m.run("back = timetable2timeseries(TT);")
    assert m.str("back.Name") == "pos" and m.str("back.DataInfo.Units") == "m"
    np.testing.assert_array_equal(m.num("back.Data"), [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(col(m, "back.Time"), [0, 1, 2])
    m.run("D = timeseries2timetable(timeseries([5; 6], {'01-Feb-2022', '03-Feb-2022'}));")
    assert m.str("class(D.Time)") == "datetime"
    np.testing.assert_array_equal(col(m, "day(D.Time)"), [1, 3])


@pytest.mark.parametrize("version", ["-v7", "-v7.3"])
def test_mat_file_round_trip(m, tmp_path, version):
    f = (tmp_path / "ts.mat").as_posix()
    m.run("ts = timeseries([1 2; 3 4; 5 6], [0 1 2.5], [1; 0; 2], 'Name', 'pos');"
          "ts.DataInfo.Units = 'm'; ts.TimeInfo.Units = 'minutes'; ts = setinterpmethod(ts, 'zoh');"
          "ts = addevent(ts, 'mark', 1); ts.UserData = 'note'; ts.TimeInfo.StartDate = '02-Mar-2020';"
          "row = timeseries([7 8 9]);"
          f"save('{f}', 'ts', 'row', '{version}'); clear ts row; L = load('{f}');")
    m.run("ts = timeseries([1 2; 3 4; 5 6], [0 1 2.5], [1; 0; 2], 'Name', 'pos');"
          "ts.DataInfo.Units = 'm'; ts.TimeInfo.Units = 'minutes'; ts = setinterpmethod(ts, 'zoh');"
          "ts = addevent(ts, 'mark', 1); ts.UserData = 'note'; ts.TimeInfo.StartDate = '02-Mar-2020';")
    assert m.num("isequal(L.ts, ts)")
    assert m.str("class(L.ts)") == "timeseries"
    np.testing.assert_array_equal(m.num("L.ts.Data"), [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_array_equal(col(m, "L.ts.Time"), [0, 1, 2.5])
    np.testing.assert_array_equal(col(m, "double(L.ts.Quality)"), [1, 0, 2])
    assert m.str("L.ts.TimeInfo.Units") == "minutes" and m.str("L.ts.DataInfo.Units") == "m"
    assert m.str("getinterpmethod(L.ts)") == "zoh" and m.str("L.ts.Events.Name") == "mark"
    assert m.str("L.ts.UserData") == "note" and m.str("L.ts.TimeInfo.StartDate") == "02-Mar-2020 00:00:00"
    np.testing.assert_array_equal(m.num("size(L.row.Data)"), [[1, 1, 3]])
    assert not m.num("L.row.IsTimeFirst")
