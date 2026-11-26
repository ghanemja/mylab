"""Help text for timeseries objects and their functions."""

_TIMES = ("Times are numbers in TimeInfo.Units; when TimeInfo.StartDate is set they may also be given as date "
          "text or datetime values.")

HELP = {
    "timeseries": (["ts = timeseries()", "ts = timeseries(name)", "ts = timeseries(data)",
                    "ts = timeseries(data, time)", "ts = timeseries(data, time, quality)",
                    "ts = timeseries(..., 'Name', name)", "ts = timeseries(data, time, name)"],
                   "A set of data samples, each tied to a time. Without times the samples are 0, 1, 2, ... "
                   "seconds apart; the samples are sorted by time. For a vector or matrix, each row is one "
                   "sample (IsTimeFirst is true); with three or more dimensions, or when only the columns of a "
                   "matrix match the times, the last dimension runs along time. Times given as date text or "
                   "datetime values set TimeInfo.StartDate and count days from it. Properties: Data, Time, Name, "
                   "Length, Quality (int8 codes, one per sample or per value), QualityInfo, IsTimeFirst, Events, "
                   "TreatNaNasMissing, UserData, TimeInfo (Units, Start, End, Increment, Length, StartDate, "
                   "Format) and DataInfo (Units, Interpolation). Property names may be written in any case "
                   "(ts.data). Methods: resample, synchronize, append, addsample, delsample, getsamples, "
                   "getdatasamples, getsampleusingtime, setinterpmethod, getinterpmethod, addevent, delevent, "
                   "detrend, filter, idealfilter, plot, get, set, and the statistics mean, median, max, min, "
                   "std, var, sum and iqr (with 'MissingData', 'Quality' and 'Weighting' options). Arithmetic "
                   "works between timeseries with the same times and between a timeseries and numbers. "
                   "length(ts) is the number of samples and size(ts) is [length 1]. A timeseries can be saved "
                   "to and loaded from MAT-files."),
    "tsdata.event": (["e = tsdata.event(name, time)", "e = tsdata.event(name, time, 'Units', u, 'StartDate', d)"],
                     "An event: a named moment in time that can be attached to a timeseries with addevent. "
                     "Properties: Name, Time, Units, StartDate and EventData."),
    "tsdata.interpolation": (["m = tsdata.interpolation('linear')", "m = tsdata.interpolation('zoh')",
                              "m = tsdata.interpolation(fh)"],
                             "An interpolation method for a timeseries (ts.DataInfo.Interpolation): straight lines "
                             "between samples, zero-order hold (each value lasts until the next sample), or a "
                             "function fh(newTimes, times, data). Properties: Name and Fhandle."),
    "timeseries2timetable": (["TT = timeseries2timetable(ts)", "TT = timeseries2timetable(ts1, ..., tsN)"],
                             "Makes a timetable with one variable per timeseries. The row times are durations "
                             "in the timeseries' time units, or datetimes when TimeInfo.StartDate is set. "
                             "Each variable is named after its timeseries; unnamed ones become Data (or "
                             "Data_1, Data_2, ...). All the timeseries must have the same times; quality codes "
                             "are dropped with a warning."),
    "timetable2timeseries": (["ts = timetable2timeseries(TT)", "ts = timetable2timeseries(TT, varname)",
                              "[ts1, ..., tsN] = timetable2timeseries(TT)"],
                             "Makes timeseries from the numeric variables of a timetable, one per output and "
                             "named after the variable (with a single output and several variables, one "
                             "timeseries holds them side by side). Duration row times become times in their "
                             "display units; datetime row times become seconds from a StartDate."),
    "getdatasamples": ("data = getdatasamples(ts, ind)",
                       "The data of the samples at the indices ind (numbers or a logical mask), laid out like "
                       "ts.Data."),
    "getsamples": ("ts2 = getsamples(ts, ind)",
                   "A timeseries with only the samples at the indices ind (numbers or a logical mask)."),
    "getsampleusingtime": (["ts2 = getsampleusingtime(ts, t)", "ts2 = getsampleusingtime(ts, tstart, tend)"],
                           "A timeseries with the samples at time t, or at the times from tstart to tend "
                           "(both included). " + _TIMES),
    "getdatasamplesize": ("sz = getdatasamplesize(ts)",
                          "The size of one data sample: [1 n] for samples in rows of n values, or the leading "
                          "dimensions when time runs along the last dimension."),
    "getinterpmethod": ("name = getinterpmethod(ts)",
                        "The name of the interpolation method resample uses: 'linear' or 'zoh'."),
    "setinterpmethod": (["ts = setinterpmethod(ts, 'linear')", "ts = setinterpmethod(ts, 'zoh')",
                         "ts = setinterpmethod(ts, fh)", "ts = setinterpmethod(ts, interpObj)"],
                        "Sets the interpolation method of a timeseries: straight lines between samples, a "
                        "zero-order hold, a function fh(newTimes, times, data) or a tsdata.interpolation object."),
    "addsample": (["ts = addsample(ts, 'Data', d, 'Time', t)",
                   "ts = addsample(ts, 'Data', d, 'Time', t, 'Quality', q, 'OverwriteFlag', tf)",
                   "ts = addsample(ts, s)"],
                  "Adds samples (one per time in t) and keeps the samples in time order. With OverwriteFlag "
                  "true a new sample replaces an old one at the same time. s can be a struct with fields "
                  "Data, Time, Quality and OverwriteFlag. " + _TIMES),
    "delsample": (["ts = delsample(ts, 'Index', ind)", "ts = delsample(ts, 'Value', times)"],
                  "Removes the samples at the given indices, or at the given times."),
    "addevent": (["ts = addevent(ts, e)", "ts = addevent(ts, name, time)"],
                 "Attaches events to a timeseries: tsdata.event objects, or names (a cell array for several) "
                 "with their times."),
    "delevent": (["ts = delevent(ts, name)", "ts = delevent(ts, name, n)"],
                 "Removes the events with the given name (or names), or only the n-th event with that name."),
    "getabstime": ("dates = getabstime(ts)",
                   "The times of a timeseries as date text (a cell column), counted from TimeInfo.StartDate "
                   "and shown in TimeInfo.Format (default 'dd-mmm-yyyy HH:MM:SS')."),
    "setabstime": (["ts = setabstime(ts, dates)", "ts = setabstime(ts, dates, format)"],
                   "Sets the times of a timeseries from dates (text or datetime, one per sample, in increasing "
                   "order): the first date becomes TimeInfo.StartDate."),
    "setuniformtime": (["ts = setuniformtime(ts, 'StartTime', t0, 'Interval', dt)",
                        "ts = setuniformtime(ts, 'StartTime', t0, 'EndTime', t1)",
                        "ts = setuniformtime(ts, 'Interval', dt, 'EndTime', t1)"],
                       "Replaces the times of a timeseries with evenly spaced ones; the data is unchanged."),
    "getqualitydesc": ("desc = getqualitydesc(ts)",
                       "The description (from QualityInfo.Code and QualityInfo.Description) of the quality code "
                       "of each sample, as a cell column."),
    "idealfilter": (["ts = idealfilter(ts, interval, 'pass')", "ts = idealfilter(ts, interval, 'notch')",
                     "ts = idealfilter(ts, interval, type, index)"],
                    "Keeps ('pass') or removes ('notch') the frequencies in the rows [low high] of interval, in "
                    "cycles per time unit, by zeroing them in the Fourier transform of each data column (after "
                    "removing its mean). Needs evenly spaced times; index picks the columns to filter."),
    "gettsafterevent": (["ts2 = gettsafterevent(ts, event)", "ts2 = gettsafterevent(ts, name, n)"],
                        "The samples after an event (a tsdata.event, or the n-th event with that name)."),
    "gettsafteratevent": (["ts2 = gettsafteratevent(ts, event)", "ts2 = gettsafteratevent(ts, name, n)"],
                          "The samples at or after an event."),
    "gettsatevent": (["ts2 = gettsatevent(ts, event)", "ts2 = gettsatevent(ts, name, n)"],
                     "The samples at the time of an event."),
    "gettsbeforeevent": (["ts2 = gettsbeforeevent(ts, event)", "ts2 = gettsbeforeevent(ts, name, n)"],
                         "The samples before an event."),
    "gettsbeforeatevent": (["ts2 = gettsbeforeatevent(ts, event)", "ts2 = gettsbeforeatevent(ts, name, n)"],
                           "The samples before or at an event."),
    "gettsbetweenevents": (["ts2 = gettsbetweenevents(ts, event1, event2)",
                            "ts2 = gettsbetweenevents(ts, name1, name2, n1, n2)"],
                           "The samples from the first event to the second (both included)."),
}
