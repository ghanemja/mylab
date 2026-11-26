"""Help text for timetables."""

_METHODS = ("Methods: 'fillwithmissing' (default: a value only where an old row time equals the new one), "
            "'fillwithconstant' (with 'Constant', value), 'previous', 'next', 'nearest', the interpolations "
            "'linear', 'spline', 'pchip', 'makima' (which extrapolate beyond the data unless 'EndValues' gives "
            "a constant), and the aggregations 'sum', 'mean', 'prod', 'min', 'max', 'count', 'firstvalue', "
            "'lastvalue' or a function handle. For an aggregation each new row time starts a bin that ends "
            "at the next one; the last bin of a time step such as 'daily' is a whole step, while the last of "
            "a list of times holds only the data at exactly that time.")

HELP = {
    "timetable": (["TT = timetable(rowTimes, var1, ..., varN)", "TT = timetable(var1, ..., varN, 'RowTimes', rowTimes)",
                   "TT = timetable(var1, ..., 'SampleRate', Fs)", "TT = timetable(var1, ..., 'TimeStep', dt)",
                   "TT = timetable(..., 'StartTime', t0, 'VariableNames', names)",
                   "TT = timetable('Size', [rows vars], 'VariableTypes', types, 'RowTimes', rowTimes)"],
                  "A table whose rows are labelled with times: a datetime or duration vector, or regular "
                  "times made from a sample rate or a time step. TT.Time (or the name of the times "
                  "variable) gives the row times and TT.Properties has RowTimes, StartTime, SampleRate and "
                  "TimeStep. Rows are selected by number, by time (a datetime, date text, timerange, "
                  "withtol) or with a logical vector. See retime, synchronize, lag, isregular, "
                  "table2timetable, timetable2table."),
    "istimetable": ("tf = istimetable(x)", "True when x is a timetable."),
    "table2timetable": (["TT = table2timetable(T)", "TT = table2timetable(T, 'RowTimes', timesOrVariable)",
                         "TT = table2timetable(T, 'SampleRate', Fs)", "TT = table2timetable(T, 'TimeStep', dt)"],
                        "Makes a timetable from a table. The row times are the first datetime or duration "
                        "variable of T (which is removed from the variables), a given vector or variable, "
                        "or regular times."),
    "timetable2table": (["T = timetable2table(TT)", "T = timetable2table(TT, 'ConvertRowTimes', false)"],
                        "Makes a table from a timetable. The row times become the first variable unless "
                        "'ConvertRowTimes' is false."),
    "array2timetable": (["TT = array2timetable(A, 'RowTimes', rowTimes)", "TT = array2timetable(A, 'SampleRate', Fs)",
                         "TT = array2timetable(A, 'TimeStep', dt, 'VariableNames', names)"],
                        "Makes a timetable whose variables are the columns of the matrix A."),
    "timerange": (["S = timerange(start, stop)", "S = timerange(start, stop, intervalType)",
                   "S = timerange(start, stop, unit)"],
                  "A subscript that selects the rows of a timetable whose times lie between start and stop: "
                  "TT(S, :). The interval includes start and excludes stop ('openright') unless intervalType "
                  "is 'closed', 'open', 'openleft' or 'closedleft'. With a unit ('years', 'months', 'days', "
                  "...) the interval runs from the start of the unit of start to the end of the unit of "
                  "stop. start and stop are datetimes, durations or date text; -Inf and Inf leave a side open."),
    "withtol": ("S = withtol(rowTimes, tol)",
                "A subscript that selects the rows of a timetable whose times are within the duration tol "
                "of one of the given times: TT(withtol(t, seconds(1)), :)."),
    "retime": (["TT2 = retime(TT, newTimeStep)", "TT2 = retime(TT, newTimeStep, method)",
                "TT2 = retime(TT, newTimes, method)", "TT2 = retime(TT, 'regular', method, 'TimeStep', dt)",
                "TT2 = retime(TT, 'regular', method, 'SampleRate', Fs)"],
               "Moves the data of a timetable to new row times. newTimeStep is 'yearly', 'quarterly', "
               "'monthly', 'weekly', 'daily', 'hourly', 'minutely' or 'secondly'; newTimes is a datetime or "
               "duration vector. " + _METHODS),
    "synchronize": (["TT = synchronize(TT1, TT2, ...)", "TT = synchronize(TT1, TT2, newTimeBasis, method)",
                     "TT = synchronize(TT1, TT2, newTimeStep, method)", "TT = synchronize(TT1, TT2, newTimes, method)"],
                    "Combines timetables into one with common row times. newTimeBasis is 'union' (default), "
                    "'intersection', 'commonrange', 'first' or 'last'; a time step or a vector of times "
                    "works as in retime. Variables with the same name get the names of the input timetables "
                    "appended. " + _METHODS),
    "lag": (["TT2 = lag(TT)", "TT2 = lag(TT, n)", "TT2 = lag(TT, dt)"],
            "Shifts the data of a regular timetable forward in time by n time steps (default 1) or by the "
            "duration dt, which must be a multiple of the time step. The row times stay; the rows that have "
            "no data left are filled with missing values. A negative shift moves the data back."),
    "isregular": (["tf = isregular(TT)", "[tf, dt] = isregular(TT)", "tf = isregular(TT, timeComponent)"],
                  "True when the row times of a timetable (or a datetime or duration vector) are equally "
                  "spaced; dt is the step. With timeComponent 'years', 'quarters', 'months', 'weeks' or "
                  "'days' the spacing is checked in calendar units, so month starts count as regular."),
}
