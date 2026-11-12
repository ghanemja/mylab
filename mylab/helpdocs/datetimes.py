"""Help text for dates, times and durations."""

_DN = "A serial date number counts days from year 0, with the time of day as the fraction."


def _unit(name: str, what: str) -> tuple:
    return ([f"d = {name}(x)", f"x = {name}(d)"],
            f"With numbers, creates a duration array of that many {what}. With a duration array, converts it "
            f"to a plain number of {what}.")


def _cal(name: str, what: str) -> tuple:
    return (f"c = {name}(n)",
            f"A calendar duration of n {what} (whole numbers). Added to a datetime it moves by calendar units, "
            "so the clock time is kept and month ends are respected: datetime(2024,1,31) + calmonths(1) is "
            "29-Feb-2024.")


HELP = {
    "datetime": (["t = datetime", "t = datetime('now' | 'today' | 'tomorrow' | 'yesterday')",
                  "t = datetime(Y, M, D)", "t = datetime(Y, M, D, H, MI, S)",
                  "t = datetime(text)", "t = datetime(text, 'InputFormat', fmt)",
                  "t = datetime(x, 'ConvertFrom', kind)", "t = datetime(..., 'Format', fmt, 'TimeZone', zone)"],
                 "Creates an array of points in time. Numeric inputs give the year, month, day and optionally "
                 "the hour, minute and second (scalars or arrays of one size; out-of-range values roll over). "
                 "Text (a char vector, string array or cell array of text) is parsed: ISO forms such as "
                 "'2024-03-15 10:30:00' and common forms such as '15-Mar-2024', '03/15/2024' or "
                 "'March 15, 2024' are recognized automatically, and 'InputFormat' states the layout "
                 "explicitly. 'ConvertFrom' accepts 'datenum', 'posixtime', 'excel', 'juliandate', "
                 "'modifiedjuliandate', 'yyyymmdd' and 'epochtime'. 'Format' sets how values are shown, using "
                 "the letters y (year), M (month: MM number, MMM short name, MMMM full name), d (day), "
                 "e (weekday: eee short, eeee full), H (hour 0-23), h (hour 1-12), m (minute), s (second), "
                 "S (fraction of a second), a (AM/PM), with literal text in single quotes. 'TimeZone' may be "
                 "'UTC', 'local' or '' (none); other zones are not available. Properties Year, Month, Day, "
                 "Hour, Minute, Second, Format and TimeZone can be read and set. Datetimes support "
                 "subtraction (giving a duration), adding durations and calendar durations, comparison, "
                 "sorting, indexing and concatenation; NaT marks a missing value."),
    "NaT": (["t = NaT", "t = NaT(n)", "t = NaT(m, n)"],
            "Not-a-Time: the missing value for datetime arrays, as a scalar or an array of the given size. "
            "It compares unequal to everything; test for it with isnat."),
    "datetime.empty": ("t = datetime.empty(m, n)", "An empty datetime array (0-by-0 by default)."),
    "duration.empty": ("d = duration.empty(m, n)", "An empty duration array (0-by-0 by default)."),
    "isnat": ("tf = isnat(t)", "True where the elements of a datetime array are NaT."),
    "isdatetime": ("tf = isdatetime(x)", "True when x is a datetime array."),
    "isduration": ("tf = isduration(x)", "True when x is a duration array."),
    "iscalendarduration": ("tf = iscalendarduration(x)", "True when x is a calendar duration array."),
    "duration": (["d = duration(H, MI, S)", "d = duration(H, MI, S, MS)", "d = duration(text)",
                  "d = duration(..., 'Format', fmt)"],
                 "Creates an array of lengths of time from hours, minutes and seconds (and optionally "
                 "milliseconds), or from text such as '01:30:00' or '2:12:00:00' (days:hours:minutes:"
                 "seconds). The Format property controls the display: 'hh:mm:ss' (default), 'dd:hh:mm:ss', "
                 "'mm:ss', 'hh:mm', fractional seconds as in 'hh:mm:ss.SSS', or a single unit 'y', 'd', 'h', "
                 "'m' or 's'. Durations can be added, subtracted, compared, scaled by numbers and divided by "
                 "each other (giving a number). In arithmetic with a duration, a plain number counts as days."),
    "seconds": _unit("seconds", "seconds"),
    "minutes": _unit("minutes", "minutes"),
    "hours": _unit("hours", "hours"),
    "days": _unit("days", "24-hour days"),
    "years": _unit("years", "years of 365.2425 days"),
    "milliseconds": _unit("milliseconds", "milliseconds"),
    "calyears": _cal("calyears", "calendar years"),
    "calquarters": _cal("calquarters", "calendar quarters"),
    "calmonths": _cal("calmonths", "calendar months"),
    "calweeks": _cal("calweeks", "calendar weeks"),
    "caldays": _cal("caldays", "calendar days"),
    "between": ("c = between(t1, t2)", "The calendar duration from t1 to t2, as whole months, then days, then "
                "the remaining time, so that t1 + c equals t2."),
    "caldiff": ("c = caldiff(t)", "Calendar durations between successive elements of a datetime vector."),
    "year": ("y = year(t)", "Year numbers of a datetime array (serial date numbers and date text are also "
             "accepted)."),
    "month": (["m = month(t)", "names = month(t, 'name')", "names = month(t, 'shortname')"],
              "Month numbers (1-12) of a datetime array, or the month names as a cell array of text. Serial "
              "date numbers and date text are also accepted."),
    "day": (["d = day(t)", "d = day(t, 'dayofyear')", "d = day(t, 'dayofweek')", "names = day(t, 'name')"],
            "Day of the month of each element of a datetime array. Options give the day of the year, the day "
            "of the week (1 is Sunday) or the weekday names ('name' or 'shortname')."),
    "hour": ("h = hour(t)", "Hour (0-23) of each element of a datetime array."),
    "minute": ("m = minute(t)", "Minute (0-59) of each element of a datetime array."),
    "second": ("s = second(t)", "Seconds, including any fraction, of each element of a datetime array."),
    "quarter": ("q = quarter(t)", "Quarter of the year (1-4) of each element of a datetime array."),
    "week": (["w = week(t)", "w = week(t, 'weekofmonth')"],
             "Week number within the year (the week containing January 1 is week 1 and weeks start on "
             "Sunday), or within the month."),
    "weekday": (["n = weekday(t)", "[n, name] = weekday(t)", "[n, name] = weekday(t, 'long')"],
                "Day of the week as a number, 1 for Sunday through 7 for Saturday, for a datetime array, "
                "serial date numbers or date text. The second output is the short day name ('Mon'), or the "
                "full name with 'long'."),
    "ymd": ("[y, m, d] = ymd(t)", "Year, month and day numbers of a datetime array as three numeric arrays."),
    "hms": ("[h, m, s] = hms(t)", "Hour, minute and second of a datetime array, or the hours, minutes and "
            "seconds that make up each element of a duration array."),
    "timeofday": ("d = timeofday(t)", "Time elapsed since midnight for each element of a datetime array, as a "
                  "duration."),
    "yyyymmdd": ("n = yyyymmdd(t)", "Each date as one number whose digits are the year, month and day, such "
                 "as 20240315."),
    "posixtime": ("p = posixtime(t)", "Seconds since 1970-01-01 00:00:00 UTC for each datetime. Values without "
                  "a time zone are taken as UTC."),
    "juliandate": (["jd = juliandate(t)", "mjd = juliandate(t, 'modifiedjuliandate')"],
                   "Julian date (days since noon on November 24, 4714 BC in the proleptic Gregorian "
                   "calendar), or the modified Julian date."),
    "exceltime": ("e = exceltime(t)", "Spreadsheet serial day numbers (day 1 is January 1, 1900) for each "
                  "datetime."),
    "convertTo": ("x = convertTo(t, kind)", "Converts a datetime array to numbers: kind is 'datenum', "
                  "'posixtime', 'excel', 'juliandate', 'modifiedjuliandate' or 'yyyymmdd'."),
    "isbetween": (["tf = isbetween(t, lower, upper)", "tf = isbetween(t, lower, upper, intervalType)"],
                  "True where lower <= t <= upper. The bounds may be datetimes or date text. intervalType is "
                  "'closed' (default), 'open', 'openleft' or 'openright'."),
    "dateshift": (["t2 = dateshift(t, 'start', unit)", "t2 = dateshift(t, 'end', unit)",
                   "t2 = dateshift(t, 'start', unit, rule)", "t2 = dateshift(t, 'dayofweek', day)",
                   "t2 = dateshift(t, 'dayofweek', day, rule)"],
                  "Moves each datetime to the start or end of the unit of time that contains it: 'year', "
                  "'quarter', 'month', 'week' (weeks start on Sunday), 'day', 'hour', 'minute' or 'second'. "
                  "The end of a year, quarter, month or week is its last day; the end of a shorter unit is "
                  "the instant the next one begins. rule is 'current' (default), 'next', 'previous', "
                  "'nearest' or a whole number of units to move. With 'dayofweek', moves to the given weekday "
                  "(a name or 1-7 with 1 = Sunday) on or after t ('next', default), on or before t "
                  "('previous') or closest to t ('nearest')."),
    "datenum": (["n = datenum(t)", "n = datenum(text)", "n = datenum(text, format)", "n = datenum(Y, M, D)",
                 "n = datenum(Y, M, D, H, MI, S)", "n = datenum([Y M D H MI S])"],
                "Converts dates to serial date numbers. " + _DN + " Input can be a datetime array, date text "
                "(optionally with a datestr-style format), separate components, or a matrix of date vectors."),
    "datestr": (["s = datestr(t)", "s = datestr(t, formatNumber)", "s = datestr(t, format)"],
                "Formats dates as text. t can be serial date numbers, date vectors, datetimes or date text; "
                "several dates give one row of text each. The default is 'dd-mmm-yyyy HH:MM:SS', or just the "
                "date when every time is midnight. Format numbers 0 to 31 select standard layouts (for "
                "example 1 is 'dd-mmm-yyyy', 13 is 'HH:MM:SS', 29 is 'yyyy-mm-dd' and 31 is "
                "'yyyy-mm-dd HH:MM:SS'). A free format uses yyyy, yy, mmmm, mmm, mm (month), dddd, ddd, dd, "
                "HH, MM (minute), SS, FFF (milliseconds) and AM or PM for a 12-hour clock."),
    "datevec": (["v = datevec(t)", "[Y, M, D, H, MI, S] = datevec(t)"],
                "Splits dates into components. Each row of v is [year month day hour minute second]. t can "
                "be serial date numbers, date text or datetimes."),
    "now": ("t = now", "The current local date and time as a serial date number. " + _DN),
    "today": ("t = today", "The current local date as a whole serial date number."),
    "clock": ("c = clock", "The current local date and time as [year month day hour minute seconds]."),
    "eomday": ("d = eomday(year, month)", "Last day of the given month, taking leap years into account: "
               "eomday(2024, 2) is 29."),
    "addtodate": ("n = addtodate(dateNumber, quantity, unit)",
                  "Adds a quantity of 'year', 'month', 'week', 'day', 'hour', 'minute', 'second' or "
                  "'millisecond' to serial date numbers. Months and years move by calendar, keeping the day "
                  "of the month where it exists."),
    "calendar": (["calendar", "calendar(year, month)", "calendar(date)", "c = calendar(...)"],
                 "Shows the calendar of a month (the current one by default), or returns it as a 6-by-7 "
                 "matrix with weeks as rows starting on Sunday and zeros for days outside the month."),
}
