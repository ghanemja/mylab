Write a MATLAB function `[counts, firstError] = scan_log(filename)` that
summarises an application log.

A valid log line looks like

```
2025-02-17 08:15:02 [WARN] disk usage at 91%
```

that is: a date `YYYY-MM-DD`, one space, a time `HH:MM:SS`, one space, a
level in square brackets, one space, and a message (the rest of the line, which
may be empty). The level is one of `ERROR`, `WARN`, `INFO`, `DEBUG`
(upper-case). Any line that does not follow this format exactly (for example
blank lines, stack-trace lines, or an unknown level such as `[FATAL]`) is
ignored.

Return:

- `counts` — a 1-by-3 double row vector `[numError numWarn numInfo]`
  (`DEBUG` lines are valid but not counted);
- `firstError` — the message of the first `ERROR` line in the file, with
  leading and trailing whitespace removed, or `''` if there is none.

Example: for the lines

```
2025-02-17 08:15:00 [INFO] started
2025-02-17 08:15:02 [ERROR] cannot open cache
2025-02-17 08:15:03 [ERROR] retry failed
```

the result is `counts = [2 0 1]`, `firstError = 'cannot open cache'`.
