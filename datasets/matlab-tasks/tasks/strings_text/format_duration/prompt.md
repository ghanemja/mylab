Write a MATLAB function

    s = format_duration(sec)

that formats a nonnegative whole number of seconds as a char vector:

- below one hour: `M:SS` (minutes without leading zeros, seconds with two
  digits), e.g. `'4:05'`;
- one hour or more: `H:MM:SS` (hours without leading zeros and not wrapped at
  24, minutes and seconds with two digits), e.g. `'25:01:01'`.

Examples:

- `format_duration(59)` returns `'0:59'`.
- `format_duration(3600)` returns `'1:00:00'`.
- `format_duration(90061)` returns `'25:01:01'`.
