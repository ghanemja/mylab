Write a MATLAB function `[wait, finish, maxQueue] = teller_queue(arrive, service)`
that simulates customers at a bank with a single teller.

- `arrive` is a vector of arrival times in non-decreasing order.
- `service` is a vector of the same length with each customer's (positive)
  service duration.

Customers are served one at a time in order of arrival (first come, first
served). Customer i starts service at the later of their arrival time and the
moment the previous customer finishes; the teller is never idle while someone
is waiting.

Return:

- `wait` — a row vector: time each customer spends waiting before service
  starts;
- `finish` — a row vector: the time each customer's service ends;
- `maxQueue` — the largest number of customers that are **waiting** (arrived
  but not yet being served) at the same moment. A customer waits during the
  half-open interval from arrival up to (not including) service start, so a
  customer who starts immediately never counts as waiting, and a customer who
  starts at the same instant another one arrives does not overlap with them.

For empty inputs, `wait` and `finish` are `zeros(1,0)` and `maxQueue` is 0.

Example:

```matlab
[w, f, q] = teller_queue([0 1 2 10], [3 3 1 2])
% w = [0 2 4 0], f = [3 6 7 12], q = 2
```
