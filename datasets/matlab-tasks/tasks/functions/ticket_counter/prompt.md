A deli counter hands out numbered tickets. Write a MATLAB function

    n = ticket_counter(cmd)

that keeps its state between calls (use a `persistent` variable):

- `ticket_counter()` issues the next ticket and returns its number. The first
  ticket ever issued is number 1, then 2, 3, ...
- `ticket_counter('peek')` returns the number of the last ticket issued
  without issuing a new one (0 if none has been issued since the start or the
  last reset).
- `ticket_counter('reset')` starts over, so the next ticket is number 1 again;
  it returns 0.
- Any other command throws an error with identifier
  `ticket_counter:badCommand`.

Example:

    ticket_counter('reset');
    ticket_counter()        % 1
    ticket_counter()        % 2
    ticket_counter('peek')  % 2
