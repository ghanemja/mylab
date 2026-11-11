A door keypad has the keys `0`-`9`, `*` and `#`. Write a MATLAB function

    [state, wrong] = keypad_lock(code, keys)

that simulates the lock. `code` is the correct code (a char vector of digits)
and `keys` is a char vector with the keys pressed, in order. The lock starts
in state `'locked'` with an empty entry buffer.

- A digit is appended to the buffer.
- `*` clears the buffer.
- `#` submits the buffer. If the buffer is empty nothing happens. If it equals
  `code` exactly, the state becomes `'open'`. Otherwise it is a wrong
  attempt. Either way the buffer is cleared.
- After the third wrong attempt the state becomes `'blocked'`.
- Once the state is `'open'` or `'blocked'`, all further keys are ignored.

Return the final `state` (`'locked'`, `'open'` or `'blocked'`) and the number
of wrong attempts `wrong`. If `keys` contains any character other than the
twelve keys, throw an error with identifier `keypad_lock:badKey` (check this
before simulating).

Examples:

- `[s, w] = keypad_lock('4071', '12#4071#')` gives `s = 'open'`, `w = 1`.
- `[s, w] = keypad_lock('4071', '1#2#3#4071#')` gives `s = 'blocked'`, `w = 3`.
- `[s, w] = keypad_lock('4071', '4071')` gives `s = 'locked'`, `w = 0`.
