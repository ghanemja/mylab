Write a MATLAB **handle** class `Ledger` (file `Ledger.m`) representing a
bank account that records every transaction.

API:

- `acct = Ledger(owner)` creates an account for `owner` (a character vector)
  with no transactions.
- Read-only properties (`SetAccess = private`):
  - `Owner` — the owner name;
  - `Entries` — a column vector of all transaction amounts in the order they
    happened: deposits positive, withdrawals negative. It starts as
    `zeros(0,1)`.
- Dependent property `Balance` — the sum of `Entries` (0 for a new account).
- `deposit(acct, amount)` — adds a positive amount. If `amount` is not a
  positive finite number, throw `'Ledger:badAmount'`.
- `withdraw(acct, amount)` — records `-amount`. Throws `'Ledger:badAmount'`
  for a non-positive or non-finite amount, and `'Ledger:insufficient'` if the
  amount exceeds the current balance (the balance may reach exactly 0).
- `transfer(from, to, amount)` — moves money between two accounts: a
  withdrawal on `from` and a deposit on `to`. If the withdrawal is not allowed
  (same errors as `withdraw`), neither account changes.

A failed operation never changes `Entries`.

Example:

```matlab
a = Ledger('Ana'); b = Ledger('Bo');
deposit(a, 100);
transfer(a, b, 30);
a.Balance    % 70
b.Entries    % 30
```
