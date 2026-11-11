a = Ledger('Ana');
b = Ledger('Bo');
assert(strcmp(a.Owner, 'Ana') && a.Balance == 0 && isequal(size(a.Entries), [0 1]), 'new account');
deposit(a, 100);
transfer(a, b, 30);
assert(a.Balance == 70 && isequal(a.Entries, [100; -30]), 'withdrawal side of transfer');
assert(isequal(b.Entries, 30) && b.Balance == 30, 'deposit side of transfer');
threw = false;
try
    withdraw(b, 30.01);
catch err
    threw = strcmp(err.identifier, 'Ledger:insufficient');
end
assert(threw, 'overdraw');
threw = false;
try
    transfer(b, a, 31);
catch err
    threw = strcmp(err.identifier, 'Ledger:insufficient');
end
assert(threw, 'transfer overdraw');
assert(b.Balance == 30 && a.Balance == 70 && numel(a.Entries) == 2, 'failed transfer changes nothing');
threw = false;
try
    deposit(a, -5);
catch err
    threw = strcmp(err.identifier, 'Ledger:badAmount');
end
assert(threw, 'negative deposit');
threw = false;
try
    withdraw(a, 0);
catch err
    threw = strcmp(err.identifier, 'Ledger:badAmount');
end
assert(threw, 'zero withdrawal');
threw = false;
try
    deposit(a, Inf);
catch err
    threw = strcmp(err.identifier, 'Ledger:badAmount');
end
assert(threw, 'infinite deposit');
c = a;
withdraw(c, 70);
assert(a.Balance == 0 && isequal(a.Entries, [100; -30; -70]), 'exact zero balance, handle copy');
threw = false;
try
    a.Entries = [1; 2];
catch
    threw = true;
end
assert(threw, 'Entries read-only');
