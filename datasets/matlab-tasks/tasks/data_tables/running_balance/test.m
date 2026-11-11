T = table({'A'; 'B'; 'A'}, [100; 20; -30], 'VariableNames', {'Account', 'Amount'});
R = running_balance(T);
assert(isequal(R.Properties.VariableNames, {'Account', 'Amount', 'Balance'}), 'new last variable');
assert(isequal(R.Balance, [100; 20; 70]), 'example');
U = table([1; 2; 3; 4; 5; 6], {'x'; 'y'; 'x'; 'X'; 'y'; 'x'}, [5; -5; 2.5; 1; 5; -7.5], ...
    'VariableNames', {'Day', 'Account', 'Amount'});
R = running_balance(U);
assert(isequal(R.Balance, [5; -5; 7.5; 1; 0; 0]), 'interleaved accounts, case-sensitive');
assert(isequal(R.Day, (1:6)') && isequal(R.Account, U.Account), 'other variables unchanged');
R = running_balance(table({'solo'}, 42, 'VariableNames', {'Account', 'Amount'}));
assert(R.Balance == 42, 'single row');
E = running_balance(U([], :));
assert(height(E) == 0 && width(E) == 4, 'empty table');
