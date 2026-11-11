T = table([3; 1; 3; 3], {'food'; 'rent'; 'food'; 'rent'}, [10; 500; 5; 450], ...
    'VariableNames', {'Month', 'Category', 'Amount'});
P = monthly_pivot(T);
assert(istable(P) && isequal(P.Properties.VariableNames, {'Category', 'M1', 'M3'}), 'example variables');
assert(isequal(P.Category, {'food'; 'rent'}), 'example categories');
assert(isequal(P.M1, [0; 500]) && isequal(P.M3, [15; 450]), 'example totals');
U = table([12; 2; 10; 2; 12; 10], {'b'; 'c'; 'a'; 'b'; 'b'; 'c'}, [1.5; 2; 3; 4; 5; -1], ...
    'VariableNames', {'Month', 'Category', 'Amount'});
P = monthly_pivot(U);
assert(isequal(P.Properties.VariableNames, {'Category', 'M2', 'M10', 'M12'}), 'numeric month order');
assert(isequal(P.Category, {'a'; 'b'; 'c'}), 'sorted categories');
assert(isequal(P.M2, [0; 4; 2]) && isequal(P.M10, [3; 0; -1]) && isequal(P.M12, [0; 6.5; 0]), 'totals and zeros');
P = monthly_pivot(table(6, {'x'}, 9, 'VariableNames', {'Month', 'Category', 'Amount'}));
assert(isequal(size(P), [1 2]) && P.M6 == 9, 'single row');
