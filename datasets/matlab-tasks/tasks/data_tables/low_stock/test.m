T = table({'bolt'; 'nut'; 'gear'}, [12; 3; 7], 'VariableNames', {'Item', 'Qty'});
R = low_stock(T, 10);
assert(istable(R) && isequal(R.Item, {'nut'; 'gear'}), 'example');
assert(isequal(R.Qty, [3; 7]), 'quantities');
R = low_stock(T, 12);
assert(height(R) == 2, 'strictly less than');
R = low_stock(T, 1);
assert(height(R) == 0 && width(R) == 2, 'no rows');
U = table({'a'; 'b'; 'c'; 'd'; 'e'}, [4; 1; 4; 0; 9], {'x'; 'y'; 'z'; 'w'; 'v'}, ...
    'VariableNames', {'Item', 'Qty', 'Shelf'});
R = low_stock(U, 5);
assert(isequal(R.Item, {'d'; 'b'; 'a'; 'c'}), 'ties keep input order');
assert(isequal(R.Shelf, {'w'; 'y'; 'x'; 'z'}), 'other variables follow rows');
assert(isequal(R.Properties.VariableNames, {'Item', 'Qty', 'Shelf'}), 'variables kept');
