orders = table([7; 8], {'pen'; 'ink'}, [3; 2], 'VariableNames', {'OrderID', 'Product', 'Qty'});
prices = table({'ink'; 'pen'}, [4.5; 1.2], 'VariableNames', {'Product', 'UnitPrice'});
T = order_costs(orders, prices);
assert(isequal(T.Properties.VariableNames, {'OrderID', 'Product', 'Qty', 'UnitPrice', 'Cost'}), 'variables');
assert(isequal(T.OrderID, [7; 8]) && isequal(T.Product, {'pen'; 'ink'}), 'row order kept');
assert(all(abs(T.UnitPrice - [1.2; 4.5]) < 1e-12) && all(abs(T.Cost - [3.6; 9]) < 1e-12), 'example');
O = table([3; 1; 2; 5], {'Cup'; 'cup'; 'plate'; 'cup'}, [1; 4; 2; 0], {'r'; 's'; 't'; 'u'}, ...
    'VariableNames', {'OrderID', 'Product', 'Qty', 'Note'});
P = table({'cup'; 'bowl'; 'plate'}, [2; 3; 5], 'VariableNames', {'Product', 'UnitPrice'});
T = order_costs(O, P);
assert(width(T) == 5 && height(T) == 4, 'extra variable dropped, all rows kept');
assert(isequal(T.OrderID, [3; 1; 2; 5]), 'original order, no sorting');
assert(isnan(T.UnitPrice(1)) && isnan(T.Cost(1)), 'unknown product (case-sensitive)');
assert(isequal(T.UnitPrice(2:4), [2; 5; 2]) && isequal(T.Cost(2:4), [8; 10; 0]), 'looked-up values');
