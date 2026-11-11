T = table({'north'; 'east'; 'north'}, [10; 5; 2.5], 'VariableNames', {'Region', 'Sales'});
S = region_totals(T);
assert(istable(S) && isequal(S.Properties.VariableNames, {'Region', 'Total', 'Count'}), 'variables');
assert(isequal(S.Region, {'east'; 'north'}), 'sorted regions');
assert(isequal(S.Total, [5; 12.5]) && isequal(S.Count, [1; 2]), 'example values');
U = table([1; 2; 3; 4; 5; 6], {'w'; 'W'; 'w'; 'e'; 'w'; 'e'}, [1; -2; 3; 4; 5; 0], ...
    'VariableNames', {'Id', 'Region', 'Sales'});
S = region_totals(U);
assert(isequal(S.Region, {'W'; 'e'; 'w'}), 'case-sensitive, extra variable ignored');
assert(isequal(S.Total, [-2; 4; 9]), 'totals with negatives and zero');
assert(isequal(S.Count, [1; 2; 3]), 'counts');
S = region_totals(table({'solo'}, 7, 'VariableNames', {'Region', 'Sales'}));
assert(height(S) == 1 && S.Total == 7 && S.Count == 1, 'single record');
