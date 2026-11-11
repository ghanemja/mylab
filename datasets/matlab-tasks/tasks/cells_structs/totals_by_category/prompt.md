Write a MATLAB function

    t = totals_by_category(expenses)

`expenses` is a struct array with fields `category` (a char vector that is a
valid MATLAB variable name, e.g. `'food'`) and `amount` (a number). Return a
scalar struct `t` with one field per distinct category, named after the
category, holding the sum of the amounts of that category. Fields appear in
the order in which their categories first occur in `expenses`. Categories are
case-sensitive (`'Food'` and `'food'` are different). For an empty `expenses`
return a struct with no fields (`struct()`).

Example:

    e = struct('category', {'food', 'rent', 'food'}, 'amount', {12.5, 800, 7.5});
    t = totals_by_category(e)
    % t.food = 20, t.rent = 800
