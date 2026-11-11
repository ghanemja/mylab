e = struct('category', {'food', 'rent', 'food'}, 'amount', {12.5, 800, 7.5});
t = totals_by_category(e);
assert(isequal(fieldnames(t), {'food'; 'rent'}), 'example fields');
assert(t.food == 20 && t.rent == 800, 'example totals');
e = struct('category', {'travel', 'rent', 'food', 'travel', 'rent', 'Food'}, ...
           'amount', {60, 25, 3, -10, 800, 1});
t = totals_by_category(e);
assert(isequal(fieldnames(t), {'travel'; 'rent'; 'food'; 'Food'}), 'first-occurrence order, case-sensitive');
assert(t.travel == 50 && t.rent == 825 && t.food == 3 && t.Food == 1, 'totals with refund');
assert(isequal(size(t), [1 1]), 'scalar struct');
t = totals_by_category(struct('category', {}, 'amount', {}));
assert(isstruct(t) && isempty(fieldnames(t)), 'empty input');
t = totals_by_category(struct('category', 'misc_2024', 'amount', 0));
assert(isequal(fieldnames(t), {'misc_2024'}) && t.misc_2024 == 0, 'single record');
