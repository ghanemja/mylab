% name: Regular days
% points: 10
assert(isequal(ticket_price(2, false), 0), 'A 2-year-old gets in free');
assert(isequal(ticket_price(10, false), 6), 'A 10-year-old pays 6');
assert(isequal(ticket_price(30, false), 11), 'A 30-year-old pays 11');
assert(isequal(ticket_price(70, false), 8), 'A 70-year-old pays 8');
