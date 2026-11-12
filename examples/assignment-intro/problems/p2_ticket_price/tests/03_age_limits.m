% name: Ages exactly on a limit
% points: 7
% hidden: true
assert(isequal(ticket_price(0, false), 0));
assert(isequal(ticket_price(4, false), 0));
assert(isequal(ticket_price(5, false), 6));
assert(isequal(ticket_price(17, false), 6));
assert(isequal(ticket_price(18, false), 11));
assert(isequal(ticket_price(64, false), 11));
assert(isequal(ticket_price(65, false), 8));
assert(isequal(ticket_price(4, true), 0));
assert(isequal(ticket_price(5, true), 4));
assert(isequal(ticket_price(65, true), 6));
