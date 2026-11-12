% name: Monday discount
% points: 8
assert(isequal(ticket_price(12, true), 4), 'On Monday a 12-year-old pays 6 - 2 = 4');
assert(isequal(ticket_price(40, true), 9), 'On Monday a 40-year-old pays 11 - 2 = 9');
assert(isequal(ticket_price(80, true), 6), 'On Monday an 80-year-old pays 8 - 2 = 6');
assert(isequal(ticket_price(3, true), 0), 'A free ticket stays free on Monday');
