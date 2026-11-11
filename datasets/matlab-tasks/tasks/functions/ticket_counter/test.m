first = ticket_counter('peek');
assert(first == 0, 'nothing issued at start');
assert(ticket_counter() == 1, 'first ticket');
assert(ticket_counter() == 2, 'second ticket');
assert(ticket_counter('peek') == 2 && ticket_counter('peek') == 2, 'peek does not issue');
assert(ticket_counter('reset') == 0, 'reset returns 0');
assert(ticket_counter('peek') == 0, 'peek after reset');
for k = 1:5
    t = ticket_counter();
end
assert(t == 5 && ticket_counter() == 6, 'counting continues');
ok = false;
try
    ticket_counter('next');
catch err
    ok = strcmp(err.identifier, 'ticket_counter:badCommand');
end
assert(ok && ticket_counter('peek') == 6, 'bad command keeps state');
