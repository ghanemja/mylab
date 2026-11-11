function n = ticket_counter(cmd)
%TICKET_COUNTER Issue, peek at or reset numbered tickets.
persistent last
if isempty(last)
    last = 0;
end
if nargin == 0
    last = last + 1;
    n = last;
    return
end
switch cmd
    case 'peek'
        n = last;
    case 'reset'
        last = 0;
        n = 0;
    otherwise
        error('ticket_counter:badCommand', 'Unknown command.');
end
end
