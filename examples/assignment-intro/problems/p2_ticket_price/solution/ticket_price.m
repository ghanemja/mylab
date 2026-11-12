function price = ticket_price(age, is_monday)
% TICKET_PRICE  Price of one museum ticket.
%   price = ticket_price(age, is_monday)

if age < 5
    price = 0;
elseif age < 18
    price = 6;
elseif age < 65
    price = 11;
else
    price = 8;
end

if is_monday && price > 0
    price = price - 2;
end

end
