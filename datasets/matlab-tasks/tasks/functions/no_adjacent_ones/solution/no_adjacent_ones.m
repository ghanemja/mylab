function c = no_adjacent_ones(n)
%NO_ADJACENT_ONES All binary strings of length N without adjacent ones.
if n == 0
    c = {''};
elseif n == 1
    c = {'0'; '1'};
else
    a = no_adjacent_ones(n - 1);
    b = no_adjacent_ones(n - 2);
    c = [strcat({'0'}, a); strcat({'10'}, b)];
end
end
