function tf = phrase_palindrome(s)
%PHRASE_PALINDROME True if the letters and digits of S form a palindrome.
s = lower(s);
keep = (s >= 'a' & s <= 'z') | (s >= '0' & s <= '9');
t = s(keep);
tf = isequal(t, fliplr(t));
tf = logical(tf);
end
