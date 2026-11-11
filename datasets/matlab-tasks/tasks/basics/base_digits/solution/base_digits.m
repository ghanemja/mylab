function d = base_digits(n, b)
%BASE_DIGITS Digits of nonnegative integer N in base B, most significant first.
if n == 0
    d = 0;
    return
end
d = zeros(1, 0);
while n > 0
    d = [mod(n, b), d];
    n = floor(n / b);
end
end
