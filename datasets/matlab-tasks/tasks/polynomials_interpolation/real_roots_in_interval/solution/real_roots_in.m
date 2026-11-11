function r = real_roots_in(c, a, b)
%REAL_ROOTS_IN Sorted distinct real roots of a polynomial in [a, b].
c = c(:)';
first = find(c ~= 0, 1);
c = c(first:end);
r = zeros(1, 0);
if numel(c) < 2
    return
end
z = roots(c);
isReal = abs(imag(z)) <= 1e-7 * max(1, abs(z));
x = sort(real(z(isReal)))';
x = x(x >= a - 1e-9 & x <= b + 1e-9);
i = 1;
while i <= numel(x)
    j = i;
    while j < numel(x) && x(j+1) - x(i) <= 1e-5
        j = j + 1;
    end
    r(end+1) = mean(x(i:j)); %#ok<AGROW>
    i = j + 1;
end
end
