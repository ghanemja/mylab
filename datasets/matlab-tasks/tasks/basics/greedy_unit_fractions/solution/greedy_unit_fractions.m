function dens = greedy_unit_fractions(num, den)
%GREEDY_UNIT_FRACTIONS Greedy decomposition of num/den into unit fractions.
dens = zeros(1, 0);
g = gcd(num, den);
num = num / g;
den = den / g;
while num > 0
    d = ceil(den / num);
    dens(end + 1) = d; %#ok<AGROW>
    num = num * d - den;
    den = den * d;
    g = gcd(num, den);
    num = num / g;
    den = den / g;
end
end
