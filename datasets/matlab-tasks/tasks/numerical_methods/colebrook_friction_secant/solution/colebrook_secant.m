function lambda = colebrook_secant(Re, rr)
%COLEBROOK_SECANT Darcy friction factor from the Colebrook equation.
if any(Re(:) <= 2300)
    error('colebrook_secant:laminar', 'Re must exceed 2300 (turbulent flow).');
end
lambda = zeros(size(Re));
for k = 1:numel(Re)
    g = @(s) s + 2*log10(rr/3.7 + 2.51*s/Re(k));
    s0 = 5;
    s1 = 10;
    g0 = g(s0);
    g1 = g(s1);
    for it = 1:100
        if g1 == g0
            break
        end
        s2 = s1 - g1*(s1 - s0)/(g1 - g0);
        s0 = s1;
        g0 = g1;
        s1 = s2;
        g1 = g(s1);
        if abs(s1 - s0) < 1e-14*abs(s1)
            break
        end
    end
    lambda(k) = 1/s1^2;
end
end
