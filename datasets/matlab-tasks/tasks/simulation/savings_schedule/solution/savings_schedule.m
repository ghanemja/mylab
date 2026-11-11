function b = savings_schedule(principal, annualRate, monthly, months)
%SAVINGS_SCHEDULE End-of-month balances with interest and deposits.
b = zeros(1, months);
bal = principal;
r = annualRate / 12;
for k = 1:months
    bal = bal * (1 + r) + monthly;
    b(k) = bal;
end
end
