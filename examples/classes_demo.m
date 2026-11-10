% classes_demo.m - using the BankAccount class defined in BankAccount.m
acct = BankAccount('Ada', 100);
acct.deposit(50);
acct.withdraw(30);
disp(acct.summary)

shared = acct;          % handle class: both names refer to the same account
shared.deposit(5);
fprintf('Balance seen through acct: %.2f\n', acct.balance);

try
    acct.withdraw(1000);
catch err
    fprintf('Refused (%s): %s\n', err.identifier, err.message);
end
