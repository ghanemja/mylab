classdef BankAccount < handle
    % BankAccount - a small handle class used by classes_demo.m
    properties
        owner = ''
        balance = 0
    end
    properties (Dependent)
        summary
    end
    methods
        function obj = BankAccount(owner, opening)
            if nargin > 0
                obj.owner = owner;
            end
            if nargin > 1
                obj.deposit(opening);
            end
        end
        function deposit(obj, amount)
            if amount <= 0
                error('BankAccount:invalidAmount', 'Deposits must be positive.');
            end
            obj.balance = obj.balance + amount;
        end
        function withdraw(obj, amount)
            if amount > obj.balance
                error('BankAccount:insufficientFunds', 'Only %.2f available.', obj.balance);
            end
            obj.balance = obj.balance - amount;
        end
        function s = get.summary(obj)
            s = sprintf('%s: %.2f', obj.owner, obj.balance);
        end
    end
end
