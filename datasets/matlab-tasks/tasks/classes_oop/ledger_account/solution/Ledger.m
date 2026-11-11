classdef Ledger < handle
    %LEDGER Bank account that records each transaction.
    properties (SetAccess = private)
        Owner = ''
        Entries = zeros(0, 1)
    end
    properties (Dependent)
        Balance
    end
    methods
        function obj = Ledger(owner)
            obj.Owner = owner;
        end
        function b = get.Balance(obj)
            b = sum(obj.Entries);
        end
        function deposit(obj, amount)
            Ledger.check(amount);
            obj.Entries(end + 1, 1) = amount;
        end
        function withdraw(obj, amount)
            Ledger.check(amount);
            if amount > obj.Balance
                error('Ledger:insufficient', 'Insufficient funds.');
            end
            obj.Entries(end + 1, 1) = -amount;
        end
        function transfer(from, to, amount)
            withdraw(from, amount);
            deposit(to, amount);
        end
    end
    methods (Static, Access = private)
        function check(amount)
            if ~isnumeric(amount) || ~isscalar(amount) || ~isfinite(amount) || amount <= 0
                error('Ledger:badAmount', 'Amount must be a positive finite number.');
            end
        end
    end
end
