classdef Ratio
    %RATIO Exact fraction kept in lowest terms.
    properties (SetAccess = private)
        Num = 0
        Den = 1
    end
    methods
        function obj = Ratio(n, d)
            if nargin < 1
                n = 0;
            end
            if nargin < 2
                d = 1;
            end
            if d == 0
                error('Ratio:zeroDenominator', 'Denominator must be nonzero.');
            end
            g = gcd(n, d);
            if g == 0
                g = 1;
            end
            n = n / g;
            d = d / g;
            if d < 0
                n = -n;
                d = -d;
            end
            obj.Num = n;
            obj.Den = d;
        end
        function r = plus(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            r = Ratio(a.Num * b.Den + b.Num * a.Den, a.Den * b.Den);
        end
        function r = minus(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            r = Ratio(a.Num * b.Den - b.Num * a.Den, a.Den * b.Den);
        end
        function r = times(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            r = Ratio(a.Num * b.Num, a.Den * b.Den);
        end
        function r = mtimes(a, b)
            r = times(a, b);
        end
        function r = mrdivide(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            if b.Num == 0
                error('Ratio:zeroDenominator', 'Division by zero.');
            end
            r = Ratio(a.Num * b.Den, a.Den * b.Num);
        end
        function r = uminus(a)
            r = Ratio(-a.Num, a.Den);
        end
        function tf = eq(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            tf = a.Num == b.Num && a.Den == b.Den;
        end
        function tf = lt(a, b)
            a = Ratio.lift(a); b = Ratio.lift(b);
            tf = a.Num * b.Den < b.Num * a.Den;
        end
        function x = double(q)
            x = q.Num / q.Den;
        end
    end
    methods (Static, Access = private)
        function r = lift(x)
            if isa(x, 'Ratio')
                r = x;
            else
                r = Ratio(x, 1);
            end
        end
    end
end
