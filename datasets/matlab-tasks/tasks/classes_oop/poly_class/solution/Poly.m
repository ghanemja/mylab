classdef Poly
    %POLY Polynomial with coefficients in ascending powers.
    properties (SetAccess = private)
        Coeffs = 0
    end
    methods
        function obj = Poly(c)
            if nargin < 1 || isempty(c)
                c = 0;
            end
            c = double(c(:).');
            last = find(c ~= 0, 1, 'last');
            if isempty(last)
                obj.Coeffs = 0;
            else
                obj.Coeffs = c(1:last);
            end
        end
        function d = degree(p)
            if isequal(p.Coeffs, 0)
                d = -1;
            else
                d = numel(p.Coeffs) - 1;
            end
        end
        function y = at(p, x)
            y = zeros(size(x));
            for k = numel(p.Coeffs):-1:1
                y = y .* x + p.Coeffs(k);
            end
        end
        function q = deriv(p)
            n = numel(p.Coeffs);
            if n <= 1
                q = Poly(0);
            else
                q = Poly(p.Coeffs(2:end) .* (1:n - 1));
            end
        end
        function r = plus(a, b)
            a = Poly.lift(a); b = Poly.lift(b);
            n = max(numel(a.Coeffs), numel(b.Coeffs));
            r = Poly([a.Coeffs, zeros(1, n - numel(a.Coeffs))] + [b.Coeffs, zeros(1, n - numel(b.Coeffs))]);
        end
        function r = uminus(a)
            r = Poly(-a.Coeffs);
        end
        function r = minus(a, b)
            r = plus(Poly.lift(a), -Poly.lift(b));
        end
        function r = mtimes(a, b)
            a = Poly.lift(a); b = Poly.lift(b);
            r = Poly(conv(a.Coeffs, b.Coeffs));
        end
        function tf = eq(a, b)
            a = Poly.lift(a); b = Poly.lift(b);
            tf = isequal(a.Coeffs, b.Coeffs);
        end
        function [q, r] = divmod(p, d)
            p = Poly.lift(p); d = Poly.lift(d);
            dd = degree(d);
            if dd < 0
                error('Poly:divideByZero', 'Division by the zero polynomial.');
            end
            rr = p.Coeffs;
            lead = d.Coeffs(end);
            dp = degree(p);
            if dp < dd
                q = Poly(0);
                r = p;
                return
            end
            qc = zeros(1, dp - dd + 1);
            for k = dp:-1:dd
                coef = rr(k + 1) / lead;
                qc(k - dd + 1) = coef;
                rr(k - dd + 1:k + 1) = rr(k - dd + 1:k + 1) - coef * d.Coeffs;
                rr(k + 1) = 0;
            end
            q = Poly(qc);
            r = Poly(rr(1:max(dd, 1)));
        end
    end
    methods (Static, Access = private)
        function p = lift(x)
            if isa(x, 'Poly')
                p = x;
            else
                p = Poly(x);
            end
        end
    end
end
