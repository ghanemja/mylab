function [p, dp] = horner_eval(c, x)
%HORNER_EVAL Value and derivative of a polynomial by Horner's rule.
p = zeros(size(x));
dp = zeros(size(x));
for k = 1:numel(c)
    dp = dp .* x + p;
    p = p .* x + c(k);
end
end
