function [total, details] = price_quote(unitPrice, varargin)
%PRICE_QUOTE Order total from unit price and name-value options.
if mod(numel(varargin), 2) ~= 0
    error('price_quote:pairs', 'Options must come in name-value pairs.');
end
opts = struct('quantity', 1, 'discount', 0, 'taxrate', 0, 'round', true);
for k = 1:2:numel(varargin)
    name = lower(varargin{k});
    if ~ischar(name) || ~isfield(opts, name)
        error('price_quote:unknownOption', 'Unknown option.');
    end
    opts.(name) = varargin{k + 1};
end
subtotal = unitPrice * opts.quantity;
discount = subtotal * opts.discount;
if opts.round
    subtotal = round(subtotal * 100) / 100;
    discount = round(discount * 100) / 100;
end
tax = (subtotal - discount) * opts.taxrate;
if opts.round
    tax = round(tax * 100) / 100;
end
total = subtotal - discount + tax;
details = struct('subtotal', subtotal, 'discount', discount, 'tax', tax);
end
