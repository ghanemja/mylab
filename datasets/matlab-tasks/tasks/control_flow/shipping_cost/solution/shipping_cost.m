function cost = shipping_cost(region, kg)
%SHIPPING_COST Shipping cost from region code and weight.
billed = ceil(kg);
switch upper(region)
    case 'HOME'
        cost = 4 + 0.5 * billed;
    case {'EU', 'UK', 'CH'}
        cost = 9 + 1.2 * billed;
    case {'US', 'CA'}
        cost = 15 + 2 * billed;
    otherwise
        error('shipping_cost:unknownRegion', 'Unknown region "%s".', region);
end
end
