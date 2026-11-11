function [names, units] = restock_list(items)
%RESTOCK_LIST Names of items at or below reorder level and units to order.
names = cell(0, 1);
units = zeros(0, 1);
if isempty(items)
    return
end
qty = [items.qty];
lvl = [items.reorder_level];
low = qty <= lvl;
if ~any(low)
    return
end
allNames = {items.name};
[names, order] = sort(allNames(low));
need = 2 * lvl(low) - qty(low);
names = reshape(names, [], 1);
units = reshape(need(order), [], 1);
end
