function T = order_costs(orders, prices)
%ORDER_COSTS Attach unit prices and costs to orders, keeping row order.
[found, loc] = ismember(orders.Product, prices.Product);
unitPrice = nan(height(orders), 1);
unitPrice(found) = prices.UnitPrice(loc(found));
T = orders(:, {'OrderID', 'Product', 'Qty'});
T.UnitPrice = unitPrice;
T.Cost = T.Qty .* unitPrice;
end
