function [stock, lost, orderDays] = stock_simulation(start, demand, reorderPoint, orderQty, leadTime)
%STOCK_SIMULATION Daily stock with lost sales and a reorder-point policy.
n = numel(demand);
stock = zeros(1, n);
orderDays = zeros(1, 0);
dueDay = zeros(1, 0);
lost = 0;
onHand = start;
for d = 1:n
    arriving = dueDay == d;
    onHand = onHand + orderQty * nnz(arriving);
    dueDay(arriving) = [];
    sold = min(onHand, demand(d));
    lost = lost + demand(d) - sold;
    onHand = onHand - sold;
    position = onHand + orderQty * numel(dueDay);
    if position <= reorderPoint
        orderDays(end + 1) = d; %#ok<AGROW>
        dueDay(end + 1) = d + leadTime; %#ok<AGROW>
    end
    stock(d) = onHand;
end
end
