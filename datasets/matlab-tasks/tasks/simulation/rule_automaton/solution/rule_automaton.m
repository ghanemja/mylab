function G = rule_automaton(rule, init, steps)
%RULE_AUTOMATON Evolve an elementary cellular automaton with zero boundaries.
n = numel(init);
lut = logical(bitget(rule, 1:8));   % lut(p+1) is the new state for pattern p
G = false(steps + 1, n);
G(1, :) = logical(init(:).');
for s = 1:steps
    prev = G(s, :);
    left = [false, prev(1:end-1)];
    right = [prev(2:end), false];
    p = 4 * left + 2 * prev + right;
    G(s + 1, :) = lut(p + 1);
end
end
