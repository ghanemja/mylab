G = rule_automaton(90, logical([0 0 0 1 0 0 0]), 3);
assert(islogical(G) && isequal(size(G), [4 7]), 'size and class');
assert(isequal(double(G), [0 0 0 1 0 0 0; 0 0 1 0 1 0 0; 0 1 0 0 0 1 0; 1 0 1 0 1 0 1]), 'rule 90');
G = rule_automaton(30, [0 0 0 0 1 0 0 0 0], 4);
assert(isequal(double(G), [0 0 0 0 1 0 0 0 0; 0 0 0 1 1 1 0 0 0; 0 0 1 1 0 0 1 0 0; ...
    0 1 1 0 1 1 1 1 0; 1 1 0 0 1 0 0 0 1]), 'rule 30 with numeric input');
G = rule_automaton(110, logical([1 0 1 1 0 0 1]), 2);
assert(isequal(double(G), [1 0 1 1 0 0 1; 1 1 1 1 0 1 1; 1 0 0 1 1 1 1]), 'rule 110 at the edges');
init = logical([1 0 1 1 0]);
assert(isequal(rule_automaton(204, init, 5), repmat(init, 6, 1)), 'identity rule');
assert(isequal(rule_automaton(30, init, 0), init), 'zero steps');
G = rule_automaton(255, false(1, 3), 1);
assert(all(G(2, :)), 'rule 255 turns everything on');
G = rule_automaton(1, true(1, 1), 2);
assert(isequal(double(G), [1; 0; 1]), 'single cell, pattern 0 maps to 1');
