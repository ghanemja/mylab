s = ValueStack();
assert(isEmpty(s) && s.Size == 0 && isequal(size(toCell(s)), [1 0]), 'new stack is empty');
push(s, 1, 'two', [3 3 3]);
assert(isequal(pop(s), [3 3 3]) && s.Size == 2, 'pop returns top');
assert(strcmp(peek(s), 'two') && s.Size == 2, 'peek does not remove');
t = s;
push(t, struct('a', 5));
assert(s.Size == 3, 'handle semantics');
push(s, []);
push(s, {});
c = toCell(s);
assert(isequal(size(c), [1 5]) && isequal(c{1}, 1) && iscell(c{5}) && isempty(c{4}), 'bottom-to-top cell, empties kept');
x = pop(s); y = pop(s); w = pop(s);
assert(iscell(x) && isempty(y) && isnumeric(y) && w.a == 5, 'pop order');
pop(s); pop(s);
assert(isEmpty(s) && islogical(isEmpty(s)), 'emptied');
threw = false;
try
    pop(s);
catch err
    threw = strcmp(err.identifier, 'ValueStack:empty');
end
assert(threw, 'pop on empty');
threw = false;
try
    peek(s);
catch err
    threw = strcmp(err.identifier, 'ValueStack:empty');
end
assert(threw, 'peek on empty');
