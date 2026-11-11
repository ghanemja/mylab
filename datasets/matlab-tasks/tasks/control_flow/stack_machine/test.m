assert(isequal(stack_machine({'push 3', 'push 4', 'push 5', 'add', 'mul'}), 27), 'example');
assert(isequal(stack_machine({'push 2', 'push 3', 'sub'}), -1), 'sub order');
assert(isequal(stack_machine({' push 7 ', 'push -2', 'mul', 'dup'}), [-14 -14]), 'negative, whitespace, dup');
assert(isequal(stack_machine({'push 1', 'push 2', 'push 3', 'rot'}), [2 3 1]), 'rot');
assert(isequal(stack_machine({'push 1', 'push 2', 'over', 'swap'}), [1 1 2]), 'over and swap');
fact = {'push 1', 'push 5', 'dup', 'jz 14', 'over', 'over', 'mul', 'swap', ...
        'push 1', 'sub', 'rot', 'pop', 'jmp 3', 'pop'};
assert(isequal(stack_machine(fact), 120), 'loop computing 5!');
assert(isequal(stack_machine({'push 1', 'halt', 'push 2'}), 1), 'halt');
s = stack_machine({'push 4', 'pop'});
assert(isempty(s) && isequal(size(s), [1 0]), 'empty stack is 1-by-0');
s = stack_machine({'push 5', 'push 0', 'jz 5', 'push 9', 'push 6', 'push 1', 'jz 9', 'push 7', 'push 0'});
assert(isequal(s, [5 6 7 0]), 'jz taken and not taken');
ids = {};
progs = {{'push 1', 'add'}, {'push 1', 'jump 3'}, {'jmp 1'}, {'push'}, {'push 1', 'jmp 9'}};
for k = 1:numel(progs)
    try
        stack_machine(progs{k});
        ids{k} = 'none';
    catch err
        ids{k} = err.identifier;
    end
end
assert(isequal(ids, {'stack_machine:underflow', 'stack_machine:badInstruction', ...
    'stack_machine:tooManySteps', 'stack_machine:badInstruction', 'stack_machine:badInstruction'}), 'errors');
