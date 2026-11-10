% data_types.m - strings, cell arrays, structs and function handles

% Text
name = 'World';
greeting = ['Hello, ' name '!'];
disp(greeting)
disp(upper(greeting))
words = strsplit('the quick brown fox', ' ');
[~, longest] = max(cellfun(@numel, words));
fprintf('%d words, longest is "%s"\n', numel(words), words{longest});

% Cell arrays hold mixed types
c = {42, 'text', [1 2 3]};
for k = 1:numel(c)
    fprintf('element %d is a %s\n', k, class(c{k}));
end

% Structs and struct arrays
student.name = 'Ada';
student.scores = [90 85 77];
student.average = mean(student.scores);
disp(student)

people = struct('name', {'Ann', 'Bob'}, 'age', {31, 45});
for p = people
    fprintf('%s is %d\n', p.name, p.age);
end

% Function handles and closures
scale = 3;
triple = @(v) scale * v;
disp(triple(1:4))
disp(arrayfun(@(n) factorial(n), 1:6))

% Error handling
try
    error('demo:bad', 'Something went wrong (code %d)', 7);
catch err
    fprintf('Caught: %s [%s]\n', err.message, err.identifier);
end
