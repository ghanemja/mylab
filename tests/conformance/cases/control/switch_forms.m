r = {};
for v = {1, 'one', 2, 'two', [1 2], 3.5}
    x = v{1};
    try
        switch x
            case 1
                r{end+1} = 'num1';
            case {2, 3}
                r{end+1} = 'num23';
            case 'one'
                r{end+1} = 'str1';
            case {'two', 'three'}
                r{end+1} = 'str23';
            otherwise
                r{end+1} = 'other';
        end
    catch
        r{end+1} = 'error';
    end
end
s1 = 'none';
switch 'abc'
    case 'ABC'
        s1 = 'upper';
    case 'abc'
        s1 = 'lower';
end
s2 = 'none';
switch 5
    case 6
        s2 = 'six';
end
s3 = 0;
switch 2
    case {1, 2}
        s3 = 1;
    case 2
        s3 = 2;
end
s4 = '';
switch 'b'
    case {'a', 'b', 'c'}
        s4 = 'in';
    otherwise
        s4 = 'out';
end
s5 = 0;
switch 3
    case 1 + 2
        s5 = 1;
end
switch 'a'
    case 97
    otherwise
end
s7 = 0;
switch true
    case 1
        s7 = 1;
end
