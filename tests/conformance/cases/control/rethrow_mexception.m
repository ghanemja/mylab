try
    try
        error('first:id', 'first message');
    catch e
        rethrow(e);
    end
catch e2
    m1 = e2.message;
    id1 = e2.identifier;
end
try
    x = [1 2 3];
    y = x(5);
catch e5
    has5 = ~isempty(e5.message);
    c5 = class(e5.message);
    isid5 = ischar(e5.identifier);
end
try
    undefined_function_cfh_xyz(1);
catch e6
    has6 = ~isempty(e6.message);
end
le = [];
try
    error('L:e', 'last');
catch
    try, le = lasterr; catch, end
end
try
    error('re:thrown', 'again %d', 2);
catch e7
    try
        rethrow(e7);
    catch e8
        m8 = e8.message;
        id8 = e8.identifier;
    end
end
try
    rethrow(struct('message', 'from struct', 'identifier', 'st:id'));
catch e9
    m9 = e9.message;
    id9 = e9.identifier;
end
