data.users = struct('name', {'Ada', 'Lin'}, 'tags', {{'admin', 'dev'}, {'ops'}});
data.meta.version = 3;
data.matrix = [10 20 30; 40 50 60];
assert(strcmp(get_by_path(data, 'users(2).name'), 'Lin'), 'struct array element field');
assert(strcmp(get_by_path(data, 'users(1).tags{2}'), 'dev'), 'cell content');
assert(get_by_path(data, 'meta.version') == 3, 'nested struct');
assert(get_by_path(data, 'matrix(4)') == 50, 'linear index');
assert(isequal(get_by_path(data, ''), data), 'empty path');
v = get_by_path(data, 'users(2).tags(1)');
assert(iscell(v) && isequal(v, {'ops'}), 'paren on cell keeps cell');
assert(get_by_path({5, {6, 7}}, '{2}{1}') == 6, 'top-level cell');
bad = {'users(2', 'users(0)', 'a..b', '.a', 'a(x)', 'users (1)', 'users{}', '1abc'};
ok = true;
for k = 1:numel(bad)
    try
        get_by_path(data, bad{k});
        ok = false;
    catch err
        ok = ok && strcmp(err.identifier, 'get_by_path:badPath');
    end
end
assert(ok, 'malformed paths');
missing = {'users(3).name', 'users.name', 'meta.author', 'matrix{1}', 'meta(2)', 'users(1).name.first'};
ok = true;
for k = 1:numel(missing)
    try
        get_by_path(data, missing{k});
        ok = false;
    catch err
        ok = ok && strcmp(err.identifier, 'get_by_path:notFound');
    end
end
assert(ok, 'paths that do not exist');
ok = false;
try
    get_by_path(data, 'nothing(1).x(');
catch err
    ok = strcmp(err.identifier, 'get_by_path:badPath');
end
assert(ok, 'syntax checked before data');
