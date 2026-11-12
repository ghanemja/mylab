m = {};
ids = {};
kinds = {'id', 'plain', 'format', 'struct', 'nested', 'idonly', 'percent'};
for k = 1:numel(kinds)
    try
        cfh_err(kinds{k});
        m{k} = 'no error';
    catch err
        m{k} = err.message;
        ids{k} = err.identifier;
    end
end
c1 = class(err);
try
    error('Top:level', 'Value is %s', 'abc');
catch e2
    m2 = e2.message;
    id2 = e2.identifier;
end
try
    error('no id here');
catch e3
    m3 = e3.message;
    id3 = e3.identifier;
    sid3 = size(e3.identifier);
end
try
    error('MyPkg:sub:leaf', 'three part id');
catch e4
    id4 = e4.identifier;
end
flow = '';
try
    flow = [flow 'a'];
    error('x:y', 'z');
    flow = [flow 'b'];
catch
    flow = [flow 'c'];
end
flow = [flow 'd'];
try
    try
        error('in:ner', 'inner');
    catch ie
        error('out:er', 'outer saw: %s', ie.message);
    end
catch oe
    m5 = oe.message;
    id5 = oe.identifier;
end
try
    cfh_cleanup(-4);
catch we
    m6 = we.message;
    id6 = we.identifier;
end
ok7 = cfh_cleanup(9);
