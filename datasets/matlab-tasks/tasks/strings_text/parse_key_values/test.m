cfg = parse_key_values('name=Ada; age = 36 ;active=TRUE');
assert(isequal(fieldnames(cfg), {'name'; 'age'; 'active'}), 'field order');
assert(strcmp(cfg.name, 'Ada') && cfg.age == 36, 'char and number');
assert(islogical(cfg.active) && cfg.active, 'logical true');
cfg = parse_key_values('x=-2.5e3;;  ; flag=false; note=  hello world  ');
assert(cfg.x == -2500 && islogical(cfg.flag) && ~cfg.flag, 'number and false, empty entries');
assert(strcmp(cfg.note, 'hello world'), 'inner space kept');
cfg = parse_key_values('url=http://a.b/?q=1; empty=');
assert(strcmp(cfg.url, 'http://a.b/?q=1') && ischar(cfg.empty) && isempty(cfg.empty), 'split at first =, empty value');
cfg = parse_key_values('a=1; b=2; a=three');
assert(isequal(fieldnames(cfg), {'a'; 'b'}) && strcmp(cfg.a, 'three'), 'repeated key');
cfg = parse_key_values('  ');
assert(isstruct(cfg) && isempty(fieldnames(cfg)), 'no entries');
ids = {'', ''};
try
    parse_key_values('a=1; oops');
catch err
    ids{1} = err.identifier;
end
try
    parse_key_values('2fast=1');
catch err
    ids{2} = err.identifier;
end
assert(isequal(ids, {'parse_key_values:syntax', 'parse_key_values:badKey'}), 'errors');
