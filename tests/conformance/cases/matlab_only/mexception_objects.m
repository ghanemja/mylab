% MException construction is not available in Octave 8; expectations are in overrides.json.
me = MException('my:id', 'Count %d of %d', 2, 5);
m2 = me.message;
id2 = me.identifier;
c2 = class(me);
try
    throw(me);
catch e3
    m3 = e3.message;
    id3 = e3.identifier;
end
try
    error(me);
catch e4
    m4 = e4.message;
end
s7 = isa(me, 'MException');
try
    error('a:b', 'msg');
catch e8
    c8 = class(e8);
    st8 = isstruct(e8.stack);
end
me2 = MException('top:id', 'no format args');
m9 = me2.message;
try
    error('Some:id', 'text');
catch e10
    r10 = getReport(e10);
    c10 = ischar(r10);
end
