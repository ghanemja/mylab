assert(strcmp(position_shift_cipher('abc'), 'bdf'), 'basic');
assert(strcmp(position_shift_cipher('Hi, there'), 'Ik, wljxl'), 'case and punctuation');
assert(strcmp(position_shift_cipher('bdf', 'decode'), 'abc'), 'decode');
assert(strcmp(position_shift_cipher('xyz', 'encode'), 'yac'), 'wrap around');
assert(strcmp(position_shift_cipher('a1 b2'), 'b1 d2'), 'digits not counted');
msg = 'Meet me at the old mill at 6 PM, bring thirty-two lanterns!';
enc = position_shift_cipher(msg);
assert(~strcmp(enc, msg) && isequal(size(enc), size(msg)), 'encoding changes text');
assert(strcmp(position_shift_cipher(enc, 'decode'), msg), 'round trip beyond 26 letters');
assert(strcmp(position_shift_cipher(repmat('a', 1, 27)), ['bcdefghijklmnopqrstuvwxyza' 'b']), 'shift 26 wraps to itself');
ok = false;
try
    position_shift_cipher('abc', 'scramble');
catch err
    ok = strcmp(err.identifier, 'position_shift_cipher:badMode');
end
assert(ok, 'bad mode');
