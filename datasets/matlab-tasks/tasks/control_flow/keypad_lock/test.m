[s, w] = keypad_lock('4071', '4071#');
assert(strcmp(s, 'open') && w == 0, 'correct code');
[s, w] = keypad_lock('4071', '12#4071#');
assert(strcmp(s, 'open') && w == 1, 'one wrong first');
[s, w] = keypad_lock('4071', '1#2#3#4071#');
assert(strcmp(s, 'blocked') && w == 3, 'blocked after three');
[s, w] = keypad_lock('4071', '4071');
assert(strcmp(s, 'locked') && w == 0, 'not submitted');
[s, w] = keypad_lock('4071', '40*4071#');
assert(strcmp(s, 'open') && w == 0, 'clear');
[s, w] = keypad_lock('4071', '##40712#4071#');
assert(strcmp(s, 'open') && w == 1, 'empty submits ignored, extra digit is wrong');
[s, w] = keypad_lock('4071', '4071#9#9#9#');
assert(strcmp(s, 'open') && w == 0, 'keys after open ignored');
[s, w] = keypad_lock('4071', '');
assert(strcmp(s, 'locked') && w == 0, 'no keys');
ok = false;
try
    keypad_lock('4071', '40A1#');
catch err
    ok = strcmp(err.identifier, 'keypad_lock:badKey');
end
assert(ok, 'bad key');
