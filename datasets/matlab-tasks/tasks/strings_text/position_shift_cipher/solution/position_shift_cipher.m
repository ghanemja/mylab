function out = position_shift_cipher(s, mode)
%POSITION_SHIFT_CIPHER Shift k-th letter by k (encode) or -k (decode).
if nargin < 2
    mode = 'encode';
end
switch mode
    case 'encode'
        sgn = 1;
    case 'decode'
        sgn = -1;
    otherwise
        error('position_shift_cipher:badMode', 'Mode must be encode or decode.');
end
out = s;
k = 0;
for i = 1:numel(s)
    c = s(i);
    if c >= 'a' && c <= 'z'
        base = double('a');
    elseif c >= 'A' && c <= 'Z'
        base = double('A');
    else
        continue
    end
    k = k + 1;
    out(i) = char(base + mod(double(c) - base + sgn * k, 26));
end
end
