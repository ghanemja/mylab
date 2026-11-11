[r, tc] = frame_rms([3 -3 3 -3 1 1], 2, 2, 2);
assert(all(abs(r - [3 3 1]) < 1e-14) && all(abs(tc - [0.25 1.25 2.25]) < 1e-14), 'example');
fs = 8000;
t = (0:199)/fs;
x = (0.2 + 0.8*(t >= 0.0125)) .* sin(2*pi*440*t) + 0.05*cos(2*pi*97*t);
[r, tc] = frame_rms(x', fs, 40, 20);
assert(isequal(size(r), [1 9]) && isequal(size(tc), [1 9]), 'row outputs, frames that fit');
assert(all(abs(r - [0.15009224890569692 0.15041941131688769 0.1515983951789865 0.14626650608890795 0.4807539194027437 0.6894568367912208 0.7249663882489865 0.7323532223929318 0.7146264968482333]) < 1e-12) && all(abs(tc - [0.0024375 0.0049375 0.0074375 0.0099375 0.0124375 0.0149375 0.0174375 0.0199375 0.0224375]) < 1e-15), 'rectangular frames');
[rh, th] = frame_rms(x, fs, 50, 30, 'hann');
assert(all(abs(rh - [0.1441624413667936 0.14432546805061838 0.1936835227556954 0.6918248488641827 0.7063955301541502 0.7074668847719572]) < 1e-12) && all(abs(th - [0.0030625 0.0068125 0.0105625 0.0143125 0.0180625 0.0218125]) < 1e-15), 'Hann-weighted frames');
[r1, ~] = frame_rms([2 2 2 2 2], 1, 3, 1, 'hann');
assert(all(abs(r1 - 2) < 1e-14), 'weighted RMS of a constant is the constant');
[rw, ~] = frame_rms([0 5 0], 1, 3, 1, 'hann');
assert(abs(rw - 5) < 1e-14, 'symmetric Hann weights the centre fully, the ends not at all');
[re, te] = frame_rms([1 2], 10, 3, 1);
assert(isequal(size(re), [1 0]) && isequal(size(te), [1 0]), 'signal shorter than a frame');
threw = false;
try
    frame_rms([1 2 3 4], 1, 2, 1, 'triangle');
catch
    threw = true;
end
assert(threw, 'unknown window must throw');
