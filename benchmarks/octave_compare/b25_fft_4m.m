function r = b25_fft_4m()
x = mod((1:2^22) * 7919, 1009);
X = fft(x);
r = abs(X(2));
end
