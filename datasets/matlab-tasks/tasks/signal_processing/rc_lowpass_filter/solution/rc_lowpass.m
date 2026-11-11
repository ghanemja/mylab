function y = rc_lowpass(x, fs, fc)
%RC_LOWPASS First-order recursive low-pass filter.
y = x;
if isempty(x)
    return
end
dt = 1 / fs;
RC = 1 / (2*pi*fc);
alpha = dt / (RC + dt);
for k = 2:numel(x)
    y(k) = y(k-1) + alpha * (x(k) - y(k-1));
end
end
