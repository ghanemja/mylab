function [f0, A] = dominant_frequency(x, fs)
%DOMINANT_FREQUENCY Frequency and amplitude of the largest spectral peak.
N = numel(x);
if N < 2
    error('dominant_frequency:short', 'Need at least 2 samples.');
end
X = fft(x(:));
k = (1:floor(N/2))';
amp = 2 * abs(X(k + 1)) / N;
if mod(N, 2) == 0
    amp(end) = abs(X(N/2 + 1)) / N;
end
[A, i] = max(amp);
f0 = k(i) * fs / N;
end
