A vibration sensor records `N` samples `x` at sampling rate `fs` (Hz). Write a
MATLAB function `[f0, A] = dominant_frequency(x, fs)` that finds the
strongest frequency component using `fft`.

- Let `X = fft(x)`. Consider only the bins `k = 1, ..., floor(N/2)` (zero-based
  bin numbers, i.e. `X(k+1)` in MATLAB indexing). The DC bin `k = 0` is
  ignored, so a constant offset never wins.
- The single-sided amplitude of bin `k` is `2*abs(X(k+1))/N`, except that when
  `N` is even the Nyquist bin `k = N/2` uses `abs(X(k+1))/N`. With this
  scaling a sinusoid of amplitude `a` whose frequency falls exactly on a bin
  gives amplitude `a`.
- Return the frequency `f0 = k*fs/N` of the bin with the largest amplitude
  (the lowest such `k` on ties) and that amplitude `A`.
- If `x` has fewer than 2 samples, throw an error.

Example: one second of `3*sin(2*pi*50*t) + 1.5` sampled at 1000 Hz gives
`f0 = 50` and `A = 3`.
