Write a MATLAB function `[r, tc] = frame_rms(x, fs, L, hop, window)` that
measures the loudness of an audio signal `x` (sampled at `fs` Hz) frame by
frame.

- Frames have length `L` samples and start at samples `1, 1 + hop,
  1 + 2*hop, ...`; only frames that fit completely inside `x` are used.
- `window` is optional: `'rect'` (the default when omitted) or `'hann'`. The
  Hann window of length `L > 1` is the symmetric one,
  `w(n) = 0.5 - 0.5*cos(2*pi*n/(L - 1))` for `n = 0, ..., L-1`; for `L = 1`
  use `w = 1`. The rectangular window is all ones. Do not use `hann` or
  `hanning` (they are toolbox functions in MATLAB).
- The (weighted) RMS of a frame `s` is `sqrt(sum(w .* s.^2) / sum(w))`.
- `r` is a row vector with one RMS value per frame; `tc` is a row vector with
  the time (in seconds, the first sample at time 0) of each frame's centre,
  `(start - 1 + (L - 1)/2) / fs` where `start` is the frame's first sample.
- If `x` is shorter than `L`, both outputs are `zeros(1, 0)`.
- An unknown window name throws an error.

Example: `[r, tc] = frame_rms([3 -3 3 -3 1 1], 2, 2, 2)` gives `r = [3 3 1]`
and `tc = [0.25 1.25 2.25]`.
