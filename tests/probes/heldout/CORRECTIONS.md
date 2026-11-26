# Corrections to the held-out checks

The held-out checks were written before the gap-closing work and run on both
engines for a baseline. A check is corrected only when its expectation was
wrong about MATLAB, never to make an engine pass. Each correction is listed
here with the reason.

Made right after the baseline run (26 Nov 2026), before any gap-closing work
was merged:

- `lang/integer_and_class_rules#12`: `single(1)/3 ~= 1/3` compares in single
  precision, where the two are equal. Now compares `double(single(1)/3)`.
- `linalg/held_out#12`: `svd` values were compared with `isequal`; they carry
  rounding error. Now compared with a 1e-12 tolerance.
- `stats_tbx/held_out#29`: the linkage matrix of four points has 3 columns,
  not 4.
- `signal_tbx/held_out#17`: `decimate` filters, so a constant signal is not
  reproduced to 1e-3 near the start. Tolerance 0.02.
- `image_tbx/held_out#12`: the closing example touched the image border,
  where the result depends on padding. The example now stays clear of it.
- `system/held_out#05`: `evalc` returns the captured text as its first
  output, so `[t, a, b] = evalc('deal(1, 2)')` is the correct form.

Made after the first merge:

- `quadrature/held_out#03`: `integral(@(x) abs(x), -1, 2)` was required to be
  within 1e-8 of 2.5. The default tolerance (RelTol 1e-6) only guarantees
  2.5e-6. GNU Octave's adaptive Gauss-Kronrod `quadgk` returns the same value
  as mylab (error 4.18e-7), and Octave's `integral` misses by 3.7e-7.
- `system/held_out#01`: `a = tic; b = tic; toc(a) >= toc(b)` is not always
  true, because the second `toc` is measured later. Replaced by a check that
  `toc` is not negative.
- test_one_argument_functions (again, Windows runner, Python 3.14):
  `sum([-Inf 3 Inf 1 -1 3 NaN])` gave NaN on one path and -NaN on the other.
  `Inf + -Inf` makes a NaN of its own (negative on x86), so the sum meets two
  NaNs and IEEE 754 does not say which one comes out. A NaN made inside a
  reduction (`Inf + -Inf`, `0 * Inf`) now counts towards the "two or more
  NaNs" rule; with one NaN source the sign is still compared bit for bit.
- `control_tbx/held_out#21`: the check meant "the denominator of
  tf(pid(1, 2, 3)) is [1 0], or [0 1 0] when padded", but wrote
  `norm(d - [1 0]) < 1e-12 || norm(d - [0 1 0]) < 1e-12`, which is a size
  error for a 1x3 `d` before `||` is reached, in MATLAB as in mylab. MATLAB's
  tfdata pads numerator and denominator to the same length, so `d` is
  [0 1 0] there. The two cases are now tested by length; the numerator check
  is unchanged.
- `stats_tbx/held_out#31`: fitctree on four observations was expected to
  split them, but MATLAB's default MinParentSize is 10 (a node with fewer
  observations is not split), so MATLAB grows a single leaf there: it
  predicts the first class for both points and the resubstitution loss is
  0.5, which is what mylab returned. The check now passes
  'MinParentSize', 2, which keeps what it tests (separable data classified
  without error).
