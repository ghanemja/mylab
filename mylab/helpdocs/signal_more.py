"""Help text for the signal functions in mylab/builtins/signal_more.py (written from scratch for mylab).

The entries for butter, cheby1, freqz, xcorr, periodogram, pwelch, findpeaks, resample, decimate and
envelope replace the shorter ones in help_data.py: those functions live in signal.py but now accept the
call forms described here.
"""

_IIR_FORMS = ("With three outputs the zeros, poles and gain are returned, with four a state-space model "
              "[A, B, C, D]. A trailing 's' designs an analog filter, whose frequencies are in rad/s.")
_BANDS = ("ftype is 'low', 'high', 'bandpass' or 'stop'; a two-element Wn gives a bandpass filter of order 2n "
          "unless 'stop' is asked for.")
_SPEC_OPTS = ("window is a vector, a length (a Hamming window of that length) or [] for the default. nfft may be "
              "replaced by a vector of frequencies. Without fs the frequencies are in rad/sample and the density "
              "is per rad/sample. A trailing 'onesided', 'twosided' or 'centered' picks the frequency range; real "
              "signals give one-sided results by default.")
_WIN = "The result is a column vector."
_SFLAG = " A second input 'periodic' gives the window used for spectral analysis (one period of n+1 samples)."
_AR = ("The signal is modelled as white noise passed through 1/A(z); a is [1 a1 ... ap], e the variance of the "
       "noise and k the reflection coefficients. Each column of a matrix is treated as a signal.")
_FILTER_IN = ("The filter is given as b and a, as a matrix of second-order sections with six columns, or as a "
              "digitalFilter object from designfilt.")

HELP = {
    # -- windows --------------------------------------------------------------------------------
    "triang": ("w = triang(n)", "Triangular window of length n whose end points are not zero. " + _WIN),
    "bartlett": ("w = bartlett(n)", "Triangular window of length n with zeros at both ends. " + _WIN),
    "barthannwin": ("w = barthannwin(n)", "Modified Bartlett-Hann window of length n. " + _WIN),
    "blackmanharris": (["w = blackmanharris(n)", "w = blackmanharris(n, sflag)"],
                       "Four-term Blackman-Harris window with very low sidelobes (about -92 dB). " + _WIN + _SFLAG),
    "nuttallwin": (["w = nuttallwin(n)", "w = nuttallwin(n, sflag)"],
                   "Nuttall's four-term Blackman-Harris window. " + _WIN + _SFLAG),
    "flattopwin": (["w = flattopwin(n)", "w = flattopwin(n, sflag)"],
                   "Flat-top window: a five-term cosine sum with a very flat main lobe, used to measure "
                   "amplitudes. It takes small negative values. " + _WIN + _SFLAG),
    "bohmanwin": ("w = bohmanwin(n)", "Bohman window of length n (zero at both ends). " + _WIN),
    "parzenwin": ("w = parzenwin(n)", "Parzen (de la Vallee Poussin) window: a piecewise cubic bell. " + _WIN),
    "gausswin": (["w = gausswin(n)", "w = gausswin(n, alpha)"],
                 "Gaussian window of length n. alpha (default 2.5) is the reciprocal of the standard deviation "
                 "measured in half window lengths, so a larger alpha gives a narrower window. " + _WIN),
    "tukeywin": (["w = tukeywin(n)", "w = tukeywin(n, r)"],
                 "Tapered cosine window: flat in the middle with cosine tapers that take up the fraction r "
                 "(default 0.5) of the length. r = 0 is a rectangular window and r = 1 a Hann window. " + _WIN),
    "chebwin": (["w = chebwin(n)", "w = chebwin(n, r)"],
                "Chebyshev window whose sidelobes are all r dB (default 100) below the main lobe. " + _WIN),
    "taylorwin": (["w = taylorwin(n)", "w = taylorwin(n, nbar)", "w = taylorwin(n, nbar, sll)"],
                  "Taylor window with nbar (default 4) nearly equal sidelobes next to the main lobe at the level "
                  "sll in dB (default -30). The window is not scaled to a peak of one. " + _WIN),
    "window": ("w = window(fhandle, n, ...)",
               "Calls a window function by handle, for example window(@hamming, 64) or window(@kaiser, 64, 5). "
               "Extra inputs are passed on."),
    # -- FIR design ---------------------------------------------------------------------------------
    "fir1": (["b = fir1(n, Wn)", "b = fir1(n, Wn, ftype)", "b = fir1(n, Wn, window)", "b = fir1(..., 'noscale')"],
             "Window-method FIR filter of order n (n+1 coefficients, linear phase). Wn is the cutoff as a "
             "fraction of the Nyquist frequency; two values give a bandpass filter and more give several bands. "
             "ftype is 'low', 'high', 'bandpass', 'stop', 'DC-0' (first band stops) or 'DC-1' (first band "
             "passes). The window is a vector of n+1 values (default Hamming). The response is scaled to one "
             "in the middle of the first passband unless 'noscale' is given. A filter that passes the Nyquist "
             "frequency needs an even order, so an odd n is raised by one."),
    "fir2": (["b = fir2(n, f, m)", "b = fir2(n, f, m, window)", "b = fir2(n, f, m, npt, lap, window)"],
             "FIR filter of order n whose magnitude response follows the points (f, m): f runs from 0 to 1 "
             "(the Nyquist frequency) and repeated values mark jumps. The wanted response is sampled on npt "
             "points (default 512), jumps are smoothed over lap points, and the impulse response is windowed "
             "(default Hamming)."),
    "firls": (["b = firls(n, f, a)", "b = firls(n, f, a, w)", "b = firls(n, f, a, 'hilbert')",
               "b = firls(n, f, a, 'differentiator')"],
              "Linear-phase FIR filter of order n that minimises the weighted squared error. f lists band edges "
              "in pairs (fractions of Nyquist), a the wanted amplitude at each edge, and w one weight per band. "
              "'hilbert' and 'differentiator' give odd-symmetric filters; for a differentiator the error is "
              "weighted by 1/f^2."),
    "firpm": (["b = firpm(n, f, a)", "b = firpm(n, f, a, w)", "b = firpm(n, f, a, 'hilbert')",
               "b = firpm(n, f, a, 'differentiator')", "[b, err] = firpm(...)"],
              "Parks-McClellan equiripple FIR filter of order n: the largest weighted error is as small as "
              "possible. f lists band edges in pairs, a the wanted amplitude at each edge and w one weight per "
              "band. err is the weighted ripple. A cell {density} as last input changes the density of the "
              "frequency grid (default 16)."),
    "kaiserord": (["[n, Wn, beta, ftype] = kaiserord(f, a, dev)", "[...] = kaiserord(f, a, dev, fs)",
                   "c = kaiserord(f, a, dev, fs, 'cell')"],
                  "Estimates the order, cutoff frequencies and Kaiser window parameter of a window-method FIR "
                  "filter. f holds the band edges, a the amplitude of each band and dev the allowed deviation "
                  "in each band. Use the result as fir1(n, Wn, ftype, kaiser(n+1, beta), 'noscale'); the 'cell' "
                  "form returns those arguments in a cell array."),
    # -- IIR design ---------------------------------------------------------------------------------
    "butter": (["[b, a] = butter(n, Wn)", "[b, a] = butter(n, Wn, ftype)", "[z, p, k] = butter(...)",
                "[A, B, C, D] = butter(...)", "[...] = butter(..., 's')"],
               "Butterworth filter of order n with the -3 dB frequency Wn. " + _BANDS + " " + _IIR_FORMS),
    "cheby1": (["[b, a] = cheby1(n, Rp, Wp)", "[b, a] = cheby1(n, Rp, Wp, ftype)", "[z, p, k] = cheby1(...)",
                "[A, B, C, D] = cheby1(...)", "[...] = cheby1(..., 's')"],
               "Chebyshev type I filter of order n with Rp dB of ripple in the passband, which ends at Wp. "
               + _BANDS + " " + _IIR_FORMS),
    "cheby2": (["[b, a] = cheby2(n, Rs, Ws)", "[b, a] = cheby2(n, Rs, Ws, ftype)", "[z, p, k] = cheby2(...)",
                "[A, B, C, D] = cheby2(...)", "[...] = cheby2(..., 's')"],
               "Chebyshev type II filter of order n: flat passband and a stopband that starts at Ws and stays "
               "Rs dB down. " + _BANDS + " " + _IIR_FORMS),
    "ellip": (["[b, a] = ellip(n, Rp, Rs, Wp)", "[b, a] = ellip(n, Rp, Rs, Wp, ftype)", "[z, p, k] = ellip(...)",
               "[A, B, C, D] = ellip(...)", "[...] = ellip(..., 's')"],
              "Elliptic (Cauer) filter of order n with Rp dB of passband ripple and Rs dB of stopband "
              "attenuation; Wp is the passband edge. " + _BANDS + " " + _IIR_FORMS),
    "besself": (["[b, a] = besself(n, Wo)", "[b, a] = besself(n, Wo, ftype)", "[z, p, k] = besself(...)"],
                "Analog Bessel filter of order n, which has a nearly constant group delay up to about Wo "
                "(rad/s). " + _BANDS + " There is no digital version; use bilinear to convert it."),
    "buttord": (["[n, Wn] = buttord(Wp, Ws, Rp, Rs)", "[n, Wn] = buttord(Wp, Ws, Rp, Rs, 's')"],
                "Lowest Butterworth order that loses at most Rp dB up to Wp and at least Rs dB from Ws, with "
                "the cutoff Wn to pass to butter. Scalars describe a lowpass (Wp < Ws) or highpass filter, "
                "two-element vectors a bandpass or bandstop filter. The stopband requirement is met exactly."),
    "cheb1ord": (["[n, Wp] = cheb1ord(Wp, Ws, Rp, Rs)", "[n, Wp] = cheb1ord(Wp, Ws, Rp, Rs, 's')"],
                 "Lowest Chebyshev type I order that meets the passband ripple Rp and stopband attenuation Rs "
                 "(dB), with the passband edge to pass to cheby1."),
    "cheb2ord": (["[n, Ws] = cheb2ord(Wp, Ws, Rp, Rs)", "[n, Ws] = cheb2ord(Wp, Ws, Rp, Rs, 's')"],
                 "Lowest Chebyshev type II order that meets the passband ripple Rp and stopband attenuation Rs "
                 "(dB), with the stopband edge to pass to cheby2."),
    "ellipord": (["[n, Wp] = ellipord(Wp, Ws, Rp, Rs)", "[n, Wp] = ellipord(Wp, Ws, Rp, Rs, 's')"],
                 "Lowest elliptic filter order that meets the passband ripple Rp and stopband attenuation Rs "
                 "(dB), with the passband edge to pass to ellip."),
    "buttap": ("[z, p, k] = buttap(n)",
               "Analog Butterworth lowpass prototype of order n with cutoff 1 rad/s: no zeros, poles on the "
               "unit circle in the left half plane, gain 1."),
    "cheb1ap": ("[z, p, k] = cheb1ap(n, Rp)",
                "Analog Chebyshev type I lowpass prototype with Rp dB of passband ripple and passband edge "
                "1 rad/s."),
    "cheb2ap": ("[z, p, k] = cheb2ap(n, Rs)",
                "Analog Chebyshev type II lowpass prototype with a stopband Rs dB down that starts at 1 rad/s."),
    "ellipap": ("[z, p, k] = ellipap(n, Rp, Rs)",
                "Analog elliptic lowpass prototype with Rp dB of passband ripple, Rs dB of stopband attenuation "
                "and passband edge 1 rad/s."),
    "besselap": ("[z, p, k] = besselap(n)",
                 "Analog Bessel lowpass prototype of order n, scaled so that its magnitude approaches that of "
                 "a Butterworth filter of the same order at high and low frequencies."),
    "lp2lp": (["[bt, at] = lp2lp(b, a, Wo)", "[At, Bt, Ct, Dt] = lp2lp(A, B, C, D, Wo)"],
              "Moves the cutoff of an analog lowpass prototype (cutoff 1 rad/s) to Wo rad/s."),
    "lp2hp": (["[bt, at] = lp2hp(b, a, Wo)", "[At, Bt, Ct, Dt] = lp2hp(A, B, C, D, Wo)"],
              "Turns an analog lowpass prototype into a highpass filter with cutoff Wo rad/s."),
    "lp2bp": (["[bt, at] = lp2bp(b, a, Wo, Bw)", "[At, Bt, Ct, Dt] = lp2bp(A, B, C, D, Wo, Bw)"],
              "Turns an analog lowpass prototype into a bandpass filter centred on Wo with bandwidth Bw "
              "(rad/s). The order doubles."),
    "lp2bs": (["[bt, at] = lp2bs(b, a, Wo, Bw)", "[At, Bt, Ct, Dt] = lp2bs(A, B, C, D, Wo, Bw)"],
              "Turns an analog lowpass prototype into a bandstop filter centred on Wo with bandwidth Bw "
              "(rad/s). The order doubles."),
    "bilinear": (["[bd, ad] = bilinear(b, a, fs)", "[zd, pd, kd] = bilinear(z, p, k, fs)",
                  "[Ad, Bd, Cd, Dd] = bilinear(A, B, C, D, fs)", "[...] = bilinear(..., fs, fp)"],
                 "Converts an analog filter to a digital one with the substitution s = 2 fs (z-1)/(z+1). "
                 "Row vectors are taken as numerator and denominator, column vectors as zeros and poles. "
                 "With fp (Hz) the frequency axis is pre-warped so that the responses agree exactly at fp."),
    "impinvar": (["[bz, az] = impinvar(b, a, fs)", "[bz, az] = impinvar(b, a, fs, tol)"],
                 "Digital filter whose impulse response is the analog filter's impulse response sampled at "
                 "fs (default 1 Hz) and scaled by 1/fs. tol (default 0.001) decides when two poles count as "
                 "repeated."),
    "designfilt": (["d = designfilt(response, Name, Value, ...)"],
                   "Designs a digital filter and returns a digitalFilter object. The response is 'lowpassfir', "
                   "'highpassfir', 'bandpassfir', 'bandstopfir', 'lowpassiir', 'highpassiir', 'bandpassiir', "
                   "'bandstopiir', 'differentiatorfir', 'hilbertfir' or 'arbmagfir'. Describe the filter with "
                   "'FilterOrder' and cutoff frequencies ('CutoffFrequency', 'HalfPowerFrequency', or "
                   "'PassbandFrequency' and 'StopbandFrequency', with a 1 or 2 suffix for band filters), or "
                   "leave the order out and give the band edges with 'PassbandRipple' and "
                   "'StopbandAttenuation' (dB) for the smallest filter that meets them. 'DesignMethod' picks "
                   "'window', 'equiripple', 'ls', 'kaiserwin', 'butter', 'cheby1', 'cheby2' or 'ellip'; "
                   "'SampleRate' gives frequencies in Hz. The object has the properties Coefficients (a row "
                   "of FIR taps or a matrix of second-order sections) and the design values, and works with "
                   "filter, filtfilt, freqz, grpdelay, phasez, impz, stepz, zplane, filtord, tf, zpk, ss, "
                   "isstable and islinphase."),
    "lowpass": (["y = lowpass(x, wpass)", "y = lowpass(x, fpass, fs)", "y = lowpass(..., Name, Value)",
                 "[y, d] = lowpass(...)"],
                "Filters x with a lowpass filter whose passband ends at wpass (a fraction of Nyquist, or Hz "
                "with fs) and removes the filter delay. 'Steepness' (0.5 to 1, default 0.85) sets how narrow "
                "the transition band is, 'StopbandAttenuation' (default 60 dB) how far the stopband is "
                "pushed down and 'ImpulseResponse' ('auto', 'fir' or 'iir') the kind of filter. d is the "
                "digitalFilter that was used. Columns of a matrix are filtered separately."),
    "highpass": (["y = highpass(x, wpass)", "y = highpass(x, fpass, fs)", "y = highpass(..., Name, Value)",
                  "[y, d] = highpass(...)"],
                 "Filters x with a highpass filter whose passband starts at wpass and removes the filter "
                 "delay. The options are those of lowpass."),
    "bandpass": (["y = bandpass(x, wpass)", "y = bandpass(x, fpass, fs)", "y = bandpass(..., Name, Value)",
                  "[y, d] = bandpass(...)"],
                 "Filters x with a bandpass filter that keeps the range wpass = [low high] and removes the "
                 "filter delay. The options are those of lowpass; 'Steepness' may have one value per edge."),
    "bandstop": (["y = bandstop(x, wpass)", "y = bandstop(x, fpass, fs)", "y = bandstop(..., Name, Value)",
                  "[y, d] = bandstop(...)"],
                 "Filters x with a bandstop filter that removes the range wpass = [low high] and compensates "
                 "for the filter delay. The options are those of lowpass."),
    # -- representations ------------------------------------------------------------------------------
    "tf2sos": (["sos = tf2sos(b, a)", "[sos, g] = tf2sos(b, a)", "sos = tf2sos(b, a, order)"],
               "Second-order sections of the digital filter b/a: each row is [b0 b1 b2 1 a1 a2]. Poles are "
               "paired with the zeros nearest to them and the sections are ordered with the poles nearest the "
               "unit circle last ('up', the default) or first ('down'). Without g the gain is put into the "
               "first section."),
    "zp2sos": (["sos = zp2sos(z, p, k)", "[sos, g] = zp2sos(z, p, k)", "sos = zp2sos(z, p, k, order)"],
               "Second-order sections from zeros, poles and gain; see tf2sos for the layout and ordering."),
    "ss2sos": (["sos = ss2sos(A, B, C, D)", "[sos, g] = ss2sos(A, B, C, D)", "sos = ss2sos(A, B, C, D, iu)"],
               "Second-order sections of a state-space model (input iu when there are several)."),
    "sos2tf": (["[b, a] = sos2tf(sos)", "[b, a] = sos2tf(sos, g)"],
               "Multiplies out second-order sections into one numerator and denominator; g is an overall gain."),
    "sos2zp": (["[z, p, k] = sos2zp(sos)", "[z, p, k] = sos2zp(sos, g)"],
               "Zeros, poles and gain of a filter given as second-order sections."),
    "sosfilt": (["y = sosfilt(sos, x)", "y = sosfilt(sos, x, dim)"],
                "Filters x through a cascade of second-order sections, along the first dimension that is not "
                "one or along dim."),
    "tf2zpk": ("[z, p, k] = tf2zpk(b, a)",
               "Zeros, poles and gain of a digital filter whose coefficients multiply 1, z^-1, z^-2, ...; the "
               "shorter vector is padded with zeros at the end."),
    "tf2zp": ("[z, p, k] = tf2zp(b, a)",
              "Zeros, poles and gain of a transfer function in descending powers of s. Each row of a matrix b "
              "is one output; its zeros fill a column of z, padded with Inf."),
    "zp2tf": ("[b, a] = zp2tf(z, p, k)",
              "Numerator and denominator from zeros, poles and gain. Columns of z with one gain each give one "
              "numerator row per output."),
    "tf2ss": ("[A, B, C, D] = tf2ss(b, a)",
              "State-space model of the transfer function b/a in controller canonical form."),
    "ss2tf": (["[b, a] = ss2tf(A, B, C, D)", "[b, a] = ss2tf(A, B, C, D, iu)"],
              "Transfer function of a state-space model from input iu (default 1); b has one row per output."),
    "zp2ss": ("[A, B, C, D] = zp2ss(z, p, k)", "State-space model from zeros, poles and gain."),
    "ss2zp": (["[z, p, k] = ss2zp(A, B, C, D)", "[z, p, k] = ss2zp(A, B, C, D, iu)"],
              "Zeros, poles and gain of a state-space model from input iu (default 1)."),
    "poly2rc": (["k = poly2rc(a)", "[k, r0] = poly2rc(a, efinal)"],
                "Reflection coefficients of the prediction polynomial a (the step-down recursion). r0 is the "
                "zero-lag autocorrelation that goes with the final prediction error efinal."),
    "rc2poly": (["a = rc2poly(k)", "[a, efinal] = rc2poly(k, r0)"],
                "Prediction polynomial from reflection coefficients (the step-up recursion)."),
    "tf2latc": (["k = tf2latc(b)", "[k, v] = tf2latc(b, a)", "k = tf2latc(1, a)"],
                "Lattice coefficients of an FIR filter b, of the all-pole filter 1/a, or lattice (k) and "
                "ladder (v) coefficients of the filter b/a."),
    "latc2tf": (["[b, a] = latc2tf(k)", "[b, a] = latc2tf(k, v)", "[b, a] = latc2tf(k, 'allpole')"],
                "Transfer function of a lattice filter: FIR by default, all-pole with 'allpole', all-pass with "
                "'allpass', or lattice-ladder when v is given."),
    "latcfilt": (["[f, g] = latcfilt(k, x)", "[f, g] = latcfilt(k, v, x)", "[f, g] = latcfilt(k, 1, x)"],
                 "Filters x with a lattice filter. f is the forward output and g the backward output; with "
                 "ladder coefficients v the filter is the lattice-ladder (pole-zero) form, with v = 1 all-pole."),
    # -- analysis -------------------------------------------------------------------------------------
    "freqz": (["h = freqz(b, a)", "[h, w] = freqz(b, a, n)", "[h, w] = freqz(b, a, n, 'whole')",
               "[h, f] = freqz(b, a, n, fs)", "h = freqz(b, a, w)", "h = freqz(b, a, f, fs)",
               "[h, w] = freqz(sos, n)", "[h, w] = freqz(d, n)", "freqz(...)"],
              "Frequency response of a digital filter at n points (default 512) from 0 to pi, or round the "
              "whole circle with 'whole'. A vector in place of n gives the frequencies to use (rad/sample, or "
              "Hz with fs). " + _FILTER_IN + " Without outputs the magnitude in dB and the phase are plotted."),
    "grpdelay": (["[gd, w] = grpdelay(b, a, n)", "[gd, f] = grpdelay(b, a, n, fs)", "gd = grpdelay(b, a, w)",
                  "[gd, w] = grpdelay(sos, n)", "[gd, w] = grpdelay(d, n)"],
                 "Group delay of a digital filter in samples: minus the slope of the phase. The inputs are "
                 "those of freqz. Points where the response is zero get a delay of 0."),
    "phasez": (["[phi, w] = phasez(b, a, n)", "[phi, f] = phasez(b, a, n, fs)", "[phi, w] = phasez(sos, n)"],
               "Unwrapped phase response of a digital filter in radians. The inputs are those of freqz."),
    "impz": (["[h, t] = impz(b, a)", "[h, t] = impz(b, a, n)", "[h, t] = impz(b, a, n, fs)", "[h, t] = impz(sos)",
              "[h, t] = impz(d)", "impz(...)"],
             "Impulse response of a digital filter. n is the number of samples (chosen from the poles when "
             "left out) or a vector of sample numbers; t holds the times (samples, or seconds with fs). "
             "Without outputs a stem plot is drawn."),
    "stepz": (["[s, t] = stepz(b, a)", "[s, t] = stepz(b, a, n)", "[s, t] = stepz(b, a, n, fs)"],
              "Step response of a digital filter; the inputs are those of impz."),
    "freqs": (["h = freqs(b, a, w)", "[h, w] = freqs(b, a)", "[h, w] = freqs(b, a, n)", "freqs(...)"],
              "Frequency response of the analog filter b(s)/a(s) at the angular frequencies w (rad/s), or at n "
              "(default 200) logarithmically spaced frequencies chosen from the poles and zeros."),
    "zplane": (["zplane(z, p)", "zplane(b, a)", "zplane(sos)", "zplane(d)"],
               "Plots zeros (o) and poles (x) with the unit circle. Column vectors are zeros and poles, row "
               "vectors numerator and denominator coefficients."),
    "filtord": (["n = filtord(b, a)", "n = filtord(sos)", "n = filtord(d)"],
                "Order of a digital filter: the highest delay with a nonzero coefficient."),
    "fftfilt": (["y = fftfilt(b, x)", "y = fftfilt(b, x, n)", "y = fftfilt(d, x)"],
                "FIR filtering of x with the coefficients b; the result equals filter(b, 1, x). The block "
                "length n is accepted for compatibility."),
    # -- correlation ----------------------------------------------------------------------------------
    "xcorr": (["r = xcorr(x, y)", "r = xcorr(x)", "r = xcorr(..., maxlag)", "r = xcorr(..., scaleopt)",
               "[r, lags] = xcorr(...)"],
              "Cross-correlation r(m) = sum of x(n+m) conj(y(n)) for lags -(N-1) to N-1, or the "
              "autocorrelation of x. A shorter signal is padded with zeros. maxlag limits the lags; scaleopt "
              "is 'none', 'biased' (divide by N), 'unbiased' (divide by N-|m|) or 'normalized'/'coeff' (one "
              "at zero lag for an autocorrelation). For a matrix with N columns the result has N^2 columns: "
              "column (i-1)N+j correlates column i with column j. lags is a row vector."),
    "xcov": (["c = xcov(x, y)", "c = xcov(x)", "c = xcov(..., maxlag)", "c = xcov(..., scaleopt)",
              "[c, lags] = xcov(...)"],
             "Cross-covariance: the cross-correlation of the signals after their means are removed. The "
             "options are those of xcorr."),
    "xcorr2": (["c = xcorr2(a, b)", "c = xcorr2(a)"],
               "Two-dimensional cross-correlation of matrices a and b (the autocorrelation with one input); "
               "the result has size(a) + size(b) - 1 elements each way."),
    "cconv": (["c = cconv(a, b)", "c = cconv(a, b, n)"],
              "Circular convolution modulo n: the ordinary convolution wrapped onto n samples. Without n the "
              "result is the ordinary convolution."),
    # -- spectra --------------------------------------------------------------------------------------
    "periodogram": (["pxx = periodogram(x)", "[pxx, w] = periodogram(x, window, nfft)",
                     "[pxx, f] = periodogram(x, window, nfft, fs)", "[...] = periodogram(..., freqrange)",
                     "[...] = periodogram(..., 'power')", "[pxx, f, pxxc] = periodogram(..., 'ConfidenceLevel', p)"],
                    "Power spectral density from one windowed FFT of the signal. " + _SPEC_OPTS + " 'power' "
                    "returns the power in each bin instead of a density. Columns of a matrix are separate "
                    "signals. Without outputs the estimate is plotted in dB."),
    "pwelch": (["pxx = pwelch(x)", "[pxx, w] = pwelch(x, window, noverlap, nfft)",
                "[pxx, f] = pwelch(x, window, noverlap, nfft, fs)", "[...] = pwelch(..., freqrange)",
                "[...] = pwelch(..., 'power')", "[...] = pwelch(..., 'maxhold')",
                "[pxx, f, pxxc] = pwelch(..., 'ConfidenceLevel', p)"],
               "Welch's power spectral density: the average of the periodograms of overlapping windowed "
               "segments. By default the signal is cut into eight Hamming-windowed segments that overlap by "
               "half. " + _SPEC_OPTS + " 'power' returns power instead of density; 'maxhold' and 'minhold' "
               "keep the largest or smallest value over the segments instead of the mean."),
    "cpsd": (["pxy = cpsd(x, y)", "[pxy, f] = cpsd(x, y, window, noverlap, nfft, fs)"],
             "Cross power spectral density of x and y by Welch's method: the average of conj(X) .* Y over the "
             "segments, so its phase is the phase by which y lags x. The inputs are those of pwelch."),
    "mscohere": (["cxy = mscohere(x, y)", "[cxy, f] = mscohere(x, y, window, noverlap, nfft, fs)"],
                 "Magnitude-squared coherence |Pxy|^2 / (Pxx Pyy): between 0 and 1, it tells how well y is "
                 "predicted from x by a linear system at each frequency. The inputs are those of pwelch."),
    "tfestimate": (["txy = tfestimate(x, y)", "[txy, f] = tfestimate(x, y, window, noverlap, nfft, fs)",
                    "[...] = tfestimate(..., 'Estimator', 'H2')"],
                   "Transfer function estimate from the input x to the output y: Pxy ./ Pxx ('H1', the "
                   "default) or Pyy ./ Pyx ('H2'). The inputs are those of pwelch."),
    "pmtm": (["pxx = pmtm(x)", "[pxx, f] = pmtm(x, nw, nfft, fs)", "[...] = pmtm(..., method)"],
             "Thomson multitaper power spectral density using 2 nw - 1 Slepian tapers (nw defaults to 4). The "
             "method is 'adapt' (adaptive weights, the default), 'unity' or 'eigen'."),
    "pyulear": (["pxx = pyulear(x, order)", "[pxx, f] = pyulear(x, order, nfft, fs)"],
                "Power spectral density from an autoregressive model fitted with the Yule-Walker equations "
                "(see aryule). nfft defaults to 256."),
    "pburg": (["pxx = pburg(x, order)", "[pxx, f] = pburg(x, order, nfft, fs)"],
              "Power spectral density from an autoregressive model fitted by Burg's method (see arburg)."),
    "pcov": (["pxx = pcov(x, order)", "[pxx, f] = pcov(x, order, nfft, fs)"],
             "Power spectral density from an autoregressive model fitted by the covariance method (see arcov)."),
    "pmcov": (["pxx = pmcov(x, order)", "[pxx, f] = pmcov(x, order, nfft, fs)"],
              "Power spectral density from an autoregressive model fitted by the modified covariance method "
              "(see armcov)."),
    "stft": (["s = stft(x)", "[s, f, t] = stft(x, fs)", "[s, f, t] = stft(x, fs, Name, Value)"],
             "Short-time Fourier transform: one column of s per windowed segment. The options are 'Window' "
             "(default a periodic Hann window of 128 samples), 'OverlapLength' (default 75% of the window), "
             "'FFTLength' and 'FrequencyRange' ('centered', the default, 'twosided' or 'onesided'). f and t "
             "are column vectors; t marks the middle of each segment. With an even FFT length the centred "
             "range ends at the positive Nyquist frequency."),
    "istft": (["x = istft(s)", "[x, t] = istft(s, fs, Name, Value)"],
              "Inverse short-time Fourier transform by weighted overlap-add. Give the same 'Window', "
              "'OverlapLength', 'FFTLength' and 'FrequencyRange' that were used for stft. 'Method' is 'wola' "
              "(default) or 'ola'; 'ConjugateSymmetric', true returns a real signal."),
    "pspectrum": (["p = pspectrum(x)", "[p, f] = pspectrum(x, fs)", "[p, f] = pspectrum(x, fs, Name, Value)"],
                  "Power spectrum of a signal on 4096 frequencies, using a Kaiser window. 'Leakage' (0 to 1, "
                  "default 0.5) trades resolution against sidelobes, 'FrequencyLimits' restricts the range, "
                  "'FrequencyResolution' sets the resolution bandwidth and 'TwoSided' asks for negative "
                  "frequencies too. A sine wave of amplitude A shows a peak of A^2/2. The 'spectrogram' and "
                  "'persistence' types are not supported."),
    "bandpower": (["p = bandpower(x)", "p = bandpower(x, fs, freqrange)", "p = bandpower(pxx, f, 'psd')",
                   "p = bandpower(pxx, f, freqrange, 'psd')"],
                  "Average power of a signal (the mean of its squared magnitude), or the power in the "
                  "frequency range [low high], found by summing a periodogram. A spectrum pxx with its "
                  "frequencies f may be given instead of the signal."),
    "obw": (["bw = obw(x, fs)", "[bw, flo, fhi, power] = obw(x, fs, freqrange, p)", "bw = obw(pxx, f)"],
            "Occupied bandwidth: the width of the band that holds p percent (default 99) of the power, with "
            "equal amounts left out below flo and above fhi."),
    "powerbw": (["bw = powerbw(x, fs)", "[bw, flo, fhi, power] = powerbw(x, fs, freqrange, r)",
                 "bw = powerbw(pxx, f)"],
                "Width of the band around the spectral peak in which the power stays within r dB (default "
                "3.01, the half-power bandwidth) of the peak."),
    "meanfreq": (["freq = meanfreq(x, fs)", "[freq, power] = meanfreq(x, fs, freqrange)", "freq = meanfreq(pxx, f)"],
                 "Mean frequency of the power spectrum: the power-weighted average frequency."),
    "medfreq": (["freq = medfreq(x, fs)", "[freq, power] = medfreq(x, fs, freqrange)", "freq = medfreq(pxx, f)"],
                "Median frequency: the frequency that splits the power of the spectrum into two equal halves."),
    "enbw": (["bw = enbw(window)", "bw = enbw(window, fs)"],
             "Equivalent noise bandwidth of a window, in frequency bins or in Hz with a sample rate."),
    "snr": (["r = snr(x, y)", "r = snr(x)", "r = snr(x, fs, n)", "[r, noisepow] = snr(...)"],
            "Signal-to-noise ratio in dB. With two inputs x is the signal and y the noise. With one signal "
            "the largest spectral peak is taken as the signal and the noise is what remains after the first "
            "n (default 6) harmonics and the mean are removed, estimated from a Kaiser-windowed periodogram."),
    "thd": (["r = thd(x)", "r = thd(x, fs, n)", "[r, harmpow, harmfreq] = thd(...)"],
            "Total harmonic distortion in dB relative to the fundamental: the power of harmonics 2 to n "
            "(default 6) over the power of the fundamental. harmpow and harmfreq list the fundamental and "
            "the harmonics."),
    "sinad": (["r = sinad(x)", "r = sinad(x, fs)", "[r, totdistpow] = sinad(...)"],
              "Ratio in dB of the fundamental to everything else except the mean (noise and distortion)."),
    "sfdr": (["r = sfdr(x)", "r = sfdr(x, fs)", "[r, spurpow, spurfreq] = sfdr(...)"],
             "Spurious-free dynamic range: the ratio in dB of the fundamental to the largest other peak."),
    # -- transforms -----------------------------------------------------------------------------------
    "dct": (["y = dct(x)", "y = dct(x, n)", "y = dct(x, n, dim)", "y = dct(..., 'Type', t)"],
            "Discrete cosine transform, scaled so that it is orthogonal. Type 2 is the default; types 1, 3 "
            "and 4 are available. x is padded or cut to n samples; matrices are transformed column by column."),
    "idct": (["x = idct(y)", "x = idct(y, n)", "x = idct(y, n, dim)", "x = idct(..., 'Type', t)"],
             "Inverse discrete cosine transform: idct(dct(x)) returns x."),
    "dct2": (["B = dct2(A)", "B = dct2(A, m, n)", "B = dct2(A, [m n])"],
             "Two-dimensional discrete cosine transform of a matrix, padded or cut to m-by-n first."),
    "idct2": (["A = idct2(B)", "A = idct2(B, m, n)", "A = idct2(B, [m n])"],
              "Inverse two-dimensional discrete cosine transform."),
    "czt": (["y = czt(x)", "y = czt(x, m, w, a)"],
            "Chirp z-transform: the z-transform of x at the m points a * w.^-(0:m-1). The defaults give the "
            "ordinary DFT; other values zoom in on part of the spectrum."),
    "goertzel": (["y = goertzel(x)", "y = goertzel(x, idx)", "y = goertzel(x, idx, dim)"],
                 "Discrete Fourier transform of x at the bins idx only (one-based, as in fft; they need not be "
                 "integers)."),
    "rceps": (["y = rceps(x)", "[y, ym] = rceps(x)"],
              "Real cepstrum: the inverse FFT of the logarithm of the magnitude spectrum. ym is the "
              "minimum-phase signal with the same magnitude spectrum."),
    "cceps": (["xhat = cceps(x)", "[xhat, nd] = cceps(x, n)"],
              "Complex cepstrum: the inverse FFT of the complex logarithm of the spectrum, with the phase "
              "unwrapped and its linear part (a delay of nd samples) removed."),
    # -- resampling -----------------------------------------------------------------------------------
    "resample": (["y = resample(x, p, q)", "y = resample(x, p, q, n)", "y = resample(x, p, q, n, beta)",
                  "y = resample(x, p, q, b)", "[y, b] = resample(...)", "[y, ty] = resample(x, tx, fs)",
                  "y = resample(..., 'Dimension', dim)"],
                 "Changes the sample rate by the factor p/q with a lowpass FIR filter and removes its delay; "
                 "the result has ceil(length(x) p/q) samples. The filter uses n (default 10) samples on each "
                 "side and a Kaiser window with parameter beta (default 5), or the coefficients b. n = 0 "
                 "picks the nearest sample. With sample times tx the signal is first interpolated onto a "
                 "uniform grid at fs ('linear', 'pchip' or 'spline')."),
    "decimate": (["y = decimate(x, r)", "y = decimate(x, r, n)", "y = decimate(x, r, 'fir')",
                  "y = decimate(x, r, n, 'fir')"],
                 "Lowers the sample rate of a vector by the integer r after lowpass filtering. The default "
                 "filter is a Chebyshev type I filter of order n (default 8) run forwards and backwards; the "
                 "kept samples are r apart and end with the last one. 'fir' uses a Hamming-window FIR filter "
                 "of order n (default 30) and keeps the first sample."),
    "upfirdn": (["y = upfirdn(x, h)", "y = upfirdn(x, h, p, q)"],
                "Inserts p-1 zeros between samples, filters with the FIR filter h and keeps every q-th sample. "
                "Columns of a matrix are processed separately."),
    "interp": (["y = interp(x, r)", "y = interp(x, r, n, cutoff)", "[y, b] = interp(...)"],
               "Raises the sample rate of a vector by the integer r. The original samples are kept and the new "
               "ones are computed from the 2n nearest samples (n defaults to 4), assuming the signal contains "
               "no frequencies above cutoff (default 0.5) times the original Nyquist frequency. b is the "
               "interpolation filter."),
    "intfilt": (["b = intfilt(l, p, alpha)", "b = intfilt(l, n, 'Lagrange')"],
                "Interpolation FIR filter for a signal with l-1 zeros inserted between samples: the "
                "least-squares filter that uses the 2p nearest samples of a signal band-limited to alpha times "
                "Nyquist, or the filter for Lagrange polynomial interpolation of order n."),
    # -- waveforms ------------------------------------------------------------------------------------
    "rectpuls": (["y = rectpuls(t)", "y = rectpuls(t, w)"],
                 "Rectangular pulse of width w (default 1) centred on 0: one for -w/2 <= t < w/2, else zero."),
    "tripuls": (["y = tripuls(t)", "y = tripuls(t, w)", "y = tripuls(t, w, s)"],
                "Triangular pulse of width w (default 1) centred on 0. The skew s (-1 to 1) moves the peak to "
                "t = s w/2."),
    "gauspuls": (["yi = gauspuls(t, fc, bw)", "[yi, yq, ye] = gauspuls(t, fc, bw, bwr)",
                  "tc = gauspuls('cutoff', fc, bw, bwr, tpe)"],
                 "Gaussian-modulated sinusoid with centre frequency fc (default 1000 Hz) and fractional "
                 "bandwidth bw (default 0.5) measured bwr dB (default -6) down. yq is the quadrature pulse and "
                 "ye the envelope. The 'cutoff' form returns the time at which the envelope has fallen by tpe "
                 "dB (default -60)."),
    "gmonopuls": (["y = gmonopuls(t, fc)", "tc = gmonopuls('cutoff', fc)"],
                  "Gaussian monopulse (the derivative of a Gaussian) with centre frequency fc (default 1000 "
                  "Hz) and unit peak. The 'cutoff' form returns the time between its two extremes."),
    "pulstran": (["y = pulstran(t, d, func, ...)", "y = pulstran(t, d, p, fs)", "y = pulstran(t, d, p, fs, method)"],
                 "Pulse train: the sum of copies of a pulse delayed by the values in d. The pulse is a function "
                 "such as @rectpuls (extra inputs are passed to it) or a sampled pulse p at rate fs. A "
                 "two-column d gives a delay and an amplitude for each copy."),
    "diric": ("y = diric(x, n)",
              "Dirichlet (periodic sinc) function sin(n x/2) / (n sin(x/2)), with its limit at multiples of "
              "2 pi."),
    "vco": (["y = vco(x, fc, fs)", "y = vco(x, [fmin fmax], fs)"],
            "Voltage-controlled oscillator: a cosine whose frequency follows x (between -1 and 1), from 0 to "
            "2 fc, or from fmin to fmax. fs defaults to 2 and fc to fs/4."),
    "buffer": (["y = buffer(x, n)", "y = buffer(x, n, p)", "y = buffer(x, n, p, opt)", "[y, z, opt] = buffer(...)"],
               "Cuts a signal into frames of n samples, one per column, overlapping by p samples (or skipping "
               "-p samples when p is negative). The first frame starts with p zeros unless opt is 'nodelay' "
               "or a vector of p starting samples. The last frame is padded with zeros; with two outputs only "
               "full frames are returned and z holds the samples left over."),
    # -- measurements ---------------------------------------------------------------------------------
    "peak2peak": (["y = peak2peak(x)", "y = peak2peak(x, dim)"],
                  "Difference between the largest and smallest value along the first dimension that is not "
                  "one, or along dim."),
    "peak2rms": (["y = peak2rms(x)", "y = peak2rms(x, dim)"],
                 "Ratio of the largest magnitude to the root-mean-square value (the crest factor)."),
    "rssq": (["y = rssq(x)", "y = rssq(x, dim)"], "Square root of the sum of squared magnitudes."),
    "findpeaks": (["pks = findpeaks(y)", "[pks, locs] = findpeaks(y)", "[pks, locs] = findpeaks(y, x)",
                   "[pks, locs] = findpeaks(y, fs)", "[pks, locs, w, p] = findpeaks(...)",
                   "[...] = findpeaks(..., Name, Value)", "findpeaks(...)"],
                  "Local maxima of a vector: samples larger than both neighbours (a flat top counts once, at "
                  "its first sample; the end points never count). locs are indices, or the matching values of "
                  "x. w are the widths and p the prominences (how far a peak stands above the higher of the "
                  "two valleys that separate it from taller peaks). Options: 'MinPeakHeight', "
                  "'MinPeakProminence', 'Threshold' (height above both neighbours), 'MinPeakDistance' "
                  "(peaks closer than or exactly this far from a taller peak are dropped), 'MinPeakWidth', "
                  "'MaxPeakWidth', 'WidthReference' ('halfprom' or 'halfheight'), 'SortStr' ('ascend' or "
                  "'descend') and 'NPeaks'. Without outputs the peaks are plotted."),
    "envelope": (["[up, lo] = envelope(x)", "[up, lo] = envelope(x, fl, 'analytic')",
                  "[up, lo] = envelope(x, wl, 'rms')", "[up, lo] = envelope(x, np, 'peak')"],
                 "Upper and lower envelopes of a signal. The default uses the magnitude of the analytic signal "
                 "(the mean is removed first and added back). 'analytic' with fl uses an FIR Hilbert filter of "
                 "that length, 'rms' a moving root-mean-square over wl samples, and 'peak' a spline through "
                 "the local maxima (and minima) that are at least np samples apart."),
    "risetime": (["r = risetime(x)", "r = risetime(x, fs)", "r = risetime(x, t)",
                  "[r, lt, ut, ll, ul] = risetime(..., 'PercentReferenceLevels', [10 90])"],
                 "Time each rising edge of a two-level waveform takes to go from the lower to the upper "
                 "reference level (10% and 90% of the way between the two state levels by default). lt and "
                 "ut are the crossing times, ll and ul the levels. 'StateLevels' gives the low and high levels "
                 "instead of estimating them."),
    "falltime": (["f = falltime(x)", "f = falltime(x, fs)", "[f, lt, ut, ll, ul] = falltime(...)"],
                 "Time each falling edge of a two-level waveform takes to go from the upper to the lower "
                 "reference level; the options are those of risetime."),
    "overshoot": (["os = overshoot(x)", "[os, oslev, osinst] = overshoot(x, fs)"],
                  "Overshoot of each transition of a two-level waveform as a percentage of the difference "
                  "between the state levels: how far the signal goes beyond the level it settles to, looked "
                  "for just after the transition."),
    "undershoot": (["us = undershoot(x)", "[us, uslev, usinst] = undershoot(x, fs)"],
                   "Undershoot of each transition as a percentage of the difference between the state levels: "
                   "how far the signal swings back towards the level it came from just after the transition."),
    "statelevels": (["levels = statelevels(x)", "levels = statelevels(x, nbins)"],
                    "Low and high levels of a two-level waveform, taken as the most frequent values in the "
                    "lower and upper halves of a histogram with nbins (default 100) bins."),
    "midcross": (["c = midcross(x)", "c = midcross(x, fs)", "[c, midlev] = midcross(...)"],
                 "Times at which a two-level waveform crosses the level halfway between its two states."),
    "findchangepts": (["ipt = findchangepts(x)", "[ipt, residual] = findchangepts(x, Name, Value)"],
                      "Indices at which a signal changes most abruptly: x(ipt) is the first sample of a new "
                      "segment. 'Statistic' is 'mean' (default), 'rms', 'std' or 'linear'; 'MaxNumChanges' "
                      "allows more than one change, 'MinThreshold' keeps every change that lowers the "
                      "residual by more than the threshold, and 'MinDistance' sets the shortest segment."),
    "finddelay": (["d = finddelay(x, y)", "d = finddelay(x, y, maxlag)"],
                  "Delay of y relative to x in samples, taken from the peak of their cross-correlation: "
                  "positive when y is a delayed copy of x."),
    "alignsignals": (["[xa, ya] = alignsignals(x, y)", "[xa, ya, d] = alignsignals(x, y, maxlag)",
                      "[...] = alignsignals(..., 'truncate')"],
                     "Lines up two signals by putting zeros in front of the one that is ahead; d is the delay "
                     "found by finddelay. 'truncate' keeps the original lengths."),
    "dtw": (["dist = dtw(x, y)", "[dist, ix, iy] = dtw(x, y)", "[...] = dtw(x, y, maxsamp)",
             "[...] = dtw(..., metric)"],
            "Dynamic time warping: the smallest total distance between x and y when samples may be repeated "
            "to stretch either signal. ix and iy are the matched sample numbers. maxsamp limits how far the "
            "two indices may drift apart; metric is 'euclidean', 'absolute' or 'squared'. Columns of matrices "
            "are samples of a multichannel signal."),
    "sgolay": (["b = sgolay(order, framelen)", "[b, g] = sgolay(order, framelen, weights)"],
               "Savitzky-Golay smoothing filters: row k of b gives the fitted value at position k of a frame of "
               "framelen samples from a least-squares polynomial of the given order; the middle row is the "
               "filter used away from the ends. Column n+1 of g gives the n-th derivative of the fit at the "
               "middle of the frame (to be scaled by n! and the sample spacing)."),
    "hampel": (["y = hampel(x)", "y = hampel(x, k)", "y = hampel(x, k, nsigma)",
               "[y, j, xmedian, xsigma] = hampel(...)"],
               "Outlier removal: a sample that differs from the median of its 2k+1 neighbours (k defaults to "
               "3) by more than nsigma (default 3) scaled median absolute deviations is replaced by that "
               "median. j marks the replaced samples."),
    "levinson": (["a = levinson(r)", "[a, e, k] = levinson(r, n)"],
                 "Levinson-Durbin recursion: the prediction polynomial of order n (default length(r)-1) from "
                 "the autocorrelation r, with the prediction error e and the reflection coefficients k."),
    "lpc": (["a = lpc(x)", "[a, g] = lpc(x, p)"],
            "Linear prediction coefficients of order p (default length(x)-1) from the autocorrelation of x; "
            "g is the variance of the prediction error."),
    "aryule": ("[a, e, k] = aryule(x, p)",
               "Autoregressive model of order p from the Yule-Walker equations (biased autocorrelation). " + _AR),
    "arburg": ("[a, e, k] = arburg(x, p)",
               "Autoregressive model of order p by Burg's method, which minimises the forward and backward "
               "prediction errors and always gives a stable model. " + _AR),
    "arcov": ("[a, e] = arcov(x, p)",
              "Autoregressive model of order p by the covariance method: least squares on the forward "
              "prediction error without windowing the data."),
    "armcov": ("[a, e] = armcov(x, p)",
               "Autoregressive model of order p by the modified covariance method: least squares on the "
               "forward and backward prediction errors."),
}
