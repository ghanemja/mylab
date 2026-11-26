"""Help text for frequency responses, stability margins, norms, pole-zero maps and root locus."""

_FREQ = (
    " The optional second argument is a vector of frequencies in rad per time unit, or a range {wmin, wmax}; "
    "without it the frequencies are chosen from the poles and zeros (up to the Nyquist frequency pi/Ts for "
    "a discrete-time model)."
)
_PLOT = " Without output arguments the plot is drawn; several models can be given, each with a line style."

HELP = {
    "bode": (
        ["bode(sys)", "bode(sys, w)", "bode(sys, {wmin, wmax})", "[mag, phase, wout] = bode(sys, ...)"],
        "Bode frequency response. mag is the gain (absolute, not dB) and phase the phase in degrees; both "
        "are ny-by-nu-by-nf arrays (use squeeze for a SISO model) and wout is a column of frequencies. The "
        "phase is continuous: it is assembled from the phase of each zero and pole, so integrators start at "
        "-90 degrees each and delays keep adding lag without wrapping." + _FREQ + _PLOT,
    ),
    "bodeplot": ("bodeplot(sys, ...)", "Draws the Bode diagram; same arguments as bode."),
    "bodemag": (
        ["bodemag(sys)", "bodemag(sys, w)", "bodemag(sys1, sys2, ...)"],
        "Bode magnitude plot only (in dB against frequency on a logarithmic axis).",
    ),
    "nyquist": (
        ["nyquist(sys)", "nyquist(sys, w)", "[re, im, wout] = nyquist(sys, ...)"],
        "Nyquist response: real and imaginary parts of the frequency response, each ny-by-nu-by-nf. The plot "
        "shows the curve for positive and negative frequencies and marks the critical point -1." + _FREQ,
    ),
    "nyquistplot": ("nyquistplot(sys, ...)", "Draws the Nyquist diagram; same arguments as nyquist."),
    "nichols": (
        ["nichols(sys)", "nichols(sys, w)", "[mag, phase, wout] = nichols(sys, ...)"],
        "Nichols response: gain and phase (degrees) as for bode; the plot shows gain in dB against phase." + _FREQ,
    ),
    "nicholsplot": ("nicholsplot(sys, ...)", "Draws the Nichols chart; same arguments as nichols."),
    "sigma": (
        ["sigma(sys)", "sigma(sys, w)", "[sv, wout] = sigma(sys, ...)", "sv = sigma(sys, w, type)"],
        "Singular values of the frequency response, one column per frequency, largest first. type 1 uses "
        "the inverse of the response, 2 uses I + response and 3 uses I + inverse (square models only)." + _FREQ,
    ),
    "sigmaplot": ("sigmaplot(sys, ...)", "Draws the singular-value plot; same arguments as sigma."),
    "freqresp": (
        ["H = freqresp(sys, w)", "H = freqresp(sys, w, units)", "[H, wout] = freqresp(sys)"],
        "Complex frequency response as an ny-by-nu-by-nf array: sys evaluated at s = j*w (or z = exp(j*w*Ts)), "
        "delays included. units is 'rad/TimeUnit' (default), 'Hz', 'kHz', 'rpm' and so on.",
    ),
    "evalfr": (
        "H = evalfr(sys, x)",
        "The transfer matrix at one complex number x, a point of the s-plane or z-plane, e.g. evalfr(sys, 2i).",
    ),
    "margin": (
        ["[Gm, Pm, Wcg, Wcp] = margin(sys)", "[Gm, Pm, Wcg, Wcp] = margin(mag, phase, w)", "margin(sys)"],
        "Gain and phase margins of the open loop sys. Gm is the factor by which the gain can grow before "
        "the closed loop becomes unstable (absolute, not dB) and Wcg the frequency where the phase is -180 "
        "degrees; Pm is the phase margin in degrees and Wcp the frequency where the gain is 1. When several "
        "crossovers exist the smallest margins are reported; Inf and NaN mean there is no crossover. For "
        "rational models the crossovers are roots of polynomials, so the values are exact up to rounding. "
        "Without outputs a Bode diagram with the margins in its title is drawn. With mag, phase (degrees) "
        "and w the margins are interpolated from data.",
    ),
    "allmargin": (
        ["S = allmargin(sys)", "S = allmargin(mag, phase, w, ts)"],
        "All crossover frequencies and margins of an open loop as a struct: GainMargin with GMFrequency, "
        "PhaseMargin with PMFrequency, DelayMargin with DMFrequency (seconds, or sample periods for a "
        "discrete-time model) and Stable, which tells whether the unity-feedback loop is stable.",
    ),
    "getGainCrossover": (
        "wc = getGainCrossover(sys, gain)",
        "All frequencies where the gain of a SISO model equals gain (1 for the 0 dB crossover).",
    ),
    "getPeakGain": (
        ["gpeak = getPeakGain(sys)", "[gpeak, fpeak] = getPeakGain(sys)",
         "[gpeak, fpeak] = getPeakGain(sys, tol, fband)"],
        "Largest gain over frequency (the H-infinity norm; the largest singular value for a model with "
        "several inputs or outputs) and the frequency where it occurs. fband = [fmin, fmax] restricts the "
        "search. For a rational SISO model the peak is found exactly from the derivative of the squared gain.",
    ),
    "pzmap": (
        ["pzmap(sys)", "[p, z] = pzmap(sys)"],
        "Poles and zeros of a model. Without outputs they are plotted in the complex plane, poles as x and "
        "zeros as o, with the unit circle for a discrete-time model.",
    ),
    "pzplot": ("pzplot(sys, ...)", "Draws the pole-zero map; same as pzmap without outputs."),
    "iopzmap": ("iopzmap(sys)", "Pole-zero map of a model; for a SISO model the same as pzmap."),
    "rlocus": (
        ["rlocus(sys)", "[r, k] = rlocus(sys)", "r = rlocus(sys, k)"],
        "Root locus: the closed-loop poles of the unity negative feedback loop around k*sys as the gain k "
        "goes from 0 to infinity. r has one row per branch and one column per gain. Without k the gains are "
        "chosen so that the branches are smooth and the points where branches meet are included.",
    ),
    "rlocusplot": ("rlocusplot(sys, ...)", "Draws the root locus; same arguments as rlocus."),
    "sgrid": (
        ["sgrid", "sgrid(zeta, wn)"],
        "Adds lines of constant damping ratio and circles of constant natural frequency to a pole-zero map "
        "or root locus in the s-plane.",
    ),
    "zgrid": (
        ["zgrid", "zgrid(zeta, wn)"],
        "Adds the unit circle and curves of constant damping ratio and natural frequency (in rad per sample) "
        "to a pole-zero map or root locus in the z-plane.",
    ),
}
