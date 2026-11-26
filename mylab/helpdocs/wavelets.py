"""Help text for the wavelet functions in mylab/builtins/wavelets.py (written from scratch for mylab)."""

_WNAME = ("The wavelet is 'haar', 'dbN' (Daubechies, N vanishing moments), 'symN' (symlets) or 'coifN' "
          "(coiflets, N = 1..5); two filter vectors may be given instead of a name.")
_MODE = (" The signal is extended at its ends with the current extension mode (see dwtmode); 'mode', m "
         "overrides it for one call.")
_BOOK = ("c holds the coarsest approximation followed by the details from coarse to fine, and l the length "
         "of each of those pieces followed by the length of the signal.")
_BOOK2 = ("C is a row vector holding the coarsest approximation followed by the horizontal, vertical and "
          "diagonal details of each level from coarse to fine (each stored column by column); S lists the "
          "size of the approximation, of the details at each level, and of the image.")

HELP = {
    "wfilters": (["[LoD, HiD, LoR, HiR] = wfilters(wname)", "[F1, F2] = wfilters(wname, type)"],
                 "The four filters of an orthogonal wavelet: low- and high-pass filters for decomposition "
                 "(LoD, HiD) and for reconstruction (LoR, HiR). type picks a pair: 'd' decomposition, 'r' "
                 "reconstruction, 'l' the two low-pass and 'h' the two high-pass filters. " + _WNAME),
    "orthfilt": (["[LoD, HiD, LoR, HiR] = orthfilt(w)"],
                 "The four filters of the orthogonal wavelet whose scaling filter is w. w is first scaled so "
                 "that the low-pass filters sum to sqrt(2)."),
    "qmf": (["y = qmf(x)", "y = qmf(x, p)"],
            "Quadrature mirror filter: x reversed with the sign of every second element changed (the even "
            "positions when p is even, the odd ones otherwise)."),
    "wrev": ("y = wrev(x)", "Reverses a vector."),
    "dwtmode": (["dwtmode(mode)", "dwtmode", "mode = dwtmode('status', 'nodisp')"],
                "Sets or shows the way signals are extended at their ends by the wavelet transforms: 'sym' "
                "(mirror, repeating the end sample; the default), 'symw' (mirror about the end sample), "
                "'asym' and 'asymw' (the same with a sign change), 'zpd' (zeros), 'sp0' (repeat the end "
                "value), 'sp1' (continue the end slope), 'ppd' (periodic) and 'per' (periodic, giving "
                "half as many coefficients as samples)."),
    "wextend": (["y = wextend('1D', mode, x, n)", "y = wextend('1D', mode, x, n, loc)",
                 "Y = wextend('2D', mode, X, [nr nc])", "Y = wextend('addrow', mode, X, n)",
                 "Y = wextend('addcol', mode, X, n)"],
                "Extends a vector or matrix by n samples with one of the modes listed under dwtmode. loc is "
                "'l' (left or top), 'r' (right or bottom) or 'b' (both, the default)."),
    "dyaddown": (["y = dyaddown(x)", "y = dyaddown(x, evenodd)", "Y = dyaddown(X, evenodd, type)"],
                 "Keeps every second element: the even-numbered ones by default, the odd-numbered ones when "
                 "evenodd is odd. For a matrix type is 'c' (columns), 'r' (rows) or 'm' (both)."),
    "dyadup": (["y = dyadup(x)", "y = dyadup(x, evenodd)", "Y = dyadup(X, evenodd, type)"],
               "Inserts zeros between elements. By default the samples go to the even positions (a zero comes "
               "first and last); with an even evenodd they go to the odd positions."),
    "wkeep": (["y = wkeep(x, n)", "y = wkeep(x, n, opt)", "Y = wkeep(X, [nr nc])"],
              "Keeps n elements from the middle of a vector, or from the left ('l'), the right ('r') or "
              "starting at a given index. For a matrix the central part of the given size is kept."),
    "wmaxlev": ("n = wmaxlev(len, wname)",
                "Largest sensible decomposition level for a signal of the given length: the deepest level at "
                "which at least one coefficient is free of end effects."),
    "dwt": (["[cA, cD] = dwt(x, wname)", "[cA, cD] = dwt(x, LoD, HiD)", "[cA, cD] = dwt(..., 'mode', m)"],
            "One level of the discrete wavelet transform: approximation coefficients cA (low-pass filtered) "
            "and detail coefficients cD (high-pass filtered), each kept at every second sample. Each has "
            "floor((n + f - 1)/2) elements for a signal of n samples and filters of f taps. " + _WNAME + _MODE),
    "idwt": (["x = idwt(cA, cD, wname)", "x = idwt(cA, cD, LoR, HiR)", "x = idwt(cA, cD, wname, n)",
              "x = idwt(..., 'mode', m)"],
             "One level of the inverse wavelet transform. n keeps the n central samples (use the length of "
             "the original signal when it was odd). Either coefficient vector may be [] to rebuild from the "
             "other one alone."),
    "wavedec": (["[c, l] = wavedec(x, n, wname)", "[c, l] = wavedec(x, n, LoD, HiD)"],
                "Wavelet decomposition of a vector to level n: the transform is applied n times to the "
                "approximation. " + _BOOK + _MODE),
    "waverec": (["x = waverec(c, l, wname)", "x = waverec(c, l, LoR, HiR)"],
                "Rebuilds a signal from the decomposition made by wavedec."),
    "appcoef": (["a = appcoef(c, l, wname)", "a = appcoef(c, l, wname, n)"],
                "Approximation coefficients at level n (default the coarsest level) of a wavedec "
                "decomposition; level 0 gives the signal."),
    "detcoef": (["d = detcoef(c, l)", "d = detcoef(c, l, n)", "[d1, d2] = detcoef(c, l, [1 2])"],
                "Detail coefficients at level n (default the coarsest level) of a wavedec decomposition. "
                "Several levels come back as several outputs, or as a cell array with one output."),
    "wrcoef": (["x = wrcoef(type, c, l, wname)", "x = wrcoef(type, c, l, wname, n)"],
               "Rebuilds the part of the signal carried by one branch of a wavedec decomposition: the "
               "approximation ('a') or the details ('d') at level n. The approximation at the coarsest level "
               "plus the details of all levels give back the signal."),
    "upcoef": (["y = upcoef(type, x, wname)", "y = upcoef(type, x, wname, n, len)"],
               "Rebuilds n levels upwards from coefficients x taken as approximation ('a') or detail ('d') "
               "coefficients; len keeps the central part of the result."),
    "wenergy": ("[ea, ed] = wenergy(c, l)",
                "Percentage of the energy of a wavedec decomposition held by the approximation (ea) and by "
                "the details of each level (ed)."),
    "dwt2": (["[cA, cH, cV, cD] = dwt2(X, wname)", "[cA, cH, cV, cD] = dwt2(X, LoD, HiD)"],
             "One level of the two-dimensional wavelet transform: the approximation and the horizontal, "
             "vertical and diagonal details. cH is low-pass along the rows and high-pass along the columns, "
             "so it responds to horizontal edges." + _MODE),
    "idwt2": (["X = idwt2(cA, cH, cV, cD, wname)", "X = idwt2(cA, cH, cV, cD, LoR, HiR)",
               "X = idwt2(..., [rows cols])"],
              "One level of the inverse two-dimensional wavelet transform; the size keeps the central part."),
    "wavedec2": (["[C, S] = wavedec2(X, n, wname)", "[C, S] = wavedec2(X, n, LoD, HiD)"],
                 "Wavelet decomposition of a matrix to level n. " + _BOOK2),
    "waverec2": (["X = waverec2(C, S, wname)", "X = waverec2(C, S, LoR, HiR)"],
                 "Rebuilds a matrix from the decomposition made by wavedec2."),
    "appcoef2": (["A = appcoef2(C, S, wname)", "A = appcoef2(C, S, wname, n)"],
                 "Approximation coefficients at level n (default the coarsest) of a wavedec2 decomposition."),
    "detcoef2": (["D = detcoef2(o, C, S, n)", "[H, V, D] = detcoef2('all', C, S, n)"],
                 "Detail coefficients at level n of a wavedec2 decomposition: o is 'h', 'v' or 'd' for one "
                 "orientation, 'all' for the three, or 'compact' for one row vector."),
    "wthresh": ("y = wthresh(x, sorh, t)",
                "Thresholding with the threshold t: 's' (soft) shrinks every value towards zero by t, 'h' "
                "(hard) sets values of magnitude t or less to zero and keeps the rest."),
    "thselect": ("thr = thselect(x, rule)",
                 "Threshold for denoising data with unit noise: 'sqtwolog' (the universal threshold "
                 "sqrt(2 log n)), 'rigrsure' (minimises Stein's unbiased risk estimate), 'heursure' (a mix "
                 "of the two) or 'minimaxi' (minimax)."),
    "wnoisest": (["s = wnoisest(c, l)", "s = wnoisest(c, l, levels)"],
                 "Noise standard deviation estimated from the detail coefficients of the given levels "
                 "(default level 1): the median absolute value divided by 0.6745."),
    "ddencmp": ("[thr, sorh, keepapp] = ddencmp(in1, 'wv', x)",
                "Default values for denoising ('den') or compression ('cmp') of x with wdencmp: a threshold, "
                "soft or hard thresholding, and whether to keep the approximation."),
    "wden": (["xd = wden(x, rule, sorh, scal, n, wname)", "[xd, cxd, lxd] = wden(...)"],
             "Denoises a signal by thresholding the details of a level-n decomposition. rule is a thselect "
             "rule, sorh is 's' or 'h', and scal says how the noise level is found: 'one' (noise of unit "
             "variance), 'sln' (from the first level) or 'mln' (separately at each level)."),
    "wdencmp": (["xc = wdencmp('gbl', x, wname, n, thr, sorh, keepapp)",
                 "xc = wdencmp('lvd', x, wname, n, thr, sorh)", "[xc, cxc, lxc, perf0, perfl2] = wdencmp(...)"],
                "Denoises or compresses a signal or image by thresholding its wavelet coefficients, with one "
                "threshold for all levels ('gbl') or one per level ('lvd'; for images a 3-by-n matrix). perf0 "
                "is the percentage of coefficients set to zero and perfl2 the percentage of energy kept."),
    "wdenoise": (["xden = wdenoise(x)", "xden = wdenoise(x, level)", "xden = wdenoise(..., Name, Value)",
                  "[xden, denoisedcfs, origcfs] = wdenoise(...)"],
                 "Denoises a signal (or each column of a matrix) in the wavelet domain. The defaults are the "
                 "'sym4' wavelet, as many levels as the length allows, and the empirical Bayes method with "
                 "the posterior median. Options: 'Wavelet', 'DenoisingMethod' ('Bayes', "
                 "'UniversalThreshold', 'SURE' or 'Minimax'), 'ThresholdRule' ('Median', 'Mean', 'Soft' or "
                 "'Hard' with 'Bayes'; 'Soft' or 'Hard' otherwise) and 'NoiseEstimate' ('LevelIndependent' "
                 "or 'LevelDependent')."),
    "cwt": (["wt = cwt(x)", "[wt, f] = cwt(x, fs)", "[wt, f, coi] = cwt(x, wname, fs)",
             "[...] = cwt(..., Name, Value)"],
            "Continuous wavelet transform with an analytic wavelet: 'morse' (the default, with symmetry 3 "
            "and time-bandwidth product 60), 'amor' (Morlet) or 'bump'. wt has one row per scale and one "
            "column per sample; f lists the centre frequency of each row from high to low (cycles per "
            "sample, or Hz with fs), with 10 scales per octave. The wavelets are scaled so that a sine wave "
            "of amplitude A gives |wt| near A. coi marks, for every sample, the frequency below which end "
            "effects matter. Options: 'VoicesPerOctave', 'TimeBandwidth', 'WaveletParameters', "
            "'FrequencyLimits', 'NumOctaves' and 'ExtendSignal'."),
    "modwt": (["w = modwt(x)", "w = modwt(x, wname)", "w = modwt(x, wname, level)"],
              "Maximal-overlap discrete wavelet transform (no downsampling, periodic ends): rows 1 to J hold "
              "the wavelet coefficients of each level and the last row the scaling coefficients, all with "
              "the length of x. The default wavelet is 'sym4' and the default level floor(log2(n)). The "
              "energy of x is split between the rows."),
    "imodwt": (["x = imodwt(w)", "x = imodwt(w, wname)"],
               "Rebuilds a signal from its maximal-overlap wavelet transform."),
    "modwtmra": (["m = modwtmra(w)", "m = modwtmra(w, wname)"],
                 "Multiresolution analysis from a maximal-overlap transform: each row is the part of the "
                 "signal at one scale, and the rows add up to the signal."),
}
