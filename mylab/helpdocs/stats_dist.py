"""Help text for the probability distribution functions (built from one table of families)."""

# prefix: (distribution, parameter list, short parameter description)
_FAMILIES = {
    "norm": ("normal", "mu, sigma", "mean mu and standard deviation sigma (defaults 0 and 1)"),
    "logn": ("lognormal", "mu, sigma", "mean mu and standard deviation sigma of the logarithm (defaults 0 and 1)"),
    "exp": ("exponential", "mu", "mean mu (default 1)"),
    "gam": ("gamma", "a, b", "shape a and scale b (b defaults to 1)"),
    "beta": ("beta", "a, b", "shape parameters a and b"),
    "wbl": ("Weibull", "a, b", "scale a and shape b (defaults 1 and 1)"),
    "unif": ("continuous uniform", "a, b", "lower limit a and upper limit b (defaults 0 and 1)"),
    "unid": ("discrete uniform", "n", "values 1, 2, ..., n"),
    "bino": ("binomial", "n, p", "n trials with success probability p"),
    "poiss": ("Poisson", "lambda", "mean lambda"),
    "geo": ("geometric", "p", "success probability p (the value counts the failures before the first success)"),
    "nbin": ("negative binomial", "r, p", "r successes with success probability p (the value counts the failures)"),
    "hyge": ("hypergeometric", "m, k, n", "population size m, k items with the property, n items drawn"),
    "rayl": ("Rayleigh", "b", "scale b (default 1)"),
    "ev": ("extreme value (type 1, for minima)", "mu, sigma", "location mu and scale sigma (defaults 0 and 1)"),
    "gev": ("generalized extreme value", "k, sigma, mu", "shape k, scale sigma and location mu (defaults 0, 1, 0)"),
    "gp": ("generalized Pareto", "k, sigma, theta", "shape k, scale sigma and threshold theta (defaults 0, 1, 0)"),
    "t": ("Student's t", "nu", "nu degrees of freedom"),
    "chi2": ("chi-square", "nu", "nu degrees of freedom"),
    "f": ("F", "nu1, nu2", "numerator and denominator degrees of freedom nu1 and nu2"),
    "nct": ("noncentral t", "nu, delta", "nu degrees of freedom and noncentrality delta"),
    "ncf": ("noncentral F", "nu1, nu2, delta", "degrees of freedom nu1, nu2 and noncentrality delta"),
    "ncx2": ("noncentral chi-square", "nu, delta", "nu degrees of freedom and noncentrality delta"),
}

_KINDS = {
    "pdf": ("y = {p}pdf(x, {a})", "Probability density (or probability mass) of the {d} distribution with {desc}, "
            "evaluated at each element of x. Scalar arguments expand to the size of the others; invalid "
            "parameters give NaN."),
    "cdf": (["p = {p}cdf(x, {a})", "p = {p}cdf(x, {a}, 'upper')"],
            "Cumulative distribution function of the {d} distribution with {desc}: the probability of a value "
            "less than or equal to x. With 'upper' it returns the upper tail probability, computed without "
            "cancellation."),
    "inv": ("x = {p}inv(p, {a})", "Inverse cumulative distribution function (quantile) of the {d} distribution "
            "with {desc}: the value x whose cumulative probability is p. p outside [0, 1] gives NaN."),
    "rnd": (["r = {p}rnd({a})", "r = {p}rnd({a}, m, n, ...)", "r = {p}rnd({a}, [m n ...])"],
            "Random numbers from the {d} distribution with {desc}. Without sizes the result has the size of the "
            "parameters. The generator is the one seeded by rng."),
    "stat": ("[m, v] = {p}stat({a})", "Mean m and variance v of the {d} distribution with {desc}."),
}

HELP = {}
for _p, (_d, _a, _desc) in _FAMILIES.items():
    for _k, (_usage, _text) in _KINDS.items():
        _u = [u.format(p=_p, a=_a) for u in _usage] if isinstance(_usage, list) else _usage.format(p=_p, a=_a)
        _name = "poisstat" if (_p, _k) == ("poiss", "stat") else f"{_p}{_k}"
        if _name == "poisstat":
            _u = "[m, v] = poisstat(lambda)"
        HELP[_name] = (_u, _text.format(d=_d, desc=_desc))

_FITS = {
    "exp": "the mean", "gam": "[shape, scale]", "beta": "[a, b]", "wbl": "[scale, shape]",
    "logn": "[mu, sigma] of the logarithm", "ev": "[location, scale]", "gev": "[shape, scale, location]",
    "gp": "[shape, scale] (threshold 0)", "nbin": "[r, p]", "rayl": "the scale", "poiss": "the mean",
}
for _p, _what in _FITS.items():
    _cens = _p in ("exp", "gam", "wbl", "logn", "ev")
    HELP[f"{_p}fit"] = (
        [f"[phat, pci] = {_p}fit(x)", f"[phat, pci] = {_p}fit(x, alpha)"]
        + ([f"[phat, pci] = {_p}fit(x, alpha, censoring, freq)"] if _cens else []),
        f"Maximum-likelihood estimate of {_what} of the {_FAMILIES[_p][0]} distribution from the data x, with "
        "100(1-alpha)% confidence intervals in pci (one column per parameter, lower limit in row 1; alpha "
        "defaults to 0.05)." + (" censoring marks right-censored observations with 1 and freq gives the count "
                                "of each observation." if _cens else ""))
for _p in ("norm", "exp", "gam", "beta", "wbl", "logn", "ev", "gev", "gp", "nbin"):
    HELP[f"{_p}like"] = (
        [f"nlogL = {_p}like(params, data)", f"[nlogL, avar] = {_p}like(params, data, censoring, freq)"],
        f"Negative log-likelihood of the {_FAMILIES[_p][0]} distribution with the given parameter vector for the "
        "data. The second output is the asymptotic covariance matrix of the parameter estimates (the inverse of "
        "the observed information), which is meaningful when params are the maximum-likelihood estimates.")

HELP.update({
    "normfit": (["[muhat, sigmahat, muci, sigmaci] = normfit(x)", "[...] = normfit(x, alpha)",
                 "[...] = normfit(x, alpha, censoring, freq)"],
                "Estimates of the mean and standard deviation of normal data (the standard deviation is the "
                "square root of the unbiased variance estimate) with exact 100(1-alpha)% confidence intervals. "
                "For a matrix x each column is fitted separately."),
    "unifit": (["[ahat, bhat, aci, bci] = unifit(x)", "[...] = unifit(x, alpha)"],
               "Maximum-likelihood estimates of the limits of a continuous uniform distribution (the smallest "
               "and largest observation) with confidence intervals."),
    "binofit": (["[phat, pci] = binofit(x, n)", "[phat, pci] = binofit(x, n, alpha)"],
                "Estimate of the success probability from x successes in n trials, with the exact "
                "(Clopper-Pearson) confidence interval. x and n may be arrays; pci has one row per element."),
    "pdf": (["y = pdf(name, x, A)", "y = pdf(name, x, A, B)", "y = pdf(name, x, A, B, C)", "y = pdf(pd, x)"],
            "Probability density of the distribution called name (for example 'Normal', 'Weibull', 'bino') with "
            "parameters A, B, C at the points x, or of a probability distribution object pd made by makedist "
            "or fitdist."),
    "cdf": (["p = cdf(name, x, A, ...)", "p = cdf(pd, x)", "p = cdf(..., 'upper')"],
            "Cumulative distribution function of a distribution given by name and parameters, or of a "
            "probability distribution object. 'upper' returns the upper tail probability."),
    "icdf": (["x = icdf(name, p, A, ...)", "x = icdf(pd, p)"],
             "Inverse cumulative distribution function (quantiles) of a distribution given by name and "
             "parameters, or of a probability distribution object."),
    "mle": (["phat = mle(data)", "[phat, pci] = mle(data, 'Distribution', name, ...)",
             "phat = mle(data, 'pdf', pdffun, 'Start', start, ...)"],
            "Maximum-likelihood parameter estimates with confidence intervals. The default distribution is "
            "normal (here the standard deviation is the maximum-likelihood one, with divisor n). Options: "
            "'Alpha', 'Censoring', 'Frequency', 'NTrials' (binomial), 'Start'; a custom distribution is given "
            "with 'pdf' (and 'cdf' for censored data), 'logpdf' (and 'logsf') or 'nloglf', together with "
            "'Start' and optionally 'LowerBound' and 'UpperBound'."),
    "statset": (["opts = statset", "opts = statset('Name', value, ...)", "opts = statset(oldopts, 'Name', value)"],
                "Options structure for the iterative statistics functions (for example MaxIter, TolFun, TolX "
                "and Display). Fields that are not set are empty."),
})
