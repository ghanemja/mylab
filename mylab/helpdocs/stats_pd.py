"""Help text for distribution objects, multivariate distributions and distribution plots."""

HELP = {
    "makedist": (["pd = makedist(name)", "pd = makedist(name, 'param', value, ...)", "makedist"],
                 "Create a probability distribution object, for example makedist('Normal', 'mu', 1, 'sigma', 2) "
                 "or makedist('Weibull', 'A', 2, 'B', 3). The parameters are properties of the object (pd.mu). "
                 "Use it with pdf, cdf, icdf, random, mean, std, var, median, iqr and truncate. Without inputs "
                 "it lists the distribution names."),
    "fitdist": (["pd = fitdist(x, name)", "pd = fitdist(x, name, 'Name', value, ...)",
                 "[pdca, gn, gl] = fitdist(x, name, 'By', group)"],
                "Fit a distribution to the data vector x by maximum likelihood and return a probability "
                "distribution object (for 'Normal' the standard deviation is the unbiased sample one). "
                "paramci(pd) gives confidence intervals and negloglik(pd) the negative log-likelihood. Options: "
                "'Censoring', 'Frequency', 'NTrials' (binomial), 'Theta' (generalized Pareto), 'By' (one fit "
                "per group, returned in a cell array). name 'Kernel' gives a kernel density estimate with the "
                "options 'Kernel', 'Width' and 'Support'."),
    "mvncdf": (["p = mvncdf(X)", "p = mvncdf(X, mu, Sigma)", "p = mvncdf(xl, xu, mu, Sigma)"],
               "Cumulative probability of the multivariate normal distribution at each row of X, or the "
               "probability of the box between the rows of xl and xu."),
    "mvtpdf": ("y = mvtpdf(X, C, df)",
               "Density of the multivariate t distribution with correlation matrix C and df degrees of freedom "
               "at each row of X."),
    "mvtcdf": (["p = mvtcdf(X, C, df)", "p = mvtcdf(xl, xu, C, df)"],
               "Cumulative probability of the multivariate t distribution with correlation matrix C and df "
               "degrees of freedom, at each row of X or over the box between xl and xu."),
    "mvtrnd": (["R = mvtrnd(C, df)", "R = mvtrnd(C, df, n)"],
               "Random vectors (one per row) from the multivariate t distribution with correlation matrix C "
               "and df degrees of freedom."),
    "wishrnd": (["W = wishrnd(Sigma, df)", "[W, D] = wishrnd(Sigma, df, D)"],
                "A random matrix from the Wishart distribution with covariance Sigma and df degrees of freedom. "
                "D is the Cholesky factor of Sigma, which can be passed back to save time."),
    "iwishrnd": (["W = iwishrnd(Tau, df)", "[W, DI] = iwishrnd(Tau, df, DI)"],
                 "A random matrix from the inverse Wishart distribution with parameter matrix Tau and df degrees "
                 "of freedom."),
    "copularnd": (["U = copularnd('Gaussian', rho, n)", "U = copularnd('t', rho, nu, n)",
                   "U = copularnd(family, alpha, n)"],
                  "Random points in the unit cube from a copula: Gaussian or t with correlation matrix rho (a "
                  "scalar means a 2-by-2 matrix), or the bivariate 'Clayton', 'Frank' or 'Gumbel' copula with "
                  "parameter alpha."),
    "ecdf": (["[f, x] = ecdf(y)", "[f, x, flo, fup] = ecdf(y, 'Name', value, ...)", "ecdf(y)"],
             "Empirical cumulative distribution function (Kaplan-Meier estimate). f starts at 0 and x repeats "
             "the smallest value first, so stairs(x, f) draws the step function. Options: 'Censoring', "
             "'Frequency', 'Alpha', 'Function' ('cdf', 'survivor' or 'cumulative hazard') and 'Bounds'. flo and "
             "fup are Greenwood confidence bounds. Without outputs the function is plotted."),
    "cdfplot": (["cdfplot(x)", "[h, stats] = cdfplot(x)"],
                "Plot the empirical cumulative distribution function of x. stats holds the minimum, maximum, "
                "mean, median and standard deviation of the sample."),
    "histfit": (["histfit(data)", "histfit(data, nbins)", "histfit(data, nbins, dist)", "h = histfit(...)"],
                "Histogram of the data with the density of a fitted distribution drawn over it (normal by "
                "default; dist is any name fitdist accepts). The number of bins defaults to the square root of "
                "the number of values. h holds the bar and the line."),
    "normplot": (["normplot(x)", "h = normplot(x)"],
                 "Normal probability plot: the sorted data against normal quantiles on a probability scale, with "
                 "a reference line through the first and third quartiles. Points close to the line indicate "
                 "normally distributed data."),
    "probplot": (["probplot(y)", "probplot(dist, y)", "probplot(dist, y, cens, freq)", "h = probplot(...)"],
                 "Probability plot of the data against a distribution: 'normal' (default), 'exponential', "
                 "'extreme value', 'lognormal', 'rayleigh', 'weibull', 'logistic', 'loglogistic' or 'half "
                 "normal'. Censored observations and frequencies are supported."),
    "qqplot": (["qqplot(x)", "qqplot(x, y)", "qqplot(x, pd)", "qqplot(x, y, pvec)", "h = qqplot(...)"],
               "Quantile-quantile plot: quantiles of x against standard normal quantiles, against the "
               "quantiles of a second sample y, or against a probability distribution object, with a "
               "reference line through the quartiles."),
}
