"""Help text for the hypothesis tests."""

_NV = " Options by name: 'Alpha' (default 0.05) and 'Tail' ('both', 'right' or 'left')."

HELP = {
    "ztest": ("[h, p, ci, zval] = ztest(x, m, sigma, ...)",
              "z-test of the hypothesis that the data come from a normal distribution with mean m, when the "
              "standard deviation sigma is known. h is 1 when the hypothesis is rejected." + _NV),
    "vartest": ("[h, p, ci, stats] = vartest(x, v, ...)",
                "Chi-square test of the hypothesis that normal data have variance v. ci is the confidence "
                "interval for the variance; stats has chisqstat and df." + _NV),
    "vartest2": ("[h, p, ci, stats] = vartest2(x, y, ...)",
                 "F-test of equal variances of two normal samples. ci is the interval for the variance ratio; "
                 "stats has fstat, df1 and df2." + _NV),
    "vartestn": (["p = vartestn(x)", "[p, stats] = vartestn(x, group, 'TestType', type, 'Display', 'off')"],
                 "Test of equal variances across several groups (columns of x, or groups given by a grouping "
                 "variable). 'TestType' is 'Bartlett' (default), 'LeveneQuadratic', 'LeveneAbsolute', "
                 "'BrownForsythe' or 'OBrien'. The summary table is printed unless 'Display' is 'off'."),
    "barttest": ("[ndim, prob, chisquare] = barttest(X, alpha)",
                 "Bartlett's test for the number of dimensions needed to explain the non-random variation of "
                 "the data: for k = 0, 1, ... it tests whether the remaining eigenvalues of the covariance are "
                 "equal."),
    "signtest": (["p = signtest(x)", "[p, h, stats] = signtest(x, y, ...)"],
                 "Sign test of zero median for x (or for the differences x - y; y may be a scalar). The exact "
                 "binomial distribution is used for fewer than 100 non-zero differences ('Method')." + _NV),
    "signrank": (["p = signrank(x)", "[p, h, stats] = signrank(x, y, ...)"],
                 "Wilcoxon signed rank test of zero median for x (or for x - y). The exact distribution is used "
                 "for 15 or fewer non-zero differences, a normal approximation otherwise ('Method'). stats has "
                 "signedrank and, for the approximation, zval." + _NV),
    "kstest": (["h = kstest(x)", "[h, p, ksstat, cv] = kstest(x, 'CDF', cdf, ...)"],
               "One-sample Kolmogorov-Smirnov test against the standard normal distribution, or against a "
               "distribution object, a function handle or a two-column table [x, F(x)] given as 'CDF'. 'Tail' "
               "is 'unequal', 'larger' or 'smaller'."),
    "kstest2": ("[h, p, ks2stat] = kstest2(x1, x2, ...)",
                "Two-sample Kolmogorov-Smirnov test that both samples come from the same continuous "
                "distribution (asymptotic p-value)."),
    "lillietest": ("[h, p, kstat, critval] = lillietest(x, 'Distribution', name, ...)",
                   "Lilliefors test that the data come from a normal ('normal', default), 'exponential' or "
                   "'extreme value' family with estimated parameters. p is limited to the range 0.001 to 0.5 "
                   "unless 'MCTol' asks for a simulated value."),
    "jbtest": ("[h, p, jbstat, critval] = jbtest(x, alpha)",
               "Jarque-Bera test of normality based on skewness and kurtosis. For fewer than 2000 values the "
               "null distribution is found by simulation, and p is limited to the range 0.001 to 0.5."),
    "adtest": ("[h, p, adstat, cv] = adtest(x, 'Distribution', dist, ...)",
               "Anderson-Darling goodness-of-fit test for the 'norm' (default), 'exp', 'ev', 'logn' or "
               "'weibull' family with estimated parameters, or for a fully specified distribution object."),
    "chi2gof": ("[h, p, stats] = chi2gof(x, 'Name', value, ...)",
                "Chi-square goodness-of-fit test. By default the data are tested against a normal distribution "
                "with estimated mean and standard deviation, using 10 bins and pooling the end bins until the "
                "expected counts reach 5. Options: 'NBins', 'Ctrs', 'Edges', 'CDF', 'Expected', 'NParams', "
                "'EMin', 'Frequency', 'Alpha'. p is NaN when no degrees of freedom are left. stats has "
                "chi2stat, df, edges, O and E."),
    "runstest": ("[h, p, stats] = runstest(x, v, ...)",
                 "Runs test for randomness: counts runs of values above and below v (default the mean), or "
                 "runs up and down with v = 'ud'. Exact for 50 or fewer values." + _NV),
    "fishertest": ("[h, p, stats] = fishertest(x, ...)",
                   "Fisher's exact test of no association in a 2-by-2 table of counts. stats has the odds "
                   "ratio and its confidence interval." + _NV),
    "anova2": ("[p, tbl, stats] = anova2(y, reps, displayopt)",
               "Two-way analysis of variance for balanced data: columns of y are the levels of one factor and "
               "each block of reps rows a level of the other. p holds the p-values for columns, rows and (when "
               "reps > 1) the interaction."),
    "anovan": ("[p, tbl, stats, terms] = anovan(y, group, 'Name', value, ...)",
               "N-way analysis of variance. group is a cell array of grouping variables. Options: 'model' "
               "('linear', 'interaction', 'full', an order or a terms matrix), 'sstype' (1, 2 or 3, default "
               "3), 'varnames', 'continuous' and 'display'."),
    "kruskalwallis": ("[p, tbl, stats] = kruskalwallis(x, group, displayopt)",
                      "Kruskal-Wallis test: a one-way analysis of variance on ranks, testing that the groups "
                      "come from the same distribution."),
    "friedman": ("[p, tbl, stats] = friedman(x, reps, displayopt)",
                 "Friedman's test: a non-parametric two-way analysis of variance that tests for column effects "
                 "after adjusting for row effects."),
    "multcompare": ("[c, m, h, gnames] = multcompare(stats, 'Name', value, ...)",
                    "Pairwise comparisons of group means from the stats output of anova1, anova2, anovan, "
                    "kruskalwallis or friedman. Each row of c is [group1, group2, lower, difference, upper, "
                    "p-value]; m has the estimates and standard errors. Options: 'Alpha', "
                    "'CriticalValueType' ('tukey-kramer', 'bonferroni', 'dunn-sidak', 'lsd', 'scheffe'), "
                    "'Display', 'Dimension' (anovan) and 'Estimate' (anova2)."),
    "sampsizepwr": (["n = sampsizepwr(testtype, p0, p1)", "n = sampsizepwr(testtype, p0, p1, pwr)",
                     "pwr = sampsizepwr(testtype, p0, p1, [], n)", "p1 = sampsizepwr(testtype, p0, [], pwr, n)"],
                    "Sample size, power or detectable alternative for the tests 'z', 't', 't2', 'var', 'p' and "
                    "'p2'. For the mean tests p0 is [mean, standard deviation]. Options: 'Alpha', 'Tail' and "
                    "'Ratio' (two-sample tests)."),
}
