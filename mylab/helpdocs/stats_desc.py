"""Help text for descriptive statistics, grouped summaries and resampling."""

HELP = {
    "grpstats": (["means = grpstats(X, group)", "[a, b, ...] = grpstats(X, group, whichstats)",
                  "tbl = grpstats(tbl, groupvars, whichstats, 'DataVars', vars)"],
                 "Summary statistics by group. whichstats is a name or a cell array of names and function "
                 "handles: 'mean', 'sem', 'numel', 'gname', 'std', 'var', 'min', 'max', 'range', 'median', "
                 "'mode', 'meanci' and 'predci'; one output is returned per statistic, with one row per group. "
                 "group may be a cell array of several grouping variables. For a table the result is a table "
                 "with the group variables, GroupCount and one variable per statistic and data variable."),
    "crosstab": (["tbl = crosstab(x1, x2, ...)", "[tbl, chi2, p, labels] = crosstab(x1, x2, ...)"],
                 "Cross-tabulation: counts of every combination of the values of the grouping variables. chi2 "
                 "and p test independence; labels is a cell array with the value names of each variable in "
                 "its own column."),
    "dummyvar": ("D = dummyvar(group)",
                 "Indicator (0/1) columns for a grouping variable of positive integers or a categorical array; "
                 "several grouping columns give several blocks of indicator columns."),
    "gscatter": (["gscatter(x, y, g)", "gscatter(x, y, g, clr, sym, siz, doleg, xnam, ynam)", "h = gscatter(...)"],
                 "Scatter plot of y against x with one colour (and marker) per group of g, and a legend."),
    "tiedrank": (["r = tiedrank(x)", "[r, tieadj] = tiedrank(x)"],
                 "Ranks of the values, with tied values given their average rank (each column of a matrix is "
                 "ranked separately). tieadj is the tie adjustment used by rank tests."),
    "trimmean": (["m = trimmean(X, percent)", "m = trimmean(X, percent, flag, dim)"],
                 "Mean after discarding the highest and lowest percent/2 of the values. flag is 'round' "
                 "(default), 'floor' or 'weighted'."),
    "moment": (["m = moment(X, order)", "m = moment(X, order, dim)", "m = moment(X, order, 'all')"],
               "Central moment of the given order: the mean of (x - mean(x)).^order."),
    "partialcorr": (["rho = partialcorr(X)", "rho = partialcorr(X, Z)", "[rho, pval] = partialcorr(X, Y, Z, ...)"],
                    "Partial correlation: the correlation between pairs of variables after removing the linear "
                    "effect of the others (or of the variables in Z). Options: 'Type' ('Pearson' or "
                    "'Spearman'), 'Rows' and 'Tail'."),
    "corrcov": ("[R, sigma] = corrcov(C)",
                "Correlation matrix R and standard deviations sigma from a covariance matrix C."),
    "nancov": (["C = nancov(X)", "C = nancov(X, Y)", "C = nancov(..., 'pairwise')"],
               "Covariance ignoring NaN: by default rows with any NaN are removed; with 'pairwise' each entry "
               "uses all rows where both variables are present."),
    "bootstrp": (["bootstat = bootstrp(nboot, bootfun, d1, ...)", "[bootstat, bootsam] = bootstrp(...)"],
                 "Bootstrap: draw nboot samples with replacement from the rows of the data and apply bootfun to "
                 "each. Every row of bootstat is one result; bootsam holds the row numbers used."),
    "bootci": (["ci = bootci(nboot, bootfun, d1, ...)", "ci = bootci(nboot, {bootfun, d1, ...}, 'Type', type)"],
               "Bootstrap confidence interval for the statistic computed by bootfun. 'Type' is 'bca' (default), "
               "'norm', 'per', 'cper' or 'stud'; 'Alpha' sets the level (default 0.05)."),
    "jackknife": ("jackstat = jackknife(jackfun, X, ...)",
                  "Jackknife: apply jackfun to the data with each row left out in turn; one result per row."),
    "cvpartition": (["c = cvpartition(n, 'KFold', k)", "c = cvpartition(n, 'HoldOut', p)",
                     "c = cvpartition(n, 'LeaveOut')", "c = cvpartition(group, 'KFold', k)"],
                    "Random partition of n observations for cross-validation. With a grouping variable the "
                    "partition is stratified (each class is spread evenly) unless 'Stratify' is false. Use "
                    "training(c, i) and test(c, i) for the logical masks of fold i; properties: NumTestSets, "
                    "TrainSize, TestSize, NumObservations and Type."),
    "crossval": (["vals = crossval(fun, X, ...)", "mse = crossval('mse', X, y, 'Predfun', predfun)",
                  "mcr = crossval('mcr', X, y, 'Predfun', predfun)"],
                 "Cross-validation. fun(XTRAIN, ..., XTEST, ...) is called once per fold (10 folds by default) "
                 "and its results are stacked in rows. The 'mse' and 'mcr' forms return the cross-validated "
                 "mean squared error or misclassification rate of predfun(XTRAIN, ytrain, XTEST). Options: "
                 "'KFold', 'Holdout', 'Leaveout', 'Partition', 'Stratify' and 'MCReps'."),
}
