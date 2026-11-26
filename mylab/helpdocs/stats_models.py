"""Help text for linear and generalized linear models."""

HELP = {
    "fitlm": (["mdl = fitlm(X, y)", "mdl = fitlm(X, y, modelspec)", "mdl = fitlm(tbl, formula, 'Name', value, ...)"],
              "Fit a linear regression model. modelspec is 'constant', 'linear' (default), 'interactions', "
              "'purequadratic', 'quadratic', 'polyIJK', a terms matrix or a formula such as "
              "'y ~ x1 + x2*x3' (a*b means a + b + a:b, x^2 includes x, -1 removes the intercept). Text and "
              "categorical table variables become indicator variables. The model has Coefficients (a table "
              "with Estimate, SE, tStat, pValue), Rsquared, RMSE, MSE, Residuals, Fitted and more, and the "
              "methods predict, feval, random, anova, coefCI, coefTest, plot, plotResiduals, addTerms and "
              "removeTerms. Options: 'CategoricalVars', 'Exclude', 'Intercept', 'PredictorVars', "
              "'ResponseVar', 'RobustOpts', 'VarNames', 'Weights'."),
    "stepwiselm": ("mdl = stepwiselm(X, y, modelspec, 'Name', value, ...)",
                   "Stepwise linear regression: starting from modelspec (default 'constant'), terms are added "
                   "while their F-test p-value is below 'PEnter' (0.05) and removed when it exceeds 'PRemove' "
                   "(0.10), within the bounds 'Lower' and 'Upper' (default 'interactions')."),
    "x2fx": ("D = x2fx(X, model)",
             "Design matrix for the predictors X: model is 'linear' (default, a constant column and X), "
             "'interaction', 'quadratic', 'purequadratic' or a terms matrix."),
    "fitglm": ("mdl = fitglm(X, y, modelspec, 'Distribution', distr, 'Name', value, ...)",
               "Fit a generalized linear model: 'Distribution' is 'normal', 'binomial', 'poisson', 'gamma' or "
               "'inverse gaussian', with its canonical 'Link' unless another is given ('identity', 'log', "
               "'logit', 'probit', 'comploglog', 'loglog', 'reciprocal' or a power). Other options: 'Offset', "
               "'Weights', 'BinomialSize', 'DispersionFlag', 'Intercept', 'CategoricalVars'. The model has "
               "Coefficients, Deviance, Dispersion, Fitted, Residuals, LogLikelihood and the methods predict, "
               "feval, random, coefCI, coefTest and devianceTest."),
    "glmfit": ("[b, dev, stats] = glmfit(X, y, distr, 'Name', value, ...)",
               "Coefficients of a generalized linear model (a constant term is added first). For 'binomial' y "
               "is a 0/1 vector or [successes trials]. Options: 'link', 'constant', 'offset', 'weights', "
               "'estdisp'. stats holds se, t, p, covb, dfe, s and the residuals."),
    "glmval": ("[yhat, dylo, dyhi] = glmval(b, X, link, stats, 'Name', value, ...)",
               "Predictions of a generalized linear model with coefficients b, and confidence bounds yhat - "
               "dylo and yhat + dyhi when stats from glmfit is given. Options: 'confidence', 'size', 'offset', "
               "'constant'."),
}
