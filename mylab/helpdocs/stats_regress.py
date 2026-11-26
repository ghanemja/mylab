"""Help text for robust, nonlinear, penalised, partial least squares and multinomial regression."""

HELP = {
    "robustfit": ("[b, stats] = robustfit(X, y, wfun, tune, const)",
                  "Robust linear regression by iteratively reweighted least squares (a constant term is added "
                  "unless const is 'off'). wfun is 'bisquare' (default), 'andrews', 'cauchy', 'fair', 'huber', "
                  "'logistic', 'ols', 'talwar' or 'welsch'. stats has the scale estimates, se, t, p, the "
                  "final weights w and the residuals."),
    "nlinfit": ("[beta, R, J, CovB, MSE, ErrorModelInfo] = nlinfit(X, Y, modelfun, beta0, ...)",
                "Nonlinear least squares: fit Y = modelfun(beta, X) starting from beta0 (Levenberg-Marquardt). "
                "R holds the residuals, J the Jacobian, CovB the coefficient covariance and MSE the error "
                "variance estimate. 'Weights' gives observation weights."),
    "nlparci": (["ci = nlparci(beta, resid, 'covar', sigma)", "ci = nlparci(beta, resid, 'jacobian', J)"],
                "Confidence intervals for the coefficients of a nonlinear fit, from the outputs of nlinfit. "
                "'alpha' sets the level (default 0.05)."),
    "nlpredci": ("[ypred, delta] = nlpredci(modelfun, X, beta, resid, 'Covar', sigma, ...)",
                 "Predictions of a nonlinear model with confidence half-widths delta. 'PredOpt' is 'curve' "
                 "(default) or 'observation'; 'SimOpt' 'on' gives simultaneous bounds; the 'Jacobian' can be "
                 "given instead of 'Covar'."),
    "fitnlm": (["mdl = fitnlm(X, y, modelfun, beta0)", "mdl = fitnlm(tbl, 'y ~ b1*exp(b2*x)', beta0)"],
               "Fit a nonlinear regression model. The model is a function handle modelfun(b, X) or a formula "
               "whose unknown names are the coefficients. The result has Coefficients, RMSE, Rsquared, Fitted, "
               "Residuals and the methods predict (with intervals), feval, coefCI and coefTest."),
    "lasso": ("[B, FitInfo] = lasso(X, y, 'Name', value, ...)",
              "Lasso and elastic-net regression. Each column of B holds the coefficients for one value of "
              "FitInfo.Lambda (ascending); FitInfo.Intercept has the constant terms. Options: 'Lambda', "
              "'Alpha' (1 = lasso), 'NumLambda', 'LambdaRatio', 'Standardize', 'Intercept', 'CV', 'DFmax', "
              "'RelTol', 'Weights', 'PredictorNames'. With 'CV' the mean squared error is cross-validated and "
              "LambdaMinMSE and Lambda1SE are reported."),
    "lassoglm": ("[B, FitInfo] = lassoglm(X, y, distr, 'Name', value, ...)",
                 "Lasso or elastic-net regularised generalized linear model ('normal', 'binomial', 'poisson', "
                 "'gamma', 'inverse gaussian'). FitInfo has Intercept, Lambda, DF and Deviance."),
    "ridge": (["b = ridge(y, X, k)", "b = ridge(y, X, k, scaled)"],
              "Ridge regression for each ridge parameter in k. With scaled = 1 (default) the coefficients "
              "refer to centred predictors scaled to unit standard deviation; with scaled = 0 they are on the "
              "original scale with the constant term first."),
    "stepwisefit": ("[b, se, pval, inmodel, stats, nextstep, history] = stepwisefit(X, y, ...)",
                    "Stepwise regression on the columns of X: columns enter when their p-value is below "
                    "'PEnter' (0.05) and leave when it exceeds 'PRemove' (0.10). inmodel flags the chosen "
                    "columns."),
    "polyconf": ("[y, delta] = polyconf(p, x, S, 'Name', value, ...)",
                 "Evaluate a polynomial and, with the structure S from polyfit, the half-width delta of the "
                 "prediction interval. Options: 'alpha', 'mu', 'predopt' ('observation' or 'curve') and "
                 "'simopt'."),
    "plsregress": ("[XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, Y, ncomp, ...)",
                   "Partial least squares regression with ncomp components (SIMPLS). BETA holds the "
                   "regression coefficients with the constant first; PCTVAR the fraction of variance explained "
                   "in X and Y; MSE the mean squared errors for 0 to ncomp components ('CV' cross-validates)."),
    "mnrfit": ("[B, dev, stats] = mnrfit(X, Y, 'model', type, ...)",
               "Multinomial logistic regression. Y holds category numbers, a categorical array or a matrix of "
               "counts. 'model' is 'nominal' (default, last category as reference), 'ordinal' (proportional "
               "odds) or 'hierarchical'."),
    "mnrval": ("pihat = mnrval(B, X, 'model', type, 'type', kind)",
               "Category probabilities of a multinomial model fitted by mnrfit. 'type' is 'category' "
               "(default), 'cumulative' or 'conditional'."),
}
