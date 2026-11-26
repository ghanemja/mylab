"""Help text for the multivariate analysis functions."""

HELP = {
    "pca": ("[coeff, score, latent, tsquared, explained, mu] = pca(X, 'Name', value, ...)",
            "Principal component analysis of the rows of X. coeff holds the component directions (the largest "
            "entry of each column is positive), score the data in those coordinates, latent the component "
            "variances, tsquared Hotelling's statistic of each row, explained the percentage of variance and "
            "mu the column means. Options: 'Algorithm' ('svd' or 'eig'), 'Centered', 'Economy', "
            "'NumComponents', 'Rows' ('complete', 'pairwise', 'all'), 'Weights', 'VariableWeights'."),
    "pcacov": ("[coeff, latent, explained] = pcacov(V)",
               "Principal components from a covariance (or correlation) matrix."),
    "pcares": ("[residuals, reconstructed] = pcares(X, ndim)",
               "Residuals of the data after keeping ndim principal components, and the reconstruction."),
    "factoran": ("[lambda, psi, T, stats, F] = factoran(X, m, 'Name', value, ...)",
                 "Maximum-likelihood factor analysis with m common factors on standardised variables: loadings "
                 "lambda, specific variances psi, rotation matrix T, fit statistics and factor scores F. "
                 "Options: 'rotate' ('varimax', 'quartimax', 'none'), 'scores', 'start', 'xtype', 'nobs'."),
    "canoncorr": ("[A, B, r, U, V, stats] = canoncorr(X, Y)",
                  "Canonical correlation analysis: coefficients A and B, canonical correlations r, canonical "
                  "variables U and V, and tests of the number of non-zero correlations."),
    "cmdscale": ("[Y, e] = cmdscale(D, p)",
                 "Classical multidimensional scaling: a configuration of points whose distances reproduce the "
                 "dissimilarities D (a square matrix or a pdist vector). e holds the eigenvalues."),
    "mdscale": ("[Y, stress, disparities] = mdscale(D, p, 'Name', value, ...)",
                "Multidimensional scaling in p dimensions. 'Criterion' is 'stress' (non-metric, default), "
                "'sstress', 'metricstress', 'metricsstress', 'sammon' or 'strain'. Other options: 'Start', "
                "'Replicates', 'Weights'."),
    "procrustes": ("[d, Z, transform] = procrustes(X, Y, 'scaling', tf, 'reflection', tf)",
                   "Fit Y to X by translation, rotation and scaling. d is the standardised sum of squared "
                   "errors, Z the transformed Y and transform holds T, b and c with Z = b*Y*T + c."),
    "mahal": ("d2 = mahal(Y, X)",
              "Squared Mahalanobis distance of each row of Y from the sample X."),
    "tsne": ("[Y, loss] = tsne(X, 'Name', value, ...)",
             "t-distributed stochastic neighbour embedding of the rows of X (exact gradient). Options: "
             "'NumDimensions', 'Perplexity', 'Exaggeration', 'LearnRate', 'Distance', 'Standardize', "
             "'NumPCAComponents', 'InitialY', 'Options' (MaxIter)."),
}
