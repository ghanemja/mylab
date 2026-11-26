"""Help text for the classification and regression learners and the model assessment tools."""

_COMMON = (" The data are (X, Y) or (tbl, ResponseVarName) / (tbl, 'y ~ x1 + x2'). Labels may be numeric, logical, "
           "text or categorical and are returned by predict in the same type. Shared options: 'ClassNames', "
           "'Prior', 'Cost', 'Weights', 'PredictorNames', 'ResponseName', 'ScoreTransform' and, to get a "
           "cross-validated model, 'CrossVal', 'KFold', 'Holdout', 'Leaveout' or 'CVPartition'. Methods: "
           "predict (labels, scores), resubPredict, loss, resubLoss, margin, edge and crossval (then "
           "kfoldLoss and kfoldPredict).")
_REG = (" The data are (X, Y) or (tbl, ResponseVarName). Methods: predict, resubPredict, loss, resubLoss and "
        "crossval (then kfoldLoss and kfoldPredict).")

HELP = {
    "fitcknn": ("mdl = fitcknn(X, Y, 'NumNeighbors', k, 'Name', value, ...)",
                "k-nearest-neighbour classifier (k = 1 by default). Options: 'Distance', 'DistanceWeight', "
                "'Standardize', 'BreakTies', 'IncludeTies', 'Exponent', 'Cov', 'Scale'. Scores are the "
                "class fractions among the neighbours." + _COMMON),
    "fitctree": ("tree = fitctree(X, Y, 'Name', value, ...)",
                 "Classification tree (CART). Options: 'MinParentSize' (10), 'MinLeafSize' (1), "
                 "'MaxNumSplits', 'SplitCriterion' ('gdi', 'deviance', 'twoing'), 'MergeLeaves', "
                 "'NumVariablesToSample'. Extra methods: view, prune and predictorImportance; properties such "
                 "as NumNodes, Children, CutPredictor, CutPoint, NodeClass and ClassProbability describe the "
                 "tree. Text and categorical predictors of a table are used through their category numbers." + _COMMON),
    "fitrtree": ("tree = fitrtree(X, Y, 'Name', value, ...)",
                 "Regression tree (CART, splits chosen to reduce the mean squared error). Options: "
                 "'MinParentSize', 'MinLeafSize', 'MaxNumSplits', 'MergeLeaves'. Extra methods: view, prune "
                 "and predictorImportance." + _REG),
    "fitcnb": ("mdl = fitcnb(X, Y, 'DistributionNames', names, ...)",
               "Naive Bayes classifier. Each predictor is modelled within each class by a 'normal' (default), "
               "'kernel', 'mvmn' (categorical) or 'mn' (multinomial counts) distribution; "
               "DistributionParameters holds the fitted values. A normal predictor that is constant within a "
               "class is an error. Scores are posterior probabilities." + _COMMON),
    "fitcdiscr": ("mdl = fitcdiscr(X, Y, 'DiscrimType', type, ...)",
                  "Discriminant analysis classifier: 'linear' (default), 'quadratic', 'diaglinear', "
                  "'diagquadratic', 'pseudolinear' or 'pseudoquadratic'; 'Gamma' regularises the covariance. "
                  "Properties: Mu, Sigma, Coeffs, Prior. Scores are posterior probabilities." + _COMMON),
    "classify": ("[class, err, posterior, logp, coeff] = classify(sample, training, group, type, prior)",
                 "Discriminant analysis in one call: classify the rows of sample using the training rows and "
                 "their groups. type is 'linear' (default), 'diaglinear', 'quadratic', 'diagquadratic' or "
                 "'mahalanobis'. err is the apparent error rate on the training data."),
    "confusionmat": (["C = confusionmat(group, grouphat)",
            "[C, order] = confusionmat(group, grouphat, 'Order', order)"],
                     "Confusion matrix: C(i, j) counts the observations of true class i predicted as class j."),
    "confusionchart": (["confusionchart(trueLabels, predictedLabels)", "confusionchart(m, classLabels)",
                        "cm = confusionchart(...)"],
                       "Draw a confusion matrix as a coloured grid with counts. 'Normalization' may be "
                       "'row-normalized', 'column-normalized' or 'total-normalized'."),
    "perfcurve": ("[X, Y, T, AUC, OPTROCPT] = perfcurve(labels, scores, posclass, 'Name', value, ...)",
                  "Receiver operating characteristic (or another performance curve) for scores of the positive "
                  "class. X and Y default to the false and true positive rates; T holds the thresholds, AUC "
                  "the area under the curve and OPTROCPT the optimal operating point. Options: 'XCrit', "
                  "'YCrit', 'NegClass', 'Cost', 'Weights', 'ProcessNaN'."),
    "sequentialfs": ("[inmodel, history] = sequentialfs(fun, X, y, 'Name', value, ...)",
                     "Sequential feature selection: columns of X are added (or removed with 'direction', "
                     "'backward') while the cross-validated criterion fun(XTRAIN, ytrain, XTEST, ytest) "
                     "improves. Options: 'cv', 'nfeatures', 'keepin', 'keepout'."),
    "relieff": ("[ranks, weights] = relieff(X, y, k)",
                "ReliefF (classification) or RReliefF (regression) predictor importance from k nearest "
                "neighbours; ranks lists the predictors from most to least important."),
    "templateTree": ("t = templateTree('Name', value, ...)",
                     "Tree learner template for fitcensemble, fitrensemble and fitcecoc."),
    "templateSVM": ("t = templateSVM('Name', value, ...)", "SVM learner template for fitcecoc."),
    "templateKNN": ("t = templateKNN('Name', value, ...)", "Nearest-neighbour learner template for fitcecoc."),
    "templateDiscriminant": ("t = templateDiscriminant('Name', value, ...)",
                             "Discriminant analysis learner template for fitcecoc."),
    "templateNaiveBayes": ("t = templateNaiveBayes('Name', value, ...)", "Naive Bayes learner template for fitcecoc."),
    "templateLinear": ("t = templateLinear('Name', value, ...)", "Linear learner template for fitcecoc."),
}
