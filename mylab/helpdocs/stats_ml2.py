"""Help text for support vector machines, linear learners, ECOC, ensembles, Gaussian processes and networks."""

from .stats_ml import _COMMON, _REG

HELP = {
    "fitcsvm": ("mdl = fitcsvm(X, Y, 'Name', value, ...)",
                "Two-class support vector machine. Options: 'KernelFunction' ('linear', 'gaussian'/'rbf', "
                "'polynomial'), 'BoxConstraint' (1), 'KernelScale' (1 or 'auto'), 'Standardize', "
                "'PolynomialOrder'. Properties: Alpha, Bias, Beta (linear kernel), SupportVectors, "
                "IsSupportVector. The second column of the score is the signed distance from the boundary." + _COMMON),
    "fitrsvm": ("mdl = fitrsvm(X, Y, 'Name', value, ...)",
                "Support vector regression with an epsilon-insensitive loss. Options: 'KernelFunction', "
                "'BoxConstraint', 'Epsilon' (default iqr(Y)/13.49), 'KernelScale', 'Standardize'." + _REG),
    "fitclinear": ("mdl = fitclinear(X, Y, 'Name', value, ...)",
                   "Two-class linear classifier: 'Learner' 'svm' (default) or 'logistic', 'Regularization' "
                   "'ridge' or 'lasso', 'Lambda' (default 1/n). Properties: Beta, Bias, Lambda." + _COMMON),
    "fitrlinear": ("mdl = fitrlinear(X, Y, 'Name', value, ...)",
                   "Linear regression for wide data: 'Learner' 'svm' (default) or 'leastsquares', "
                   "'Regularization' 'ridge' or 'lasso', 'Lambda' (default 1/n), 'Epsilon'." + _REG),
    "fitcecoc": ("mdl = fitcecoc(X, Y, 'Learners', learners, 'Coding', coding, ...)",
                 "Multiclass model built from binary learners (error-correcting output codes). 'Coding' is "
                 "'onevsone' (default), 'onevsall', 'ordinal', 'binarycomplete', 'ternarycomplete' or a "
                 "matrix; 'Learners' is 'svm' (default), 'linear', 'knn', 'tree', 'discriminant', "
                 "'naivebayes' or a template. predict returns the labels, the negated average binary losses "
                 "and the binary learner scores." + _COMMON),
    "fitcensemble": ("mdl = fitcensemble(X, Y, 'Method', method, 'NumLearningCycles', n, ...)",
                     "Ensemble of classification trees: 'Bag' (random forest), 'AdaBoostM1', 'AdaBoostM2', "
                     "'LogitBoost' or 'GentleBoost' (default LogitBoost for two classes, AdaBoostM2 "
                     "otherwise; 100 cycles). Options: 'Learners' (templateTree), 'LearnRate'. Extra methods: "
                     "oobLoss and oobPredict (bagging), predictorImportance." + _COMMON),
    "fitrensemble": ("mdl = fitrensemble(X, Y, 'Method', method, 'NumLearningCycles', n, ...)",
                     "Ensemble of regression trees: 'LSBoost' (default) or 'Bag'." + _REG),
    "fitensemble": ("mdl = fitensemble(X, Y, method, nlearn, learners, ...)",
                    "The older form of fitcensemble and fitrensemble."),
    "TreeBagger": ("B = TreeBagger(NumTrees, X, Y, 'Method', 'classification', 'Name', value, ...)",
                   "Bagged decision trees (random forest). Options: 'Method' ('classification' or "
                   "'regression'), 'OOBPrediction', 'MinLeafSize', 'NumPredictorsToSample', 'InBagFraction'. "
                   "predict returns class labels as a cell array of text (with scores) or regression values; "
                   "oobError and oobPredict use the out-of-bag observations."),
    "fitrgp": ("mdl = fitrgp(X, Y, 'KernelFunction', kernel, ...)",
               "Gaussian process regression. The kernel ('squaredexponential' by default, 'exponential', "
               "'matern32', 'matern52', 'rationalquadratic' and their 'ard' forms), the noise 'Sigma' and the "
               "'BasisFunction' coefficients are fitted by maximising the marginal likelihood. "
               "[ypred, ysd, yint] = predict(mdl, Xnew) gives predictions with standard deviations and "
               "intervals." + _REG),
    "fitcnet": ("mdl = fitcnet(X, Y, 'LayerSizes', sizes, ...)",
                "Small fully connected neural network classifier (softmax output). Options: 'LayerSizes' (10), "
                "'Activations' ('relu', 'tanh', 'sigmoid', 'none'), 'Lambda', 'IterationLimit', "
                "'Standardize'." + _COMMON),
    "fitrnet": ("mdl = fitrnet(X, Y, 'LayerSizes', sizes, ...)",
                "Small fully connected neural network for regression. Options as for fitcnet." + _REG),
}
