"""Help text for distances, nearest neighbours and cluster analysis."""

_DIST = ("'euclidean', 'squaredeuclidean', 'seuclidean', 'mahalanobis', 'cityblock', 'minkowski', 'chebychev', "
         "'cosine', 'correlation', 'spearman', 'hamming', 'jaccard' or a function handle")

HELP = {
    "pdist": (["D = pdist(X)", "D = pdist(X, distance)", "D = pdist(X, distance, param)"],
              f"Pairwise distances between the rows of X as a row vector (use squareform for the matrix). "
              f"distance is {_DIST}."),
    "pdist2": (["D = pdist2(X, Y)", "D = pdist2(X, Y, distance)", "[D, I] = pdist2(X, Y, distance, 'Smallest', k)"],
               f"Distances between every row of X and every row of Y. distance is {_DIST}. 'Smallest' or "
               "'Largest' keeps the k extreme distances in each column with their row numbers."),
    "knnsearch": ("[idx, D] = knnsearch(X, Y, 'K', k, 'Distance', distance, ...)",
                  "For each row of Y, the k nearest rows of X (default 1) and their distances. With "
                  "'IncludeTies' the results are cell arrays that also hold points tied with the k-th."),
    "rangesearch": ("[idx, D] = rangesearch(X, Y, r, 'Distance', distance, ...)",
                    "For each row of Y, the rows of X within distance r (as cell arrays, nearest first)."),
    "KDTreeSearcher": ("ns = KDTreeSearcher(X, 'Distance', distance)",
                       "Nearest-neighbour searcher for the points X; use knnsearch(ns, Y, ...) and "
                       "rangesearch(ns, Y, r)."),
    "ExhaustiveSearcher": ("ns = ExhaustiveSearcher(X, 'Distance', distance)",
                           "Nearest-neighbour searcher that works with every distance; use knnsearch(ns, Y, ...) "
                           "and rangesearch(ns, Y, r)."),
    "createns": ("ns = createns(X, 'NSMethod', method, 'Distance', distance)",
                 "Create a KDTreeSearcher or an ExhaustiveSearcher for the points X."),
    "linkage": (["Z = linkage(X)", "Z = linkage(X, method)", "Z = linkage(X, method, metric)",
            "Z = linkage(y, method)"],
                "Agglomerative hierarchical cluster tree from data X or from a distance vector y. method is "
                "'single' (default), 'complete', 'average', 'weighted', 'centroid', 'median' or 'ward'. Each "
                "row of Z joins two clusters (new clusters are numbered after the observations) at the "
                "height in column 3."),
    "cluster": (["T = cluster(Z, 'MaxClust', n)", "T = cluster(Z, 'Cutoff', c)",
                 "T = cluster(Z, 'Cutoff', c, 'Criterion', 'distance')"],
                "Cut a linkage tree into clusters: at most n clusters, or by a threshold on the inconsistency "
                "coefficient (default) or on the linkage distance. Clusters are numbered in order of first "
                "appearance."),
    "clusterdata": (["T = clusterdata(X, cutoff)", "T = clusterdata(X, 'MaxClust', n, 'Linkage', method, ...)"],
                    "Hierarchical clustering of data in one step (pdist, linkage and cluster)."),
    "inconsistent": ("Y = inconsistent(Z, depth)",
                     "Inconsistency of each link of a tree: mean and standard deviation of the heights of the "
                     "links up to depth levels below (default 2), their number, and the coefficient."),
    "cophenet": ("[c, d] = cophenet(Z, Y)",
                 "Cophenetic correlation between the tree Z and the distances Y it was built from; d holds the "
                 "cophenetic distances."),
    "optimalleaforder": ("order = optimalleaforder(tree, D)",
                         "Leaf order of a tree that maximises the similarity of neighbouring leaves."),
    "dendrogram": (["dendrogram(Z)", "dendrogram(Z, P)", "[H, T, outperm] = dendrogram(Z, 'Name', value, ...)"],
                   "Draw a linkage tree. At most P leaves are shown (default 30; 0 shows all). Options: "
                   "'Orientation', 'Labels', 'ColorThreshold'."),
    "kmedoids": ("[idx, C, sumd, D, midx, info] = kmedoids(X, k, 'Name', value, ...)",
                 "k-medoids clustering: like k-means but each centre is one of the data points (rows midx). "
                 "Options: 'Distance', 'Start', 'Replicates', 'Algorithm'."),
    "dbscan": ("[idx, corepts] = dbscan(X, epsilon, minpts, 'Distance', distance)",
               "Density-based clustering: points with at least minpts neighbours within epsilon are core "
               "points; clusters are numbered from 1 and noise points get -1. A distance matrix can be given "
               "with 'Distance', 'precomputed'."),
    "spectralcluster": ("[idx, V, D] = spectralcluster(X, k, 'Name', value, ...)",
                        "Spectral clustering on a nearest-neighbour (or 'epsilon') similarity graph: k-means on "
                        "the first k eigenvectors of the graph Laplacian."),
    "fitgmdist": ("gm = fitgmdist(X, k, 'Name', value, ...)",
                  "Fit a Gaussian mixture with k components by expectation-maximisation. Options: "
                  "'CovarianceType', 'SharedCovariance', 'RegularizationValue', 'Replicates', 'Start', "
                  "'Options'. The model has mu, Sigma, ComponentProportion, NumComponents, AIC, BIC, "
                  "NegativeLogLikelihood, Converged and the methods cluster, posterior, pdf, cdf, random and "
                  "mahal."),
    "gmdistribution": ("gm = gmdistribution(mu, sigma, p)",
                       "Gaussian mixture distribution with the component means in the rows of mu, covariances "
                       "sigma and mixing proportions p."),
    "gmdistribution.fit": ("gm = gmdistribution.fit(X, k, ...)", "The older name of fitgmdist."),
    "silhouette": (["s = silhouette(X, clust)", "s = silhouette(X, clust, distance)", "[s, h] = silhouette(...)"],
                   "Silhouette value of each point: how much closer it is to its own cluster than to the "
                   "nearest other one (between -1 and 1). The default distance is 'sqEuclidean'. Asking for "
                   "the second output (or none) draws the silhouette plot."),
    "evalclusters": ("eva = evalclusters(X, clust, criterion, 'KList', ks)",
                     "Choose the number of clusters. clust is 'kmeans', 'linkage', 'gmdistribution', a function "
                     "handle or a matrix of solutions; criterion is 'CalinskiHarabasz', 'DaviesBouldin', "
                     "'silhouette' or 'gap'. The result has OptimalK, OptimalY, CriterionValues and InspectedK."),
}
