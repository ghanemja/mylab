"""Help text for mylab/builtins/pointcloud.py (written from scratch for mylab)."""

HELP = {
    "pointCloud": (["ptCloud = pointCloud(xyzPoints)", "ptCloud = pointCloud(xyzPoints, 'Color', C, 'Normal', N, "
                    "'Intensity', v)"],
                   "A point cloud. xyzPoints is M-by-3, or M-by-N-by-3 for an organized cloud (one point per pixel "
                   "of a depth image). Properties: Location, Count, XLimits, YLimits, ZLimits, Color (uint8 RGB "
                   "per point, or one colour for all), Normal and Intensity. Methods: findNearestNeighbors(pc, "
                   "point, K) and findNeighborsInRadius(pc, point, r) return [indices, distances]; "
                   "findPointsInROI(pc, [xmin xmax ymin ymax zmin zmax]); select(pc, indices) (or row and column "
                   "indices for an organized cloud, with 'OutputSize', 'full' to keep the layout); "
                   "removeInvalidPoints(pc) drops points with NaN or Inf coordinates and also returns the indices "
                   "kept; copy(pc)."),
    "pcread": (["ptCloud = pcread(filename)"],
               "Reads a PLY file (ASCII or binary, either byte order) or a PCD file (ASCII or binary; "
               "binary_compressed is not supported). Vertex colours, normals and intensity are read when present; "
               "faces are ignored."),
    "pcwrite": (["pcwrite(ptCloud, filename)", "pcwrite(ptCloud, filename, 'Encoding', 'ascii'|'binary')"],
                "Writes a point cloud to a .ply or .pcd file with its colours, normals and intensity. Coordinates are "
                "stored as single precision."),
    "pcshow": (["pcshow(ptCloud)", "pcshow(xyzPoints)", "pcshow(xyzPoints, C)", "pcshow(..., 'MarkerSize', s)",
                "h = pcshow(...)"],
               "Plots the points with scatter3, coloured by the cloud's colours, by C (one RGB row per point or one "
               "colour), or by height, and sets equal axis scaling."),
    "pcshowpair": (["pcshowpair(ptCloudA, ptCloudB)"],
                   "Plots two clouds together, the first in magenta and the second in green, to compare them (for "
                   "example before and after registration)."),
    "pcdownsample": (["ptCloudOut = pcdownsample(ptCloud, 'random', percentage)",
                      "ptCloudOut = pcdownsample(ptCloud, 'gridAverage', gridStep)",
                      "ptCloudOut = pcdownsample(ptCloud, 'nonuniformGridSample', maxNumPoints)"],
                     "Reduces the number of points: a random fraction (using the global random generator), the "
                     "average of the points (and colours and normals) in each cube of side gridStep, or one point "
                     "from each cell of an adaptive split that leaves at most maxNumPoints points per cell."),
    "pcdenoise": (["[ptCloudOut, inlierIndices, outlierIndices] = pcdenoise(ptCloud)",
                   "[...] = pcdenoise(ptCloud, 'NumNeighbors', k, 'Threshold', t, 'PreserveStructure', tf)"],
                  "Removes isolated points: a point is an outlier when its mean distance to its k nearest neighbours "
                  "(default 4) is more than Threshold (default 1) standard deviations above the average of that "
                  "distance."),
    "pcnormals": (["normals = pcnormals(ptCloud)", "normals = pcnormals(ptCloud, K)"],
                  "Surface normals from a plane fitted to each point's K nearest neighbours (default 6), oriented "
                  "towards the origin (the sensor position for camera-centred clouds)."),
    "planeModel": (["model = planeModel(parameters)"],
                   "A plane a*x + b*y + c*z + d = 0 with properties Parameters [a b c d] (normalised so [a b c] has "
                   "unit length) and Normal. plot(model) draws it over the current x and y limits."),
    "pcfitplane": (["[model, inlierIndices, outlierIndices, meanError] = pcfitplane(ptCloud, maxDistance)",
                    "[...] = pcfitplane(ptCloud, maxDistance, referenceVector, maxAngularDistance)",
                    "[...] = pcfitplane(..., 'SampleIndices', idx, 'MaxNumTrials', n, 'Confidence', c)"],
                   "Fits a plane robustly (MSAC on random triples, then a least-squares refit). Points within "
                   "maxDistance are inliers. With a reference vector, only planes whose normal is within "
                   "maxAngularDistance degrees (default 5) of it are accepted. Uses the global random generator."),
    "pcregistericp": (["tform = pcregistericp(moving, fixed)", "[tform, movingReg, rmse] = pcregistericp(...)",
                       "[...] = pcregistericp(..., 'Metric', 'pointToPoint'|'pointToPlane', 'MaxIterations', n, "
                       "'Tolerance', [dT dR], 'InitialTransform', tform0, 'InlierRatio', r, 'InlierDistance', d)"],
                      "Rigidly aligns moving to fixed with the iterative closest point method and returns the "
                      "rigidtform3d that maps moving onto fixed, the transformed cloud and the RMS distance to the "
                      "closest fixed points. Stops after MaxIterations (default 20) or when a step moves less than "
                      "Tolerance (default [0.01 0.5], translation and degrees). InlierRatio or InlierDistance ignore "
                      "far-away pairs."),
    "pctransform": (["ptCloudOut = pctransform(ptCloud, tform)", "ptCloudOut = pctransform(ptCloud, D)"],
                    "Applies a 3-D transformation (rigidtform3d, affinetform3d, ...) to the points and normals, or "
                    "adds a displacement field D of the same size as Location."),
    "pcmerge": (["ptCloudOut = pcmerge(ptCloudA, ptCloudB, gridStep)"],
                "Joins two clouds and averages the points within each cube of side gridStep."),
    "pcsegdist": (["[labels, numClusters] = pcsegdist(ptCloud, minDistance)"],
                  "Splits a cloud into clusters: points closer than minDistance (directly or through a chain of "
                  "points) share a label. Invalid points get label 0."),
}
