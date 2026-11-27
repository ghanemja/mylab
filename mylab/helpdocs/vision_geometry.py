"""Help text for the geometry functions in mylab/builtins/vision_geometry.py (written from scratch for mylab)."""

_T3 = ("Methods: transformPointsForward(tform, P) and transformPointsInverse(tform, P) map M-by-3 points (or "
       "separate x, y, z arrays), and invert(tform) gives the inverse transformation.")
_ROBUST = ("Options: 'MaxNumTrials' (default 1000), 'Confidence' in percent (default 99) and 'MaxDistance', the "
           "largest distance in pixels between a transformed point and its match for it to count as an inlier "
           "(default 1.5). The fit is an MSAC search over random minimal samples (using the global random "
           "generator, so rng makes it repeatable) followed by a least-squares refit on the inliers.")

HELP = {
    "rigidtform3d": (["tform = rigidtform3d", "tform = rigidtform3d(R, t)", "tform = rigidtform3d(eulerAngles, t)",
                      "tform = rigidtform3d(A)"],
                     "A 3-D rotation followed by a translation, mapping column vectors: [x; y; z] = R*[u; v; w] + t. "
                     "eulerAngles are [rx ry rz] in degrees, applied about x, then y, then z. Properties: R, "
                     "Translation, A (4-by-4), Dimensionality. " + _T3),
    "simtform3d": (["tform = simtform3d(scale, R, t)", "tform = simtform3d(A)"],
                   "A 3-D similarity: uniform scale, rotation and translation. Properties: Scale, R, Translation, "
                   "A. " + _T3),
    "affinetform3d": (["tform = affinetform3d(A)"],
                      "A general 3-D affine transformation given by a 4-by-4 matrix A whose last row is [0 0 0 1]. "
                      + _T3),
    "transltform3d": (["tform = transltform3d(t)", "tform = transltform3d(tx, ty, tz)"],
                      "A pure 3-D translation. " + _T3),
    "rigid3d": (["tform = rigid3d(rotation, translation)", "tform = rigid3d(T)"],
                "The older 3-D rigid transformation in the row-vector convention: [x y z] = [u v w]*Rotation + "
                "Translation, with T the 4-by-4 matrix whose last column is [0 0 0 1]'. " + _T3),
    "affine3d": (["tform = affine3d(T)"],
                 "The older 3-D affine transformation in the row-vector convention (T is 4-by-4 with last column "
                 "[0 0 0 1]'). " + _T3),
    "rotvec2mat3d": (["R = rotvec2mat3d(v)"],
                     "Rotation matrix of a rotation vector (axis times angle in radians). An N-by-3 input gives a "
                     "3-by-3-by-N array."),
    "rotmat2vec3d": (["v = rotmat2vec3d(R)"],
                     "Rotation vector (axis times angle in radians, one row per matrix) of a 3-by-3 rotation matrix "
                     "or 3-by-3-by-N array."),
    "rotationVectorToMatrix": (["R = rotationVectorToMatrix(v)"],
                               "The older conversion: the transpose of rotvec2mat3d(v), as used by the row-vector "
                               "convention of the older camera functions."),
    "rotationMatrixToVector": (["v = rotationMatrixToVector(R)"],
                               "The older conversion; the inverse of rotationVectorToMatrix."),
    "estgeotform2d": (["tform = estgeotform2d(matchedPoints1, matchedPoints2, transformType)",
                       "[tform, inlierIndex] = estgeotform2d(...)", "[tform, inlierIndex, status] = estgeotform2d(...)",
                       "[...] = estgeotform2d(..., 'MaxNumTrials', n, 'Confidence', c, 'MaxDistance', d)"],
                      "Robustly fits a transformation that maps matchedPoints1 onto matchedPoints2 (M-by-2 [x y] or "
                      "points objects), ignoring outliers. transformType is 'rigid' (rigidtform2d), 'similarity' "
                      "(simtform2d), 'affine' (affinetform2d) or 'projective' (projtform2d); these work with imwarp. "
                      "inlierIndex is an M-by-1 logical vector. status is 0 on success, 1 when there are too few "
                      "points (2, 2, 3 and 4 are needed) and 2 when too few inliers were found; without the status "
                      "output these cases raise an error. " + _ROBUST),
    "estimateGeometricTransform2D": (["[tform, inlierIndex, status] = estimateGeometricTransform2D(matchedPoints1, "
                                      "matchedPoints2, transformType, ...)"],
                                     "Like estgeotform2d, but returns the older affine2d or projective2d objects. "
                                     "'rigid' and 'similarity' results are affine2d. " + _ROBUST),
    "estimateGeometricTransform": (["[tform, inlierPoints1, inlierPoints2, status] = estimateGeometricTransform("
                                    "matchedPoints1, matchedPoints2, transformType, ...)"],
                                   "The oldest form: transformType is 'similarity', 'affine' or 'projective'; returns "
                                   "affine2d or projective2d and the inlier points themselves. " + _ROBUST),
    "estimateFundamentalMatrix": (["F = estimateFundamentalMatrix(matchedPoints1, matchedPoints2)",
                                   "[F, inliersIndex, status] = estimateFundamentalMatrix(..., 'Method', m, "
                                   "'DistanceType', d, 'DistanceThreshold', t, 'NumTrials', n, 'Confidence', c, "
                                   "'InlierPercentage', p, 'ReportRuntimeError', tf)"],
                                  "Fits the 3-by-3 fundamental matrix F with [x2 y2 1]*F*[x1 y1 1]' = 0 for "
                                  "corresponding points, rank 2, scaled to unit norm. Method is 'RANSAC' (default), "
                                  "'MSAC', 'LMedS', 'LTS' or 'Norm8Point' (all points, no outlier rejection). "
                                  "DistanceType is 'Sampson' (default) or 'Algebraic'; DistanceThreshold (default "
                                  "0.01) is in those units. At least 8 point pairs are needed. status is 0 (success), "
                                  "1 (too few points) or 2 (too few inliers); with 'ReportRuntimeError', false these "
                                  "are reported in status instead of raising an error."),
    "estimateEssentialMatrix": (["[E, inliersIndex, status] = estimateEssentialMatrix(matchedPoints1, "
                                 "matchedPoints2, intrinsics)",
                                 "[...] = estimateEssentialMatrix(p1, p2, intrinsics1, intrinsics2, 'MaxNumTrials', "
                                 "n, 'Confidence', c, 'MaxDistance', d)"],
                                "Fits the essential matrix of two calibrated views with OpenCV's five-point RANSAC "
                                "solver, after removing lens distortion. intrinsics are cameraIntrinsics or "
                                "cameraParameters. MaxDistance (default 0.1) is a squared distance in pixels. "
                                "Needs OpenCV."),
    "estrelpose": (["[relativePose, validPointsFraction] = estrelpose(E, intrinsics, inlierPoints1, inlierPoints2)",
                    "[...] = estrelpose(F, intrinsics1, intrinsics2, inlierPoints1, inlierPoints2)"],
                   "Recovers the pose of camera 2 relative to camera 1 from an essential (or fundamental) matrix, as "
                   "a rigidtform3d that maps camera-2 coordinates into camera-1 coordinates. The translation has unit "
                   "length (the scale cannot be recovered from two views). Of the four decompositions, the one that "
                   "puts the most triangulated points in front of both cameras wins; validPointsFraction is that "
                   "fraction."),
    "relativeCameraPose": (["[orientation, location, validPointsFraction] = relativeCameraPose(E, intrinsics, "
                            "inlierPoints1, inlierPoints2)", "relativePose = relativeCameraPose(...)"],
                           "The older form of estrelpose: orientation is the transposed rotation (row-vector "
                           "convention) and location is the 1-by-3 position of camera 2 in camera-1 coordinates. "
                           "With a single output it returns the rigidtform3d of estrelpose."),
    "triangulate": (["worldPoints = triangulate(matchedPoints1, matchedPoints2, cameraMatrix1, cameraMatrix2)",
                     "[worldPoints, reprojectionErrors, validIndex] = triangulate(...)",
                     "[...] = triangulate(matchedPoints1, matchedPoints2, stereoParams)"],
                    "Finds the 3-D points seen at matching image points by linear (DLT) triangulation. Camera "
                    "matrices are 3-by-4 projection matrices (cameraProjection) or the older 4-by-3 form "
                    "(cameraMatrix). reprojectionErrors is the mean pixel error per point and validIndex marks "
                    "points in front of both cameras. With stereoParameters the points are undistorted first and "
                    "the result is in camera-1 coordinates."),
    "epipolarLine": (["lines = epipolarLine(F, points)"],
                     "Epipolar lines in image 2 of points in image 1: each row [a b c] describes a*x + b*y + c = 0. "
                     "Use epipolarLine(F', points) for lines in image 1."),
    "isEpipoleInImage": (["isIn = isEpipoleInImage(F, imageSize)", "[isIn, epipole] = isEpipoleInImage(F, imageSize)"],
                         "Whether the epipole of image 1 (the null vector of F) lies inside an image of the given "
                         "[rows cols] size, and its [x y] location."),
}
