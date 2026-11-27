"""Help text for the camera functions in mylab/builtins/vision_calib.py (written from scratch for mylab)."""

_K = ("K is [fx s cx; 0 fy cy; 0 0 1] in MATLAB pixel coordinates (the centre of the top-left pixel is (1, 1)).")

HELP = {
    "cameraIntrinsics": (["intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize)",
                          "intrinsics = cameraIntrinsics(..., 'RadialDistortion', k, 'TangentialDistortion', p, "
                          "'Skew', s)"],
                         "Describes a pinhole camera: FocalLength [fx fy] in pixels, PrincipalPoint [cx cy], "
                         "ImageSize [rows cols], RadialDistortion [k1 k2] or [k1 k2 k3], TangentialDistortion "
                         "[p1 p2], Skew and K. " + _K),
    "cameraParameters": (["params = cameraParameters", "params = cameraParameters('K', K, 'RadialDistortion', k, "
                          "'TangentialDistortion', p, 'ImageSize', sz, 'WorldUnits', u, ...)"],
                         "Holds a calibrated camera: the intrinsics (Intrinsics, K, IntrinsicMatrix = K', "
                         "FocalLength, PrincipalPoint, Skew, RadialDistortion, TangentialDistortion, ImageSize), the "
                         "pose of each calibration pattern (RotationVectors, TranslationVectors, RotationMatrices "
                         "in the row-vector convention, PatternExtrinsics as a cell array of rigidtform3d), and the "
                         "fit quality (ReprojectionErrors, ReprojectedPoints, MeanReprojectionError). "
                         "estimateCameraParameters returns one; you can also build one from a known K."),
    "generateCheckerboardPoints": (["worldPoints = generateCheckerboardPoints(boardSize, squareSize)"],
                                   "World [x y] coordinates of the inner corners of a checkerboard with boardSize "
                                   "[rows cols] squares, in the order detectCheckerboardPoints returns them: the "
                                   "first corner is at (0, 0) and points run down each column (y grows fastest)."),
    "detectCheckerboardPoints": (["[imagePoints, boardSize] = detectCheckerboardPoints(I)",
                                  "[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(images)",
                                  "[imagePoints, boardSize, pairsUsed] = detectCheckerboardPoints(images1, images2)"],
                                 "Finds the inner corners of a checkerboard (OpenCV's sector-based detector, which "
                                 "also finds the board size). images is an M-by-N-by-1-by-P (or -by-3-by-P) array, a "
                                 "cell array of images, or file names. imagePoints is M-by-2 for one image, "
                                 "M-by-2-by-P for several, and M-by-2-by-P-by-2 for stereo pairs; boardSize counts "
                                 "squares [rows cols]. Corners are ordered to match generateCheckerboardPoints, with "
                                 "the first corner nearest the top-left of the image; boards seen upside down are "
                                 "therefore numbered differently. Images where no board (or a board of another size) "
                                 "is found are left out and marked false in imagesUsed. Needs OpenCV."),
    "estimateCameraParameters": (["[params, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, "
                                  "worldPoints)",
                                  "[...] = estimateCameraParameters(..., 'ImageSize', [rows cols], "
                                  "'NumRadialDistortionCoefficients', 2, 'EstimateTangentialDistortion', false, "
                                  "'WorldUnits', 'mm')",
                                  "stereoParams = estimateCameraParameters(stereoImagePoints, worldPoints, ...)"],
                                 "Calibrates a camera from views of a planar pattern (Zhang's method with OpenCV's "
                                 "calibrateCamera). imagePoints is M-by-2-by-P (NaN rows for corners that were not "
                                 "seen); worldPoints is M-by-2 on the pattern plane. Gives ImageSize when you can "
                                 "(it seeds the principal point). An M-by-2-by-P-by-2 input calibrates a stereo pair "
                                 "and returns stereoParameters. estimationErrors is a struct of standard errors of "
                                 "the intrinsics. Skew is not estimated (EstimateSkew is accepted and ignored). "
                                 "Needs OpenCV."),
    "stereoParameters": (["stereoParams = stereoParameters(cameraParams1, cameraParams2, rotationOfCamera2, "
                          "translationOfCamera2)"],
                         "Describes a calibrated stereo pair. rotationOfCamera2 and translationOfCamera2 follow the "
                         "row-vector convention: a point X1 (row) in camera-1 coordinates is X1*R + t in camera-2 "
                         "coordinates. Properties also include PoseCamera2 (rigidtform3d of camera 2 in camera-1 "
                         "coordinates), FundamentalMatrix, EssentialMatrix and MeanReprojectionError."),
    "rectifyStereoImages": (["[J1, J2, reprojectionMatrix] = rectifyStereoImages(I1, I2, stereoParams)",
                             "[...] = rectifyStereoImages(..., 'OutputView', 'valid'|'full')"],
                            "Undistorts and rectifies a stereo pair so that matching points lie on the same row. "
                            "'valid' (default) crops to the area both images cover; 'full' keeps everything. "
                            "reprojectionMatrix Q maps [x y disparity 1]*Q to homogeneous 3-D camera-1 "
                            "coordinates (see reconstructScene). Needs OpenCV."),
    "disparitySGM": (["D = disparitySGM(I1, I2)", "D = disparitySGM(I1, I2, 'DisparityRange', [lo hi], "
                      "'UniquenessThreshold', u)"],
                     "Disparity map of a rectified grayscale pair by semi-global matching (OpenCV's SGBM). D is "
                     "single, x1 - x2 in pixels, with NaN where no reliable match was found. DisparityRange "
                     "defaults to [0 128]; hi - lo must be a multiple of 8. Needs OpenCV."),
    "disparityBM": (["D = disparityBM(I1, I2)", "D = disparityBM(I1, I2, 'DisparityRange', [lo hi], 'BlockSize', n, "
                     "'UniquenessThreshold', u, 'TextureThreshold', t, 'DistanceThreshold', d)"],
                    "Disparity map by block matching (OpenCV's StereoBM): odd BlockSize (default 15), DisparityRange "
                    "default [0 64]. Unreliable pixels are NaN. Needs OpenCV."),
    "reconstructScene": (["xyzPoints = reconstructScene(disparityMap, reprojectionMatrix)",
                          "xyzPoints = reconstructScene(disparityMap, stereoParams)"],
                         "3-D coordinates (M-by-N-by-3, in camera-1 world units) of every pixel of a disparity map; "
                         "NaN disparities give NaN points."),
    "undistortImage": (["J = undistortImage(I, intrinsics)", "[J, newOrigin] = undistortImage(I, intrinsics, "
                        "interp, 'OutputView', view, 'FillValues', v)"],
                       "Removes lens distortion. view is 'same' (default: same size as I), 'full' (all of the "
                       "undistorted image) or 'valid' (only pixels with data); newOrigin is the [x y] of J's "
                       "top-left pixel in the undistorted coordinates. interp is 'linear' (default), 'nearest' or "
                       "'cubic'. Needs OpenCV."),
    "undistortPoints": (["undistortedPoints = undistortPoints(points, intrinsics)",
                         "[undistortedPoints, reprojectionErrors] = undistortPoints(...)"],
                        "Moves M-by-2 image points to where an ideal (distortion-free) camera with the same K would "
                        "see them, by inverting the distortion model iteratively."),
    "world2img": (["imagePoints = world2img(worldPoints, cameraExtrinsics, intrinsics)",
                   "imagePoints = world2img(..., 'ApplyDistortion', true)"],
                  "Projects M-by-3 world points (or M-by-2 points on Z = 0) into the image. cameraExtrinsics is a "
                  "rigidtform3d that maps world to camera coordinates. Lens distortion is ignored unless "
                  "'ApplyDistortion' is true."),
    "worldToImage": (["imagePoints = worldToImage(intrinsics, rotationMatrix, translationVector, worldPoints)",
                      "imagePoints = worldToImage(intrinsics, tform, worldPoints)"],
                     "The older form of world2img with the row-vector convention: camera = world*R + t."),
    "img2world2d": (["worldPoints = img2world2d(imagePoints, cameraExtrinsics, intrinsics)"],
                    "Maps image points to [x y] on the world plane Z = 0, undoing lens distortion."),
    "pointsToWorld": (["worldPoints = pointsToWorld(intrinsics, rotationMatrix, translationVector, imagePoints)",
                       "worldPoints = pointsToWorld(intrinsics, tform, imagePoints)"],
                      "The older form of img2world2d (row-vector convention for R and t)."),
    "cameraProjection": (["P = cameraProjection(intrinsics, tform)"],
                         "The 3-by-4 projection matrix K*[R t] of a camera with extrinsics tform (rigidtform3d, world "
                         "to camera)."),
    "cameraMatrix": (["camMatrix = cameraMatrix(intrinsics, rotationMatrix, translationVector)",
                      "camMatrix = cameraMatrix(intrinsics, tform)"],
                     "The older 4-by-3 camera matrix [R; t]*K' for the row-vector convention, so that "
                     "[X Y Z 1]*camMatrix is proportional to [x y 1]."),
    "estimateExtrinsics": (["camExtrinsics = estimateExtrinsics(imagePoints, worldPoints, intrinsics)"],
                           "Pose of a calibrated camera from at least four 2-D/3-D correspondences (OpenCV's solvePnP; "
                           "planar targets use the IPPE solver). Returns a rigidtform3d from world to camera."),
    "extrinsics": (["[rotationMatrix, translationVector] = extrinsics(imagePoints, worldPoints, intrinsics)"],
                   "The older form of estimateExtrinsics in the row-vector convention."),
    "estworldpose": (["[worldPose, inlierIdx, status] = estworldpose(imagePoints, worldPoints, intrinsics)"],
                     "Pose of the camera in world coordinates (the inverse of estimateExtrinsics) as a rigidtform3d. "
                     "All points are treated as inliers; no outlier search is done."),
}
