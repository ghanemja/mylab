"""Help text for the feature functions in mylab/builtins/vision_features.py (written from scratch for mylab)."""

_POINTS = ("A column array of points: Location (M-by-2 [x y], single), Metric (strength), Count{extra}. Index it "
           "like a vector (p(1:10), p(end), p([3 1])), join with [p1; p2], and use selectStrongest(p, N), "
           "selectUniform(p, N, imageSize), isempty, length and plot(p) on it. Passing a points object where a "
           "matrix of points is expected uses its Location.")
_SCALED = ", Scale and Orientation (radians, measured from the image x axis)"
_NEEDS_CV = "Needs OpenCV (pip install \"mylab-lang[vision]\")."
_DETECT_COMMON = ("The image must be 2-D (grayscale). 'ROI', [x y w h] restricts the search to a rectangle; the "
                  "locations are still in full-image coordinates. Points come back strongest first.")

HELP = {
    "cornerPoints": (["p = cornerPoints(locations)", "p = cornerPoints(locations, 'Metric', m)"],
                     "Creates a cornerPoints array from an M-by-2 matrix of [x y] locations. "
                     + _POINTS.format(extra="")),
    "ORBPoints": (["p = ORBPoints(locations, 'Metric', m, 'Scale', s, 'Orientation', theta)"],
                  "Creates an ORBPoints array. Scale is the pyramid scale factor of each point (1 for the finest "
                  "level). " + _POINTS.format(extra=_SCALED)),
    "SIFTPoints": (["p = SIFTPoints(locations, 'Metric', m, 'Scale', s, 'Orientation', theta)"],
                   "Creates a SIFTPoints array; Scale is the blob's Gaussian scale (sigma). "
                   + _POINTS.format(extra=_SCALED)),
    "SURFPoints": (["p = SURFPoints(locations, 'Metric', m, 'Scale', s, 'Orientation', theta, "
                    "'SignOfLaplacian', sgn)"],
                   "Creates a SURFPoints array (default Scale 1.6). SignOfLaplacian (int8, -1, 0 or 1) tells dark "
                   "blobs from bright ones. " + _POINTS.format(extra=_SCALED + ", SignOfLaplacian")),
    "BRISKPoints": (["p = BRISKPoints(locations, 'Metric', m, 'Scale', s, 'Orientation', theta)"],
                    "Creates a BRISKPoints array; Scale is the diameter of the sampling pattern in pixels. "
                    + _POINTS.format(extra=_SCALED)),
    "KAZEPoints": (["p = KAZEPoints(locations, 'Metric', m, 'Scale', s, 'Orientation', theta)"],
                   "Creates a KAZEPoints array; Scale is the blob scale. " + _POINTS.format(extra=_SCALED)),
    "MSERRegions": (["r = MSERRegions(pixelLists)"],
                    "Creates MSER regions from a cell array of M-by-2 [x y] pixel lists. Each region has Location "
                    "(centroid), Axes ([major minor] lengths of the ellipse with the same second moments), "
                    "Orientation (radians) and PixelList. plot(r) draws the ellipses; plot(r, 'showPixelList', "
                    "true) also marks the pixels."),
    "detectHarrisFeatures": (["points = detectHarrisFeatures(I)",
                              "points = detectHarrisFeatures(I, 'MinQuality', q, 'FilterSize', n, 'ROI', roi)"],
                             "Finds corners with the Harris-Stephens measure and returns a cornerPoints array. The "
                             "image is smoothed with an n-by-n Gaussian (FilterSize, odd, default 5); corners are "
                             "local maxima whose response exceeds MinQuality (default 0.01) times the strongest, "
                             "refined to sub-pixel positions. " + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectMinEigenFeatures": (["points = detectMinEigenFeatures(I)",
                                "points = detectMinEigenFeatures(I, 'MinQuality', q, 'FilterSize', n, 'ROI', roi)"],
                               "Finds corners with the minimum-eigenvalue (Shi-Tomasi) measure; the options and "
                               "output are those of detectHarrisFeatures. " + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectFASTFeatures": (["points = detectFASTFeatures(I)",
                            "points = detectFASTFeatures(I, 'MinContrast', c, 'MinQuality', q, 'ROI', roi)"],
                           "Finds corners with the FAST segment test. MinContrast (0..1, default 0.2) is the "
                           "intensity difference a corner needs, as a fraction of the class range; MinQuality "
                           "(default 0.1) drops corners weaker than that fraction of the strongest. Returns a "
                           "cornerPoints array. " + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectORBFeatures": (["points = detectORBFeatures(I)",
                           "points = detectORBFeatures(I, 'ScaleFactor', s, 'NumLevels', n, 'ROI', roi)"],
                          "Finds oriented FAST corners over an image pyramid (ScaleFactor default 1.2, NumLevels "
                          "default 8) and returns an ORBPoints array. " + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectSIFTFeatures": (["points = detectSIFTFeatures(I)",
                            "points = detectSIFTFeatures(I, 'ContrastThreshold', c, 'EdgeThreshold', e, "
                            "'NumLayersInOctave', n, 'Sigma', s, 'ROI', roi)"],
                           "Finds scale-invariant blobs (difference of Gaussians) and returns a SIFTPoints array. "
                           "ContrastThreshold (default 0.0133) removes weak blobs, EdgeThreshold (default 10) "
                           "removes edge-like ones; NumLayersInOctave defaults to 3 and Sigma to 1.6. "
                           + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectBRISKFeatures": (["points = detectBRISKFeatures(I)",
                             "points = detectBRISKFeatures(I, 'MinContrast', c, 'MinQuality', q, 'NumOctaves', n, "
                             "'ROI', roi)"],
                            "Finds BRISK keypoints (multi-scale AGAST corners) and returns a BRISKPoints array; "
                            "MinContrast defaults to 0.2, MinQuality to 0.1 and NumOctaves to 4. " + _DETECT_COMMON
                            + " " + _NEEDS_CV),
    "detectKAZEFeatures": (["points = detectKAZEFeatures(I)",
                            "points = detectKAZEFeatures(I, 'Diffusion', d, 'Threshold', t, 'NumOctaves', n, "
                            "'NumScaleLevels', k, 'ROI', roi)"],
                           "Finds blobs in a nonlinear (anisotropic diffusion) scale space and returns a KAZEPoints "
                           "array. Diffusion is 'region' (default), 'sharpedge' or 'edge'; Threshold defaults to "
                           "0.0001. " + _DETECT_COMMON + " " + _NEEDS_CV),
    "detectMSERFeatures": (["regions = detectMSERFeatures(I)", "[regions, cc] = detectMSERFeatures(I)",
                            "regions = detectMSERFeatures(I, 'ThresholdDelta', d, 'RegionAreaRange', [lo hi], "
                            "'MaxAreaVariation', v, 'ROI', roi)"],
                           "Finds maximally stable extremal regions and returns an MSERRegions array; cc is a "
                           "bwconncomp-style struct of the same regions. ThresholdDelta is the intensity step in "
                           "percent of the range (default 2), RegionAreaRange the allowed sizes in pixels (default "
                           "[30 14000]) and MaxAreaVariation the allowed size change between steps (default 0.25). "
                           + _NEEDS_CV),
    "detectSURFFeatures": (["points = detectSURFFeatures(I)",
                            "points = detectSURFFeatures(I, 'MetricThreshold', t, 'NumOctaves', n, "
                            "'NumScaleLevels', k, 'ROI', roi)"],
                           "Finds blobs as maxima of the determinant of the Hessian over scale and returns a "
                           "SURFPoints array (with SignOfLaplacian). mylab computes this itself with Gaussian "
                           "derivatives (NumPy/SciPy) instead of OpenCV's non-free SURF, so the detections and "
                           "metric values are close to, but not the same as, other SURF implementations. "
                           "MetricThreshold defaults to 1000, NumOctaves to 3 and NumScaleLevels to 4. "
                           + _DETECT_COMMON),
    "binaryFeatures": (["f = binaryFeatures(featureVectors)"],
                       "Wraps an M-by-N uint8 matrix of packed bits (one descriptor per row) as a binaryFeatures "
                       "object with properties Features, NumBits and NumFeatures. matchFeatures compares binary "
                       "features with the Hamming distance."),
    "extractFeatures": (["[features, validPoints] = extractFeatures(I, points)",
                         "[...] = extractFeatures(I, points, 'Method', m, 'BlockSize', n, 'Upright', tf, "
                         "'FeatureSize', 64)"],
                        "Computes a descriptor around each point. points is a points object, MSERRegions or an "
                        "M-by-2 matrix. Method is 'Auto' (default: picks the descriptor matching the points type), "
                        "'SIFT', 'SURF' (64 or 128 values, computed in NumPy), 'KAZE' (numeric), 'ORB', 'BRISK', "
                        "'FREAK' (binaryFeatures; FREAK needs an OpenCV build with the extra modules) or 'Block' "
                        "(the raw BlockSize-by-BlockSize neighbourhood, default 11). 'Auto' uses BRISK descriptors "
                        "for corner points. validPoints holds the points that got a descriptor (points too close to "
                        "the border are dropped), with orientations filled in. Upright true skips orientation "
                        "estimation. " + _NEEDS_CV),
    "matchFeatures": (["indexPairs = matchFeatures(features1, features2)",
                       "[indexPairs, matchMetric] = matchFeatures(..., 'Method', m, 'MatchThreshold', t, "
                       "'MaxRatio', r, 'Metric', 'SSD', 'Unique', tf)"],
                      "Matches two sets of descriptors and returns a P-by-2 uint32 matrix of row indices "
                      "[i1 i2]. Numeric features are scaled to unit length and compared with 'SSD' (default) or "
                      "'SAD'; binaryFeatures use the Hamming distance. A match is kept when its distance is at "
                      "most MatchThreshold percent of the largest possible distance (default 1 for numeric, 10 "
                      "for binary features), the ratio of the best to the second-best distance is at most "
                      "MaxRatio (default 0.6). Several features of the first set may match the same feature of "
                      "the second; 'Unique', true keeps only mutual best matches. Both 'Exhaustive' and 'Approximate' "
                      "methods compute exact nearest neighbours. matchMetric holds the distances (Hamming "
                      "distances as a percentage of the bit count)."),
    "showMatchedFeatures": (["showMatchedFeatures(I1, I2, points1, points2)",
                             "showMatchedFeatures(I1, I2, points1, points2, method)",
                             "showMatchedFeatures(..., 'PlotOptions', {style1, style2, lineStyle}, 'Parent', ax)",
                             "ax = showMatchedFeatures(...)"],
                            "Shows two images with matched points and lines joining each pair. method is "
                            "'falsecolor' (default: I1 in red, I2 in cyan, overlaid), 'blend' (the average of the "
                            "two) or 'montage' (side by side). The default styles are 'ro', 'g+' and 'y-'."),
}
