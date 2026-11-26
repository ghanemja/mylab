"""Help text for the image functions in mylab/builtins/images_more.py (written from scratch for mylab)."""

_COLOURS = "The input is an M-by-3 list of colours or an M-by-N-by-3 image; the result has the same shape."
_WP = (" 'WhitePoint' names the reference white ('d65' by default; also 'd50', 'd55', 'a', 'c', 'e', 'icc', "
       "'f2', 'f7', 'f11' or [X Y Z]) and 'ColorSpace' the kind of RGB ('srgb', 'adobe-rgb-1998' or "
       "'linear-rgb').")
_CONN = " conn is 4 or 8 (the default) for images and 6, 18 or 26 for volumes, or a 3-by-3 matrix of neighbours."
_CLASS = "The result has the class of the input."
_TFORM = ("A geometric transformation is an object made by affine2d, projective2d, affinetform2d, "
          "rigidtform2d, simtform2d, transltform2d, projtform2d, fitgeotrans or fitgeotform2d.")

HELP = {
    # -- colour ---------------------------------------------------------------------------------------
    "rgb2lab": (["lab = rgb2lab(rgb)", "lab = rgb2lab(rgb, Name, Value)"],
                "Converts RGB colours to CIE 1976 L*a*b*: L from 0 to 100, a (green to red) and b (blue to "
                "yellow). Integer images are scaled to 0..1 first. " + _COLOURS + _WP),
    "lab2rgb": (["rgb = lab2rgb(lab)", "rgb = lab2rgb(lab, Name, Value)"],
                "Converts L*a*b* colours to RGB. Colours outside the RGB gamut give values outside 0..1 "
                "unless 'OutputType' is 'uint8' or 'uint16'." + _WP),
    "rgb2xyz": (["xyz = rgb2xyz(rgb)", "xyz = rgb2xyz(rgb, Name, Value)"],
                "Converts RGB colours to CIE 1931 XYZ (Y = 1 for white). " + _COLOURS + _WP),
    "xyz2rgb": (["rgb = xyz2rgb(xyz)", "rgb = xyz2rgb(xyz, Name, Value)"],
                "Converts CIE XYZ colours to RGB; 'OutputType' may be 'double', 'single', 'uint8' or 'uint16'."
                + _WP),
    "xyz2lab": (["lab = xyz2lab(xyz)", "lab = xyz2lab(xyz, 'WhitePoint', wp)"],
                "Converts CIE XYZ colours to L*a*b* for the given reference white (default 'd65')."),
    "lab2xyz": (["xyz = lab2xyz(lab)", "xyz = lab2xyz(lab, 'WhitePoint', wp)"],
                "Converts L*a*b* colours to CIE XYZ for the given reference white (default 'd65')."),
    "whitepoint": (["xyz = whitepoint", "xyz = whitepoint(name)"],
                   "XYZ coordinates (Y = 1) of a standard illuminant: 'd65', 'd50', 'd55', 'a', 'c', 'e', 'f2', "
                   "'f7', 'f11' or 'icc' (the default)."),
    "rgb2ycbcr": ("ycbcr = rgb2ycbcr(rgb)",
                  "Converts RGB to the YCbCr colour space of digital video (ITU-R BT.601): luma from 16 to "
                  "235 and chroma from 16 to 240 on the uint8 scale, or the same divided by 255 for double "
                  "images. " + _CLASS),
    "ycbcr2rgb": ("rgb = ycbcr2rgb(ycbcr)", "Converts YCbCr colours back to RGB. " + _CLASS),
    "rgb2ntsc": ("yiq = rgb2ntsc(rgb)",
                 "Converts RGB to the NTSC colour space: luminance Y and the two chrominance values I and Q. "
                 "The result is double."),
    "ntsc2rgb": ("rgb = ntsc2rgb(yiq)", "Converts NTSC (YIQ) colours to RGB, limited to the range 0..1."),
    # -- contrast and thresholds ----------------------------------------------------------------------
    "imhistmatch": (["J = imhistmatch(I, ref)", "J = imhistmatch(I, ref, nbins)", "[J, hgram] = imhistmatch(...)"],
                    "Adjusts the gray levels of I so that its histogram resembles that of ref, using nbins "
                    "(default 64) bins. The channels of an RGB image are matched one by one."),
    "adapthisteq": (["J = adapthisteq(I)", "J = adapthisteq(I, Name, Value)"],
                    "Contrast-limited adaptive histogram equalisation: the image is cut into tiles ('NumTiles', "
                    "default [8 8]), each tile is equalised with its histogram clipped at 'ClipLimit' (0 to "
                    "1, default 0.01) so that noise is not amplified too much, and the tile mappings are "
                    "blended to hide the seams. 'NBins' (default 256), 'Range' ('full' or 'original') and "
                    "'Distribution' ('uniform', 'rayleigh' or 'exponential', with 'Alpha') are also "
                    "available."),
    "multithresh": (["thresh = multithresh(A)", "[thresh, metric] = multithresh(A, n)"],
                    "n (default 1) thresholds that split the values of A into classes by Otsu's method, which "
                    "makes the classes as different from each other as possible. metric is between 0 and 1; "
                    "higher means a cleaner split. Use the thresholds with imquantize."),
    "imquantize": (["Q = imquantize(A, levels)", "[Q, index] = imquantize(A, levels, values)"],
                   "Replaces each element by the number of its interval: 1 up to and including levels(1), 2 up "
                   "to levels(2), and so on. With values the result takes values(k) for interval k."),
    "adaptthresh": (["T = adaptthresh(I)", "T = adaptthresh(I, sensitivity)", "T = adaptthresh(..., Name, Value)"],
                    "Locally adaptive threshold for imbinarize: a fraction of the local average brightness. "
                    "sensitivity (0 to 1, default 0.5) turns more pixels into foreground as it grows. "
                    "'NeighborhoodSize' (default about one eighth of the image), 'Statistic' ('mean', 'median' "
                    "or 'gaussian') and 'ForegroundPolarity' ('bright' or 'dark') are available."),
    # -- filters --------------------------------------------------------------------------------------
    "imboxfilt": (["B = imboxfilt(A)", "B = imboxfilt(A, filterSize)", "B = imboxfilt(..., Name, Value)"],
                  "Mean filter with a box of odd size (default 3). 'Padding' is 'replicate' (default), "
                  "'symmetric', 'circular' or a number; 'NormalizationFactor' replaces the division by the "
                  "number of pixels. " + _CLASS),
    "imsharpen": (["B = imsharpen(A)", "B = imsharpen(A, Name, Value)"],
                  "Sharpens by unsharp masking: adds back 'Amount' (default 0.8) times the difference between "
                  "the image and a Gaussian blur of standard deviation 'Radius' (default 1). 'Threshold' (0 to "
                  "1) leaves low-contrast detail alone. Only the lightness of an RGB image is sharpened."),
    "stdfilt": (["J = stdfilt(I)", "J = stdfilt(I, nhood)"],
                "Standard deviation of the neighbourhood of each pixel (3-by-3 by default); the image is "
                "mirrored at its edges. The result is double."),
    "rangefilt": (["J = rangefilt(I)", "J = rangefilt(I, nhood)"],
                  "Difference between the largest and smallest value in the neighbourhood of each pixel "
                  "(3-by-3 by default)."),
    "entropyfilt": (["J = entropyfilt(I)", "J = entropyfilt(I, nhood)"],
                    "Entropy of the gray levels in the neighbourhood of each pixel (9-by-9 by default), "
                    "computed on 256 levels."),
    "imgradient": (["[Gmag, Gdir] = imgradient(I)", "[Gmag, Gdir] = imgradient(I, method)",
                    "[Gmag, Gdir] = imgradient(Gx, Gy)"],
                   "Magnitude and direction (degrees, counter-clockwise from the positive x axis) of the image "
                   "gradient. The method is 'sobel' (default), 'prewitt', 'central', 'intermediate' or "
                   "'roberts'."),
    "imgradientxy": (["[Gx, Gy] = imgradientxy(I)", "[Gx, Gy] = imgradientxy(I, method)"],
                     "Gradient of an image along x (left to right) and y (top to bottom). The method is "
                     "'sobel' (default), 'prewitt', 'central' or 'intermediate'."),
    "imbilatfilt": (["J = imbilatfilt(I)", "J = imbilatfilt(I, degreeOfSmoothing, spatialSigma)",
                     "J = imbilatfilt(..., Name, Value)"],
                    "Bilateral filter: smooths while keeping edges, because a neighbour counts less the more "
                    "its value differs. degreeOfSmoothing is the variance allowed in value (default 1% of "
                    "the squared range of the class) and spatialSigma the width of the Gaussian in space "
                    "(default 1). 'NeighborhoodSize' and 'Padding' are available."),
    "imguidedfilter": (["B = imguidedfilter(A)", "B = imguidedfilter(A, G)", "B = imguidedfilter(..., Name, Value)"],
                       "Edge-preserving smoothing guided by the image G (A itself by default): locally B is a "
                       "linear function of G. 'NeighborhoodSize' defaults to [5 5] and 'DegreeOfSmoothing' to "
                       "1% of the squared range of the class."),
    "integralBoxFilter": (["B = integralBoxFilter(intA)", "B = integralBoxFilter(intA, filterSize)"],
                          "Box filter computed from an integral image: the mean over every window that fits "
                          "inside the image, so the result is smaller than the image by the filter size "
                          "minus one."),
    "blockproc": (["B = blockproc(A, [m n], fun)", "B = blockproc(A, [m n], fun, Name, Value)"],
                  "Applies fun to each m-by-n block of A and joins the results. fun gets a struct with the "
                  "fields data, location ([row col] of the block), blockSize, imageSize and border. Options: "
                  "'BorderSize' [v h] adds pixels around each block, 'TrimBorder' (default true) removes them "
                  "from the result, 'PadPartialBlocks' fills up the blocks at the right and bottom edges, and "
                  "'PadMethod' is a number, 'replicate' or 'symmetric'."),
    "nlfilter": (["B = nlfilter(A, [m n], fun)"],
                 "Sliding-window operation: B(i, j) is fun applied to the m-by-n neighbourhood of A(i, j), "
                 "which must return a scalar. The image is padded with zeros."),
    "colfilt": (["B = colfilt(A, [m n], 'sliding', fun)", "B = colfilt(A, [m n], 'distinct', fun)"],
                "Neighbourhood operation on columns: every m-by-n neighbourhood becomes one column of a "
                "temporary matrix that is passed to fun. For 'sliding', fun returns one value per column; for "
                "'distinct' blocks it returns a matrix of the same size."),
    "col2im": (["A = col2im(B, [m n], [M N], 'distinct')", "A = col2im(B, [m n], [M N], 'sliding')"],
               "Rearranges columns back into an image: the reverse of im2col."),
    # -- test images and features ---------------------------------------------------------------------
    "checkerboard": (["I = checkerboard", "I = checkerboard(n)", "I = checkerboard(n, p, q)"],
                     "Checkerboard image with squares of n pixels (default 10), 2p rows and 2q columns of "
                     "squares (default p = q = 4). The light squares are white in the left half and gray "
                     "(0.7) in the right half."),
    "phantom": (["P = phantom(n)", "P = phantom(def, n)", "[P, E] = phantom(...)"],
                "Head phantom made of ellipses, used to test tomography: 'Modified Shepp-Logan' (default, "
                "better contrast) or 'Shepp-Logan', n-by-n pixels (default 256). A matrix with one row "
                "[value a b x0 y0 phi] per ellipse defines another phantom."),
    "normxcorr2": ("C = normxcorr2(template, A)",
                   "Normalised cross-correlation of a template with an image: values from -1 to 1, with a peak "
                   "of 1 where the image matches the template exactly. C has size(A) + size(template) - 1 "
                   "elements each way; a peak at (r, c) means the template's lower right corner sits at A(r, c)."),
    "corner": (["C = corner(I)", "C = corner(I, method)", "C = corner(I, N)", "C = corner(..., Name, Value)"],
               "Corner points as [x y] rows, strongest first, at most N (default 200). The method is 'Harris' "
               "(default) or 'MinimumEigenvalue'; 'QualityLevel' (default 0.01), 'SensitivityFactor' (default "
               "0.04) and 'FilterCoefficients' are available."),
    "hough": (["[H, theta, rho] = hough(BW)", "[H, theta, rho] = hough(BW, Name, Value)"],
              "Hough transform for lines: every foreground pixel votes for all lines "
              "rho = x cos(theta) + y sin(theta) through it (x and y count from 0 at the first column and "
              "row). H has one row per rho and one column per theta (degrees, -90 to 89 by default). "
              "'RhoResolution' and 'Theta' change the grid."),
    "houghpeaks": (["peaks = houghpeaks(H, numpeaks)", "peaks = houghpeaks(H, numpeaks, Name, Value)"],
                   "Row and column of the largest values of a Hough matrix. After each peak its neighbourhood "
                   "('NHoodSize') is cleared; values below 'Threshold' (default half the maximum) are ignored."),
    "houghlines": (["lines = houghlines(BW, theta, rho, peaks)", "lines = houghlines(..., Name, Value)"],
                   "Line segments in BW that belong to the given Hough peaks: a struct array with point1, "
                   "point2 ([x y]), theta and rho. Gaps shorter than 'FillGap' (default 20) are bridged and "
                   "segments shorter than 'MinLength' (default 40) are dropped."),
    "radon": (["R = radon(I)", "[R, xp] = radon(I, theta)"],
              "Radon transform: each column of R holds the sums of the image along parallel lines for one "
              "angle (degrees, 0:179 by default). xp gives the distance of each row from the centre pixel."),
    "iradon": (["I = iradon(R, theta)", "I = iradon(R, theta, interp, filter, frequency_scaling, output_size)",
                "[I, H] = iradon(...)"],
               "Rebuilds an image from its projections by filtered back-projection. interp is 'linear' "
               "(default) or 'nearest'; the filter is 'Ram-Lak' (default), 'Shepp-Logan', 'Cosine', 'Hamming', "
               "'Hann' or 'none'."),
    "deconvwnr": (["J = deconvwnr(I, psf)", "J = deconvwnr(I, psf, nsr)"],
                  "Removes a known blur (the point-spread function psf) with a Wiener filter. nsr is the "
                  "noise-to-signal power ratio (default 0, a plain inverse filter)."),
    "deconvlucy": (["J = deconvlucy(I, psf)", "J = deconvlucy(I, psf, iter)", "J = deconvlucy(I, psf, iter, dampar)"],
                   "Removes a known blur with the Richardson-Lucy algorithm, which keeps the image "
                   "non-negative. iter (default 10) is the number of iterations; dampar damps changes "
                   "smaller than the noise. The plain algorithm is used, without acceleration."),
    "deconvreg": (["J = deconvreg(I, psf)", "[J, lagra] = deconvreg(I, psf, np, lrange, regop)"],
                  "Removes a known blur with a regularised filter that favours smooth results. np is the "
                  "noise power; the multiplier lagra is searched in lrange (default [1e-9 1e9]) so that the "
                  "blurred result differs from I by that amount. regop defaults to the Laplacian."),
    "demosaic": ("RGB = demosaic(I, sensorAlignment)",
                 "Turns a Bayer-pattern sensor image into an RGB image by gradient-corrected interpolation. "
                 "The alignment names the colours of the top-left 2-by-2 pixels: 'gbrg', 'grbg', 'bggr' or "
                 "'rggb'."),
    # -- morphology -----------------------------------------------------------------------------------
    "imreconstruct": (["J = imreconstruct(marker, mask)", "J = imreconstruct(marker, mask, conn)"],
                      "Morphological reconstruction: the marker is grown by repeated dilation but never above "
                      "the mask. For binary images this keeps the objects of the mask that contain a marker "
                      "pixel." + _CONN),
    "imregionalmax": (["BW = imregionalmax(I)", "BW = imregionalmax(I, conn)"],
                      "Regional maxima: connected groups of equal pixels all of whose neighbours are lower."
                      + _CONN),
    "imregionalmin": (["BW = imregionalmin(I)", "BW = imregionalmin(I, conn)"],
                      "Regional minima: connected groups of equal pixels all of whose neighbours are higher."
                      + _CONN),
    "imhmax": (["J = imhmax(I, h)", "J = imhmax(I, h, conn)"],
               "Lowers every peak by h and removes the maxima that are less than h high."),
    "imhmin": (["J = imhmin(I, h)", "J = imhmin(I, h, conn)"],
               "Raises every valley by h and removes the minima that are less than h deep."),
    "imextendedmax": (["BW = imextendedmax(I, h)", "BW = imextendedmax(I, h, conn)"],
                      "Regional maxima after peaks lower than h have been removed (the maxima of imhmax)."),
    "imextendedmin": (["BW = imextendedmin(I, h)", "BW = imextendedmin(I, h, conn)"],
                      "Regional minima after valleys shallower than h have been removed (the minima of imhmin)."),
    "imimposemin": (["J = imimposemin(I, BW)", "J = imimposemin(I, BW, conn)"],
                    "Changes I so that its only regional minima are where BW is true; used to place the seeds "
                    "of a watershed."),
    "watershed": (["L = watershed(A)", "L = watershed(A, conn)"],
                  "Watershed transform: A is flooded from its regional minima and each catchment basin gets a "
                  "label. Pixels where basins meet are 0." + _CONN),
    "imclearborder": (["J = imclearborder(I)", "J = imclearborder(I, conn)"],
                      "Removes the objects that touch the edge of a binary image, or in a gray image the "
                      "structures that are lighter than their surroundings and connected to the edge."),
    "bwlabeln": (["L = bwlabeln(BW)", "[L, n] = bwlabeln(BW, conn)"],
                 "Labels the connected components of a binary array of any dimension (8-connected in 2-D and "
                 "26-connected in 3-D by default)."),
    "bwulterode": (["BW2 = bwulterode(BW)", "BW2 = bwulterode(BW, method, conn)"],
                   "Ultimate erosion: the regional maxima of the distance to the background, one spot for "
                   "each rounded part of an object."),
    "bwhitmiss": (["BW2 = bwhitmiss(BW, SE1, SE2)", "BW2 = bwhitmiss(BW, interval)"],
                  "Hit-or-miss transform: keeps the pixels whose neighbourhood contains SE1 in the foreground "
                  "and SE2 in the background. An interval is a matrix of 1 (foreground), -1 (background) and 0 "
                  "(either)."),
    "bwselect": (["BW2 = bwselect(BW, c, r)", "[BW2, idx] = bwselect(BW, c, r, n)"],
                 "Keeps the objects that contain the pixels at columns c and rows r (n-connected, default 8)."),
    "bwtraceboundary": (["B = bwtraceboundary(BW, P, fstep)", "B = bwtraceboundary(BW, P, fstep, conn, N, dir)"],
                        "Follows the boundary of an object from the pixel P = [row col], looking first in the "
                        "compass direction fstep ('N', 'NE', 'E', ...). B lists the [row col] of at most N "
                        "boundary pixels, going 'clockwise' (default) or 'counterclockwise'."),
    "bwconvhull": (["CH = bwconvhull(BW)", "CH = bwconvhull(BW, method, conn)"],
                   "Convex hull image: the smallest convex region that covers all foreground pixels ('union', "
                   "the default) or one hull per object ('objects')."),
    "bwpropfilt": (["BW2 = bwpropfilt(BW, prop, range)", "BW2 = bwpropfilt(BW, prop, n, keep)",
                    "BW2 = bwpropfilt(BW, I, prop, ...)"],
                   "Keeps the objects whose regionprops property lies in [low high], or the n objects with the "
                   "'largest' (default) or 'smallest' values."),
    "bwareafilt": (["BW2 = bwareafilt(BW, range)", "BW2 = bwareafilt(BW, n, keep, conn)"],
                   "Keeps the objects whose area lies in [low high], or the n largest (or 'smallest') objects."),
    "bweuler": (["e = bweuler(BW)", "e = bweuler(BW, conn)"],
                "Euler number: the number of objects minus the number of holes (8-connected objects by "
                "default)."),
    "bwarea": ("a = bwarea(BW)",
               "Area of the foreground with a weighting of the 2-by-2 pixel patterns that allows for the "
               "slanted edges of objects; close to, but not the same as, the pixel count."),
    "bwskel": ("B = bwskel(BW)",
               "Skeleton of a binary image: objects are thinned to lines one pixel wide that keep their "
               "connectivity."),
    "makelut": ("lut = makelut(fun, n)",
                "Lookup table for applylut: fun is called with every possible n-by-n (n = 2 or 3) binary "
                "neighbourhood and its results are stored."),
    "applylut": ("A = applylut(BW, lut)",
                 "Neighbourhood operation on a binary image by table lookup: each pixel's 2-by-2 or 3-by-3 "
                 "neighbourhood selects an entry of the table made by makelut."),
    "regionprops3": (["stats = regionprops3(BW, properties)", "stats = regionprops3(L, V, properties)"],
                     "Measurements of the objects in a volume, returned as a table with one row per object: "
                     "'Volume', 'Centroid', 'BoundingBox', 'EquivDiameter', 'Extent', 'VoxelIdxList', "
                     "'VoxelList', 'Image', and with an intensity volume 'MeanIntensity', 'MinIntensity', "
                     "'MaxIntensity' and 'VoxelValues'."),
    # -- geometry -------------------------------------------------------------------------------------
    "affine2d": (["tform = affine2d", "tform = affine2d(T)"],
                 "Affine transformation with the 3-by-3 matrix T used as [x y 1] = [u v 1] * T (the last "
                 "column must be [0; 0; 1]). Methods: transformPointsForward, transformPointsInverse, invert, "
                 "outputLimits, isRigid, isSimilarity and isTranslation."),
    "projective2d": (["tform = projective2d", "tform = projective2d(T)"],
                     "Projective transformation with the 3-by-3 matrix T used as [x y w] = [u v 1] * T, "
                     "followed by division by w."),
    "affinetform2d": (["tform = affinetform2d", "tform = affinetform2d(A)"],
                      "Affine transformation with the 3-by-3 matrix A used as [x; y; 1] = A * [u; v; 1] (the "
                      "last row must be [0 0 1])."),
    "projtform2d": (["tform = projtform2d", "tform = projtform2d(A)"],
                    "Projective transformation with the 3-by-3 matrix A used as [x; y; w] = A * [u; v; 1]."),
    "rigidtform2d": (["tform = rigidtform2d", "tform = rigidtform2d(angle, translation)",
                      "tform = rigidtform2d(R, translation)", "tform = rigidtform2d(A)"],
                     "Rotation by an angle in degrees (or a rotation matrix R) followed by a translation. "
                     "Properties: RotationAngle, Translation, R and A."),
    "simtform2d": (["tform = simtform2d", "tform = simtform2d(scale, angle, translation)", "tform = simtform2d(A)"],
                   "Similarity transformation: scaling, rotation by an angle in degrees and translation. "
                   "Properties: Scale, RotationAngle, Translation, R and A."),
    "transltform2d": (["tform = transltform2d", "tform = transltform2d([tx ty])"],
                      "Translation by [tx ty]. Properties: Translation and A."),
    "fitgeotrans": ("tform = fitgeotrans(movingPoints, fixedPoints, type)",
                    "Fits a transformation that takes the moving points onto the fixed points (n-by-2 matrices "
                    "of [x y]) in the least-squares sense: 'nonreflectivesimilarity' (2 pairs or more), "
                    "'similarity' (3), 'affine' (3) or 'projective' (4). Returns an affine2d or projective2d "
                    "object."),
    "fitgeotform2d": ("tform = fitgeotform2d(movingPoints, fixedPoints, type)",
                      "Fits a transformation that takes the moving points onto the fixed points: 'similarity', "
                      "'reflectivesimilarity', 'affine' or 'projective'. Returns a simtform2d, affinetform2d or "
                      "projtform2d object."),
    "imref2d": (["R = imref2d(imageSize)", "R = imref2d(imageSize, pixelExtentX, pixelExtentY)",
                 "R = imref2d(imageSize, xWorldLimits, yWorldLimits)"],
                "Describes where an image lies in world coordinates. Properties: ImageSize, XWorldLimits, "
                "YWorldLimits, PixelExtentInWorldX, PixelExtentInWorldY, ImageExtentInWorldX, "
                "ImageExtentInWorldY, XIntrinsicLimits and YIntrinsicLimits. Methods: worldToIntrinsic, "
                "intrinsicToWorld, worldToSubscript, contains and sizesMatch."),
    "imwarp": (["B = imwarp(A, tform)", "B = imwarp(A, tform, interp)", "[B, RB] = imwarp(A, RA, tform)",
                "B = imwarp(..., 'OutputView', R)", "B = imwarp(..., 'FillValues', v)"],
               "Applies a geometric transformation to an image. The result is just large enough to hold the "
               "whole transformed image unless 'OutputView' gives an imref2d to draw into; RB tells where the "
               "result lies. interp is 'linear' (default), 'nearest' or 'cubic'; pixels that come from outside "
               "the image get 'FillValues' (default 0). " + _TFORM),
    "maketform": (["T = maketform('affine', A)", "T = maketform('affine', U, X)", "T = maketform('projective', A)",
                   "T = maketform('projective', U, X)"],
                  "Older form of a transformation: a struct for imtransform, tformfwd and tforminv. A is used "
                  "as [x y 1] = [u v 1] * A, or the transformation is found from three (affine) or four "
                  "(projective) points U and their images X."),
    "cp2tform": ("T = cp2tform(movingPoints, fixedPoints, type)",
                 "Fits a transformation to pairs of control points and returns a maketform struct. The type is "
                 "'nonreflective similarity', 'similarity', 'affine' or 'projective'."),
    "tformfwd": (["X = tformfwd(T, U)", "[x, y] = tformfwd(T, u, v)"],
                 "Applies a transformation (a maketform struct or a transformation object) to points."),
    "tforminv": (["U = tforminv(T, X)", "[u, v] = tforminv(T, x, y)"],
                 "Applies the inverse of a transformation to points."),
    "imtransform": (["B = imtransform(A, T)", "B = imtransform(A, T, interp)",
                     "[B, xdata, ydata] = imtransform(..., Name, Value)"],
                    "Applies a transformation made by maketform to an image. The result covers the transformed "
                    "image unless 'XData' and 'YData' give the range to show; 'XYScale', 'Size' and "
                    "'FillValues' are available. interp is 'bilinear' (default), 'nearest' or 'bicubic'."),
    "impyramid": ("B = impyramid(A, direction)",
                  "One step of a Gaussian pyramid: 'reduce' smooths and halves the size (rounding up), "
                  "'expand' interpolates to 2n-1 pixels each way."),
    "imresize3": (["B = imresize3(V, scale)", "B = imresize3(V, [rows cols planes])", "B = imresize3(..., method)"],
                  "Resizes a volume. The method is 'cubic' (default), 'linear', 'nearest', 'box', 'triangle', "
                  "'lanczos2' or 'lanczos3'; shrinking is anti-aliased unless 'Antialiasing' is false."),
    # -- arithmetic, profiles and display -------------------------------------------------------------
    "imlincomb": (["Z = imlincomb(K1, A1, K2, A2, ...)", "Z = imlincomb(..., K)", "Z = imlincomb(..., outputClass)"],
                  "Linear combination K1*A1 + K2*A2 + ... (+ K) of images, computed in double and rounded and "
                  "saturated only once at the end. The result has the class of the first image unless another "
                  "class is named."),
    "improfile": (["c = improfile(I, xi, yi)", "c = improfile(I, xi, yi, n)", "[cx, cy, c] = improfile(...)",
                   "c = improfile(..., method)"],
                  "Values of an image along the line (or polyline) through the points (xi, yi), at n evenly "
                  "spaced points (about one per pixel by default). The method is 'nearest' (default), "
                  "'bilinear' or 'bicubic'. Choosing the line with the mouse is not supported."),
    "imtile": (["out = imtile(images)", "out = imtile(images, Name, Value)"],
               "Arranges several images (a cell array, or a stack along the third or fourth dimension) in a "
               "grid and returns one image. 'GridSize' [rows cols], 'BorderSize', 'BackgroundColor' and "
               "'ThumbnailSize' are available; images of other sizes are resized to the first one."),
    "montage": (["montage(images)", "montage(images, Name, Value)", "h = montage(...)"],
                "Shows several images side by side in one figure; see imtile for the inputs and options."),
    "imfuse": (["C = imfuse(A, B)", "C = imfuse(A, B, method)", "C = imfuse(..., Name, Value)"],
               "Combines two images into one uint8 picture for comparison: 'falsecolor' (default; A in green "
               "and B in magenta, so equal areas look gray), 'blend', 'diff' (absolute difference), "
               "'checkerboard' or 'montage' (side by side). 'Scaling' is 'independent' (default), 'joint' or "
               "'none'; 'ColorChannels' is 'green-magenta', 'red-cyan' or [R G B] with 1 for A, 2 for B and 0 "
               "for none. The smaller image is padded with zeros."),
    "imshowpair": (["imshowpair(A, B)", "imshowpair(A, B, method)", "h = imshowpair(...)"],
                   "Shows two images together to compare them; the methods and options are those of imfuse."),
    "dicomread": (["X = dicomread(filename)", "X = dicomread(info)"],
                  "Reads the pixel data of a DICOM file. Needs the optional pydicom package."),
    "dicominfo": ("info = dicominfo(filename)",
                  "Reads the header of a DICOM file into a struct with one field per data element (and Width "
                  "and Height). Needs the optional pydicom package."),
}
