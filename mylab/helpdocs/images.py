"""Help text for the image functions in mylab/builtins/images.py (written from scratch for mylab)."""

_CLASSES = ("Value ranges follow the class: 0..255 for uint8, 0..65535 for uint16, -32768..32767 for int16 "
            "and 0..1 for double and single.")
_SE = ("The structuring element is a matrix of zeros and ones, such as the result of strel; its centre is "
       "element floor((size+1)/2).")

HELP = {
    # -- files -----------------------------------------------------------------------------------
    "imread": (["A = imread(filename)", "A = imread(filename, fmt)", "[X, map] = imread(...)",
                "[A, map, alpha] = imread(...)", "A = imread(filename, k)", "A = imread(..., 'Frames', 'all')"],
               "Reads an image file (PNG, JPEG, BMP, GIF, TIFF and the other formats Pillow knows) or an "
               "http(s) address. Grayscale images come back as an H-by-W array and colour images as "
               "H-by-W-by-3, of class uint8 (uint16 for 16-bit grayscale files, logical for 1-bit files). "
               "For an indexed file the first output holds zero-based uint8 colour numbers and map is the "
               "N-by-3 colormap; use ind2rgb to get colours. alpha is the transparency channel, or empty. "
               "A frame number (or 'Frames', k or 'all') picks frames of an animated GIF or multi-page TIFF; "
               "several frames are stacked along the fourth dimension. Files are also looked for on the "
               "search path. 16-bit RGB files are read with 8 bits per channel. The sample pictures that "
               "ship with MATLAB are not included."),
    "imwrite": (["imwrite(A, filename)", "imwrite(A, filename, fmt)", "imwrite(X, map, filename)",
                 "imwrite(..., 'Quality', q)", "imwrite(..., 'Alpha', a)",
                 "imwrite(..., 'WriteMode', 'append', 'DelayTime', seconds, 'LoopCount', n)"],
                "Writes an image to a file; the format comes from the extension or from fmt ('png', 'jpg', "
                "'bmp', 'gif', 'tif', ...). uint8 data is written as is, double and single data is taken "
                "to span 0..1, logical data becomes a 1-bit image and 16-bit grayscale is kept in PNG and "
                "TIFF files. With a colormap the first input is an indexed image. 'Quality' (0-100, default "
                "75) applies to JPEG and 'Alpha' adds transparency to a PNG. For GIF files 'WriteMode', "
                "'append' adds a frame to an existing file, 'DelayTime' is the time each frame is shown in "
                "seconds, and 'LoopCount', Inf makes the animation repeat forever; a 4-D array writes "
                "several frames at once. Other options are accepted and ignored."),
    "imfinfo": (["info = imfinfo(filename)", "info = imfinfo(filename, fmt)"],
                "Describes an image file without loading the pixels: a struct with Filename, FileModDate, "
                "FileSize, Format, Width, Height, BitDepth, ColorType ('grayscale', 'truecolor' or "
                "'indexed'), NumberOfFrames, DelayTime (hundredths of a second, GIF) and Colormap. A file "
                "with several frames gives one struct per frame."),
    "getframe": (["F = getframe", "F = getframe(h)"],
                 "Captures the current figure (or the figure or axes h) as a movie frame: a struct whose "
                 "cdata field is the H-by-W-by-3 uint8 picture and whose colormap field is empty. The whole "
                 "figure is captured even when an axes is given."),
    "frame2im": (["A = frame2im(F)", "[A, map] = frame2im(F)"],
                 "Picture stored in a movie frame from getframe or im2frame (its cdata field)."),
    "im2frame": (["F = im2frame(A)", "F = im2frame(X, map)"],
                 "Wraps an image in a movie-frame struct with fields cdata and colormap."),
    # -- class and colour ------------------------------------------------------------------------
    "im2double": (["B = im2double(A)", "B = im2double(X, 'indexed')"],
                  "Converts an image to double, rescaling integer classes to 0..1 (a uint8 255 becomes 1). "
                  "double input is returned unchanged. With 'indexed' the values are colour numbers, which "
                  "are only shifted to count from 1."),
    "im2single": (["B = im2single(A)"], "Like im2double, but the result is of class single."),
    "im2uint8": (["B = im2uint8(A)", "B = im2uint8(X, 'indexed')"],
                 "Converts an image to uint8, rescaling as needed: 0..1 floating-point values map to 0..255 "
                 "(values outside are clipped), uint16 values are divided by 257, and true becomes 255."),
    "im2uint16": (["B = im2uint16(A)"],
                  "Converts an image to uint16, rescaling so that the full range of the input class maps to "
                  "0..65535."),
    "im2int16": (["B = im2int16(A)"],
                 "Converts an image to int16, rescaling so that the full range of the input class maps to "
                 "-32768..32767."),
    "mat2gray": (["B = mat2gray(A)", "B = mat2gray(A, [low high])"],
                 "Rescales a matrix to a double image between 0 and 1: low maps to 0 and high to 1, with "
                 "values outside clipped. Without limits the smallest and largest elements are used."),
    "rgb2gray": (["G = rgb2gray(RGB)", "newmap = rgb2gray(map)"],
                 "Grayscale version of an RGB image, the weighted sum 0.2989 R + 0.5870 G + 0.1140 B, in the "
                 "class of the input. A colormap is converted to a gray colormap. im2gray is the same but "
                 "also accepts images that are already grayscale."),
    "im2gray": (["G = im2gray(A)"], "Like rgb2gray, and a grayscale image is passed through unchanged."),
    "rgb2hsv": (["HSV = rgb2hsv(RGB)", "[h, s, v] = rgb2hsv(RGB)", "hsvmap = rgb2hsv(map)"],
                "Converts colours from red-green-blue to hue-saturation-value. The result is double with "
                "all three components between 0 and 1; integer images are rescaled first."),
    "hsv2rgb": (["RGB = hsv2rgb(HSV)", "[r, g, b] = hsv2rgb(h, s, v)", "map = hsv2rgb(hsvmap)"],
                "Converts hue-saturation-value colours (each 0..1) to red-green-blue, as double."),
    "ind2rgb": (["RGB = ind2rgb(X, map)"],
                "Turns an indexed image into a double RGB image by looking every colour number up in the "
                "colormap. Integer-class indices count from 0, double indices from 1."),
    "ind2gray": (["G = ind2gray(X, map)"], "Grayscale image from an indexed image and its colormap."),
    "rgb2ind": (["[X, map] = rgb2ind(RGB, n)", "X = rgb2ind(RGB, map)"],
                "Converts an RGB image to an indexed image. With a number, at most n colours are chosen "
                "(exactly, when the picture has no more than n; otherwise by median-cut quantization); with "
                "a colormap every pixel gets the nearest colour in it. X is zero-based uint8 for up to 256 "
                "colours. No dithering is applied."),
    "gray2ind": (["[X, map] = gray2ind(I)", "[X, map] = gray2ind(I, n)"],
                 "Converts a grayscale image to an indexed image with an n-level gray colormap (64 by "
                 "default, 2 for a binary image)."),
    "label2rgb": (["RGB = label2rgb(L)", "RGB = label2rgb(L, map)", "RGB = label2rgb(L, map, zerocolor)",
                   "RGB = label2rgb(L, map, zerocolor, 'shuffle')"],
                  "Colours a label matrix for display: each label gets a colour from the colormap (a name "
                  "such as 'jet', the default, or an N-by-3 matrix) and label 0 gets zerocolor (white by "
                  "default; a letter such as 'k' or an RGB triple). 'shuffle' mixes the colour order. The "
                  "result is uint8."),
    "imcomplement": (["B = imcomplement(A)"],
                     "Negative of an image: 255 - A for uint8, 1 - A for floating point, and logical NOT "
                     "for a binary image."),
    # -- thresholds, histograms, contrast --------------------------------------------------------
    "im2bw": (["BW = im2bw(I)", "BW = im2bw(I, level)", "BW = im2bw(X, map, level)"],
              "Binary image that is true where the brightness is above level, a fraction of the class range "
              "(default 0.5). RGB and indexed images are converted to grayscale first."),
    "imbinarize": (["BW = imbinarize(I)", "BW = imbinarize(I, T)", "BW = imbinarize(I, 'adaptive')",
                    "BW = imbinarize(I, 'adaptive', 'Sensitivity', s, 'ForegroundPolarity', 'dark')"],
                   "Binary image from a grayscale image. With no threshold Otsu's method picks one for the "
                   "whole image (see graythresh). T is a threshold between 0 and 1, or a matrix of them. "
                   "'adaptive' compares each pixel with the mean of its neighbourhood (about one eighth of "
                   "the image size), which copes with uneven lighting: a pixel is false when it is clearly "
                   "darker than its surroundings (15% at the default 'Sensitivity' of 0.5), so evenly lit "
                   "areas come out true. Its results are close to, but not identical with, other "
                   "implementations."),
    "graythresh": (["level = graythresh(I)", "[level, em] = graythresh(I)"],
                   "Threshold between 0 and 1 chosen by Otsu's method, which splits a 256-bin histogram into "
                   "two groups with the smallest spread inside the groups. em (0..1) says how well the two "
                   "groups separate."),
    "otsuthresh": (["level = otsuthresh(counts)", "[level, em] = otsuthresh(counts)"],
                   "Otsu's threshold computed from histogram counts; level is the chosen bin as a fraction "
                   "of the bin range."),
    "imhist": (["imhist(I)", "[counts, binLocations] = imhist(I)", "[counts, x] = imhist(I, n)",
                "[counts, x] = imhist(X, map)"],
               "Histogram of a grayscale image with n bins (256 by default, 2 for a binary image) spread "
               "over the range of the image's class. With outputs it returns the counts and bin centres as "
               "columns; with none it draws the histogram as a bar chart. For an indexed image the counts "
               "are per colormap entry."),
    "imadjust": (["J = imadjust(I)", "J = imadjust(I, [low_in high_in])",
                  "J = imadjust(I, [low_in high_in], [low_out high_out])", "J = imadjust(I, in, out, gamma)"],
                 "Stretches contrast: values from low_in to high_in are mapped to low_out..high_out (0..1 "
                 "by default), with values outside clipped. Limits are fractions of the class range; [] "
                 "means [0 1]. With no limits, the darkest and brightest 1% of pixels are saturated. gamma "
                 "bends the mapping (below 1 brightens). For RGB images give limits, optionally one column "
                 "per channel."),
    "stretchlim": (["lim = stretchlim(I)", "lim = stretchlim(I, tol)"],
                   "Limits [low; high] (0..1) that leave the fraction tol of pixels saturated at each end "
                   "(1% by default); one column per colour channel. Meant for imadjust."),
    "histeq": (["J = histeq(I)", "J = histeq(I, n)", "J = histeq(I, hgram)", "[J, T] = histeq(...)"],
               "Histogram equalization: remaps gray levels so that the output uses n levels (64 by default) "
               "about equally often, or so that its histogram resembles hgram. T is the mapping applied to "
               "the 256 input levels. Results can differ slightly from other implementations in how ties "
               "between output levels are resolved."),
    "entropy": (["e = entropy(I)"],
                "Entropy of a grayscale image in bits, -sum(p .* log2(p)) over its 256-bin histogram."),
    # -- arithmetic and comparison ---------------------------------------------------------------
    "imadd": (["Z = imadd(X, Y)"],
              "Adds two images, or a constant to an image, in the class of the image: integer results are "
              "rounded and clipped to the class range instead of wrapping."),
    "imsubtract": (["Z = imsubtract(X, Y)"],
                   "Subtracts images (or a constant) in the class of the image; unsigned results below 0 "
                   "become 0."),
    "immultiply": (["Z = immultiply(X, Y)"], "Multiplies images element by element, saturating integer classes."),
    "imdivide": (["Z = imdivide(X, Y)"], "Divides images element by element, rounding and saturating integer classes."),
    "imabsdiff": (["Z = imabsdiff(X, Y)"],
                  "Absolute difference of two images in their own class, without the underflow that X - Y "
                  "would give for unsigned integers."),
    "immse": (["err = immse(X, Y)"], "Mean squared error between two arrays of the same size, computed in double."),
    "psnr": (["p = psnr(A, ref)", "p = psnr(A, ref, peakval)", "[p, snr] = psnr(...)"],
             "Peak signal-to-noise ratio in decibels, 10*log10(peak^2 / MSE). peak defaults to the range of "
             "the image class (255 for uint8, 1 for double). snr is the signal-to-noise ratio."),
    "ssim": (["s = ssim(A, ref)", "[s, map] = ssim(A, ref)", "s = ssim(..., 'DynamicRange', r)"],
             "Structural similarity index of two grayscale images (1 means identical), using a Gaussian "
             "window with standard deviation 1.5 and the usual constants (0.01 and 0.03 times the dynamic "
             "range, squared). map holds the value at every pixel."),
    "mean2": (["m = mean2(A)"], "Mean of all elements of an array, as double."),
    "std2": (["s = std2(A)"], "Standard deviation of all elements of an array (normalised by n-1)."),
    "corr2": (["r = corr2(A, B)"], "Correlation coefficient between two arrays of the same size."),
    "integralImage": (["J = integralImage(I)"],
                      "Summed-area table: J(r+1, c+1) is the sum of I(1:r, 1:c). The result is one row and "
                      "one column larger than I, with zeros in its first row and column."),
    # -- filtering -------------------------------------------------------------------------------
    "imfilter": (["B = imfilter(A, h)", "B = imfilter(A, h, padding)", "B = imfilter(..., 'conv')",
                  "B = imfilter(..., 'full')"],
                 "Filters an image with the kernel h. By default this is correlation, the result has the "
                 "size of A and pixels outside the image count as 0. padding may be a number, 'replicate' "
                 "(repeat the border), 'symmetric' (mirror) or 'circular' (wrap around). 'conv' rotates the "
                 "kernel by 180 degrees, 'full' returns the larger full result. The output has the class of "
                 "A, so integer images are rounded and clipped; convert with im2double to keep fractions "
                 "and negative values. A 2-D kernel is applied to each colour channel."),
    "filter2": (["Y = filter2(h, X)", "Y = filter2(h, X, shape)"],
                "Two-dimensional correlation of X with the kernel h in double precision. shape is 'same' "
                "(default), 'full' or 'valid'."),
    "fspecial": (["h = fspecial('average', hsize)", "h = fspecial('gaussian', hsize, sigma)",
                  "h = fspecial('disk', radius)", "h = fspecial('laplacian', alpha)",
                  "h = fspecial('log', hsize, sigma)", "h = fspecial('motion', len, theta)",
                  "h = fspecial('sobel')", "h = fspecial('prewitt')", "h = fspecial('unsharp', alpha)"],
                 "Ready-made filter kernels for imfilter. 'average' is a box of equal weights (3-by-3 by "
                 "default); 'gaussian' a bell of size hsize (3) and standard deviation sigma (0.5); 'disk' "
                 "a circular average of the given radius (5) with partly covered pixels weighted by area; "
                 "'laplacian' the 3-by-3 Laplacian with shape alpha (0.2); 'log' a Laplacian of Gaussian "
                 "(5, 0.5); 'motion' a line of length len (9) at theta degrees (0) that imitates camera "
                 "motion; 'sobel' and 'prewitt' emphasise horizontal edges (transpose for vertical); "
                 "'unsharp' sharpens. The 'motion' weights are an approximation and can differ in the last "
                 "digits from other implementations."),
    "imgaussfilt": (["B = imgaussfilt(A)", "B = imgaussfilt(A, sigma)", "B = imgaussfilt(..., 'FilterSize', n)",
                     "B = imgaussfilt(..., 'Padding', p)"],
                    "Gaussian blur with standard deviation sigma (0.5 by default; [sy sx] for different "
                    "amounts down and across). The kernel is 2*ceil(2*sigma)+1 wide unless 'FilterSize' is "
                    "given, and the border is repeated unless 'Padding' says 'symmetric', 'circular' or a "
                    "number."),
    "medfilt2": (["B = medfilt2(A)", "B = medfilt2(A, [m n])", "B = medfilt2(..., 'symmetric')"],
                 "Median filter: each pixel becomes the median of its m-by-n neighbourhood (3-by-3 by "
                 "default), which removes salt-and-pepper noise. Outside the image is 0 unless 'symmetric' "
                 "mirrors the border. Colour images are filtered one channel at a time."),
    "ordfilt2": (["B = ordfilt2(A, order, domain)", "B = ordfilt2(A, order, domain, 'symmetric')"],
                 "Order-statistic filter: each pixel becomes the order-th smallest value among the "
                 "neighbours marked by the nonzero elements of domain. order 1 is a minimum filter."),
    "wiener2": (["J = wiener2(I)", "J = wiener2(I, [m n])", "J = wiener2(I, [m n], noise)",
                 "[J, noise] = wiener2(...)"],
                "Adaptive noise removal: smooths strongly where the local variance is close to the noise "
                "power and little near edges. The noise power is estimated as the average local variance "
                "when not given."),
    "imnoise": (["J = imnoise(I, 'gaussian', mean, variance)", "J = imnoise(I, 'salt & pepper', density)",
                 "J = imnoise(I, 'speckle', variance)", "J = imnoise(I, 'poisson')"],
                "Adds noise to an image. Mean and variance refer to a 0..1 intensity scale (defaults 0 and "
                "0.01); density is the fraction of pixels set to black or white (0.05); speckle noise is "
                "multiplicative (0.05). Uses the generator controlled by rng."),
    "padarray": (["B = padarray(A, padsize)", "B = padarray(A, padsize, padval)",
                  "B = padarray(A, padsize, method)", "B = padarray(..., direction)"],
                 "Enlarges an array by padsize(k) elements along dimension k. The new elements are 0, padval, "
                 "or come from the array itself: 'replicate' repeats the border, 'symmetric' mirrors it and "
                 "'circular' wraps around. direction is 'both' (default), 'pre' or 'post'."),
    "im2col": (["B = im2col(A, [m n])", "B = im2col(A, [m n], 'distinct')"],
               "Rearranges m-by-n blocks of a matrix into columns. 'sliding' (default) takes every block "
               "position; 'distinct' takes non-overlapping blocks, padding the matrix with zeros."),
    "edge": (["BW = edge(I)", "BW = edge(I, method)", "BW = edge(I, method, threshold)",
              "BW = edge(I, method, threshold, direction)", "BW = edge(I, 'canny', threshold, sigma)",
              "[BW, threshold] = edge(...)"],
             "Finds edges in a grayscale image and returns them as a binary image. Methods: 'sobel' "
             "(default), 'prewitt' and 'roberts' threshold the gradient magnitude and thin the result "
             "('nothinning' skips that; direction 'horizontal' or 'vertical' limits the search); 'log' marks "
             "zero crossings after a Laplacian-of-Gaussian filter (sigma 2); 'canny' smooths (sigma "
             "sqrt(2)), keeps ridges of the gradient and links weak edges to strong ones, with thresholds "
             "[low high]. Thresholds are chosen automatically when omitted. Edge maps agree closely with "
             "other implementations but individual pixels may differ."),
    # -- morphology ------------------------------------------------------------------------------
    "strel": (["se = strel('disk', r)", "se = strel('square', w)", "se = strel('rectangle', [m n])",
               "se = strel('line', len, deg)", "se = strel('diamond', r)", "se = strel('octagon', r)",
               "se = strel(nhood)"],
              "Structuring element for imdilate, imerode, imopen and imclose. In mylab it is simply a "
              "logical matrix that marks the neighbourhood, so it can be displayed, indexed and combined "
              "like any matrix; there is no strel object and no .Neighborhood property (use getnhood, or the "
              "value itself). 'disk' is the exact disk of pixels within distance r of the centre, which "
              "for larger radii is not the same as the polygon approximation used elsewhere."),
    "getnhood": (["nhood = getnhood(se)"], "Neighbourhood of a structuring element as a logical matrix."),
    "imdilate": (["B = imdilate(A, se)"],
                 "Dilation: each pixel becomes the largest value under the structuring element, which grows "
                 "bright regions (and true regions of a binary image). " + _SE),
    "imerode": (["B = imerode(A, se)"],
                "Erosion: each pixel becomes the smallest value under the structuring element, which "
                "shrinks bright regions. " + _SE),
    "imopen": (["B = imopen(A, se)"],
               "Opening, an erosion followed by a dilation: removes bright details smaller than the "
               "structuring element while keeping larger shapes."),
    "imclose": (["B = imclose(A, se)"],
                "Closing, a dilation followed by an erosion: fills dark gaps and holes smaller than the "
                "structuring element."),
    "imtophat": (["B = imtophat(A, se)"], "The image minus its opening: keeps small bright details."),
    "imbothat": (["B = imbothat(A, se)"], "The closing minus the image: keeps small dark details."),
    "bwlabel": (["L = bwlabel(BW)", "L = bwlabel(BW, conn)", "[L, n] = bwlabel(...)"],
                "Labels the connected groups of true pixels in a binary image with 1, 2, ..., n (0 is "
                "background). conn is 8 (default; diagonal neighbours touch) or 4. Groups are numbered in "
                "the order their first pixel appears going down the columns."),
    "bwconncomp": (["CC = bwconncomp(BW)", "CC = bwconncomp(BW, conn)"],
                   "Connected components of a 2-D binary image as a struct with fields Connectivity, "
                   "ImageSize, NumObjects and PixelIdxList (a cell with the linear indices of each "
                   "component)."),
    "labelmatrix": (["L = labelmatrix(CC)"], "Label matrix built from the struct returned by bwconncomp."),
    "bwareaopen": (["BW2 = bwareaopen(BW, p)", "BW2 = bwareaopen(BW, p, conn)"],
                   "Removes connected groups with fewer than p pixels from a binary image."),
    "bwperim": (["P = bwperim(BW)", "P = bwperim(BW, conn)"],
                "Outline of the objects in a binary image: true pixels that touch a false pixel or the "
                "image border (4-connected by default)."),
    "imfill": (["BW2 = imfill(BW, 'holes')", "BW2 = imfill(BW, locations)"],
               "Fills holes in a binary image (background regions that cannot be reached from the border), "
               "or flood-fills the background starting from the given linear indices or [row col] rows. "
               "Grayscale images and interactive use are not supported."),
    "bwdist": (["D = bwdist(BW)", "[D, idx] = bwdist(BW)", "D = bwdist(BW, method)"],
               "Distance from every pixel to the nearest true pixel, as single. method is 'euclidean' "
               "(default), 'cityblock', 'chessboard' or 'quasi-euclidean'. idx is the linear index of that "
               "nearest pixel."),
    "bwboundaries": (["B = bwboundaries(BW)", "[B, L, n] = bwboundaries(BW)"],
                     "Traces the outer boundary of each object in a binary image (8-connected). B is a cell "
                     "array with one k-by-2 matrix of [row column] points per object, going clockwise and "
                     "ending at its starting point; L is the label matrix and n the number of objects. "
                     "Boundaries of holes inside objects are not traced, with or without 'noholes'."),
    "poly2mask": (["BW = poly2mask(x, y, m, n)"],
                  "m-by-n binary mask that is true for the pixels whose centres lie inside the polygon with "
                  "corner coordinates x (columns) and y (rows). Pixels exactly on the edge may be classified "
                  "differently from other implementations."),
    "bwmorph": (["BW2 = bwmorph(BW, operation)", "BW2 = bwmorph(BW, operation, n)"],
                "Applies a binary morphological operation n times (once by default; Inf repeats until "
                "nothing changes). Operations: 'thin' and 'skel' (reduce shapes to one-pixel-wide lines, "
                "both by Zhang-Suen thinning), 'remove' (keep outlines), 'clean' (drop isolated pixels), "
                "'fill' (fill isolated holes), 'majority', 'bridge', 'diag', 'spur', 'endpoints', "
                "'branchpoints', 'dilate', 'erode', 'open', 'close', 'perim4' and 'perim8'. Thinned shapes "
                "can differ by a few pixels from other implementations."),
    "regionprops": (["stats = regionprops(BW, properties)", "stats = regionprops(L, properties)",
                     "stats = regionprops(CC, properties)", "stats = regionprops(L, I, properties)"],
                    "Measures the regions of a binary image (8-connected groups), a label matrix or a "
                    "bwconncomp struct and returns a struct array with one element per region. Properties: "
                    "Area, Centroid ([x y]), BoundingBox ([left top width height], measured from pixel "
                    "edges), MajorAxisLength, MinorAxisLength, Eccentricity, Orientation (of the ellipse "
                    "with the same second moments), EquivDiameter, Extent, Perimeter, Circularity, Image, "
                    "FilledImage, FilledArea, EulerNumber, PixelIdxList, PixelList and SubarrayIdx; with a "
                    "grayscale image I also MeanIntensity, MinIntensity, MaxIntensity, PixelValues and "
                    "WeightedCentroid. The default is 'basic' (Area, Centroid, BoundingBox); 'all' gives "
                    "everything listed. Perimeter is estimated from the chain of boundary pixels (with the "
                    "usual correction for diagonal steps and corners) and can differ a little from other "
                    "implementations. Convex-hull properties and table output are not "
                    "supported."),
    # -- geometry --------------------------------------------------------------------------------
    "imresize": (["B = imresize(A, scale)", "B = imresize(A, [rows cols])", "B = imresize(..., method)",
                  "B = imresize(..., 'Antialiasing', tf)"],
                 "Resizes an image by a scale factor or to a given size; one of rows and cols may be NaN to "
                 "keep the aspect ratio. method is 'bicubic' (default), 'bilinear', 'nearest', 'box', "
                 "'lanczos2' or 'lanczos3'. When shrinking, the kernel is widened so that fine detail is "
                 "averaged rather than aliased (not for 'nearest'; switch off with 'Antialiasing', false). "
                 "The result has the class of A. Resizing an indexed image together with its colormap is "
                 "not supported. " + _CLASSES),
    "imrotate": (["B = imrotate(A, angle)", "B = imrotate(A, angle, method)", "B = imrotate(A, angle, method, 'crop')"],
                 "Rotates an image counter-clockwise by angle degrees about its centre. method is 'nearest' "
                 "(default), 'bilinear' or 'bicubic'. The result is large enough to hold the whole rotated "
                 "image ('loose') unless 'crop' keeps the original size; uncovered corners are 0. Multiples "
                 "of 90 degrees are exact. 'bicubic' uses a cubic spline, so values differ slightly from "
                 "other implementations."),
    "imcrop": (["J = imcrop(I, [xmin ymin width height])", "[J, rect] = imcrop(I, rect)"],
               "Cuts a rectangle out of an image. The rectangle is in pixel coordinates, so with whole "
               "numbers the result has height+1 rows and width+1 columns; a BoundingBox from regionprops "
               "gives exactly the region. Choosing the rectangle with the mouse is not supported."),
    "imtranslate": (["B = imtranslate(A, [tx ty])", "B = imtranslate(A, [tx ty], method)",
                     "B = imtranslate(..., 'FillValues', v)"],
                    "Shifts an image tx pixels to the right and ty pixels down, keeping its size. Uncovered "
                    "pixels are 0 or the fill value; fractional shifts are interpolated ('linear' by "
                    "default, or 'nearest' or 'cubic')."),
}
