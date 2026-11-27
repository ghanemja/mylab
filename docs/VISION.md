# Computer vision

mylab runs the common computer vision workflows of MATLAB-language code:
feature detection and matching, robust geometric fitting, camera calibration,
stereo, object detection, tracking and point clouds. The names, call forms
and results follow MATLAB's Computer Vision Toolbox, but the implementation is
mylab's own, built on [OpenCV](https://opencv.org), NumPy and SciPy. Results
are close to MATLAB's, not identical: detectors find similar but not the same
points, and robust estimators draw different random samples.

## Install

OpenCV is optional. Install it together with mylab:

```
pip install "mylab-lang[vision]"
```

This installs `opencv-python-headless`. Without OpenCV the point cloud
functions, the Kalman filter, `detectSURFFeatures`, the box utilities and the
geometry functions that need no image processing (`estgeotform2d`,
`estimateFundamentalMatrix`, `estrelpose`, `triangulate`, ...) still work; the
others stop with a message that shows the install command.

In the web app nothing has to be installed. Pyodide's OpenCV package (4.10) is
downloaded the first time a program calls a function that needs it. You see
the note "Loading computer vision (OpenCV)…" while that happens.

## A first example

```matlab
% two views of a synthetic scene, the second shifted and rotated
rng(1); I = zeros(240, 320);
for k = 1:40
    x = randi([20 290]); y = randi([20 210]);
    I(y:y+randi([5 20]), x:x+randi([5 20])) = rand;
end
I = imgaussfilt(I, 1);
J = imwarp(I, rigidtform2d(8, [12 -5]), 'OutputView', imref2d(size(I)));

p1 = detectSIFTFeatures(I);   p2 = detectSIFTFeatures(J);
[f1, v1] = extractFeatures(I, p1);
[f2, v2] = extractFeatures(J, p2);
pairs = matchFeatures(f1, f2);
m1 = v1(pairs(:, 1));  m2 = v2(pairs(:, 2));

[tform, inliers] = estgeotform2d(m1, m2, 'rigid');
tform.RotationAngle      % about 8
tform.Translation        % about [12 -5]
showMatchedFeatures(I, J, m1(inliers), m2(inliers), 'montage')
```

## What works

### Features
`detectHarrisFeatures`, `detectMinEigenFeatures`, `detectFASTFeatures`,
`detectORBFeatures`, `detectSIFTFeatures`, `detectBRISKFeatures`,
`detectKAZEFeatures`, `detectMSERFeatures` and `detectSURFFeatures`, with the
point objects `cornerPoints`, `ORBPoints`, `SIFTPoints`, `SURFPoints`,
`BRISKPoints`, `KAZEPoints` and `MSERRegions`. Point objects have `Location`,
`Metric`, `Count` and (for the scale-aware ones) `Scale` and `Orientation`;
they can be indexed (`p(1:10)`, `p(end)`), concatenated (`[p1; p2]`) and
passed to `selectStrongest`, `selectUniform` and `plot`.

`extractFeatures` computes SIFT, SURF, KAZE (numeric) or ORB, BRISK and FREAK
(`binaryFeatures`) descriptors, or raw blocks. `matchFeatures` matches them
with the SSD, SAD or Hamming distance, the ratio test, a match threshold and
an optional one-to-one constraint. `showMatchedFeatures` draws the matches.

### Geometry
`estgeotform2d` (rigid, similarity, affine and projective; MSAC with
`MaxNumTrials`, `Confidence`, `MaxDistance` and a status output) and the older
`estimateGeometricTransform2D` and `estimateGeometricTransform`. The results
are the transform objects that `imwarp` and `transformPointsForward` take.
`estimateFundamentalMatrix` (RANSAC, MSAC, LMedS, LTS, normalized 8-point),
`estimateEssentialMatrix`, `estrelpose` and `relativeCameraPose`,
`triangulate`, `epipolarLine`, `isEpipoleInImage`. 3-D transforms:
`rigidtform3d`, `simtform3d`, `affinetform3d`, `transltform3d`, `rigid3d`,
`affine3d`, and the rotation-vector conversions.

### Drawing into images
`insertShape` (rectangles, circles, lines, polygons, filled with opacity),
`insertMarker`, `insertText` and `insertObjectAnnotation`.

### Detection
- `vision.CascadeObjectDetector` runs OpenCV cascade classifiers. Give it the
  path of a cascade XML file, or a model name. Model names map to the
  BSD-licensed cascades that opencv-python 4.x ships (MATLAB's own models are
  not available, so results differ):

  | Model name | OpenCV cascade |
  | --- | --- |
  | `FrontalFaceCART` (default) | `haarcascade_frontalface_alt2.xml` (tree-based Haar) |
  | `FrontalFaceLBP` | `haarcascade_frontalface_default.xml` (opencv-python has no LBP face model) |
  | `ProfileFace` | `haarcascade_profileface.xml` |
  | `UpperBody` | `haarcascade_upperbody.xml` |
  | `LeftEye`, `LeftEyeCART` | `haarcascade_lefteye_2splits.xml` |
  | `RightEye`, `RightEyeCART` | `haarcascade_righteye_2splits.xml` |
  | `EyePairBig`, `EyePairSmall` | single eyes from `haarcascade_eye.xml`, joined into one box per pair |
  | `Mouth`, `Nose` | not available |

  OpenCV 5 wheels no longer include these files. If they are not found, set
  the environment variable `MYLAB_HAARCASCADES` to a folder that holds them
  (for example the `cv2/data` folder of an opencv-python 4.x installation).
  The web app's OpenCV includes them.
- `vision.PeopleDetector` and `peopleDetector` find upright people with HOG
  features and OpenCV's built-in linear SVMs (the Dalal-Triggs 64x128 model
  and the Daimler 48x96 model). MATLAB's `peopleDetector` uses a deep network;
  mylab's does not.
- `yolov4ObjectDetector`, `yoloxObjectDetector` and `yolov8ObjectDetector`
  run a YOLO-style network from an ONNX file you supply, through OpenCV's dnn
  module, and `detect(detector, I)` returns boxes, scores and categorical
  labels. **mylab ships no network weights and cannot download MATLAB's
  pretrained detectors**: export a model to ONNX yourself (for example with
  the Ultralytics or YOLOX tools) and pass its file name and class names.
- `selectStrongestBbox`, `selectStrongestBboxMulticlass` (non-maximum
  suppression), `bboxOverlapRatio`, `bboxPrecisionRecall`,
  `evaluateDetectionPrecision` and `evaluateDetectionMissRate`.

### Cameras and stereo
`cameraIntrinsics`, `cameraParameters`, `stereoParameters`,
`generateCheckerboardPoints`, `detectCheckerboardPoints` (which also finds the
board size), `estimateCameraParameters` (single cameras and stereo pairs),
`undistortImage`, `undistortPoints`, `world2img`, `worldToImage`,
`img2world2d`, `pointsToWorld`, `cameraProjection`, `cameraMatrix`,
`estimateExtrinsics`, `extrinsics`, `estworldpose`, `rectifyStereoImages`,
`disparitySGM`, `disparityBM` and `reconstructScene`.

### Tracking and motion
`vision.PointTracker` (pyramidal KLT with the forward-backward check),
`opticalFlowLK`, `opticalFlowHS`, `opticalFlowFarneback` with `estimateFlow`
and `reset`, the `opticalFlow` object, `vision.KalmanFilter` and
`configureKalmanFilter`. System objects can be called like functions
(`tracker(frame)`) or with `step`.

### Point clouds
`pointCloud` (organized or not, with colours, normals and intensity, and the
methods `findNearestNeighbors`, `findNeighborsInRadius`, `findPointsInROI`,
`select`, `removeInvalidPoints`, `copy`), `pcread` and `pcwrite` (PLY ASCII
and binary, PCD ASCII and binary), `pcshow`, `pcshowpair`, `pcdownsample`,
`pcdenoise`, `pcnormals`, `pcfitplane` with `planeModel`, `pcregistericp`,
`pcmerge`, `pctransform` and `pcsegdist`.

## Conventions

- Points are M-by-2 `[x y]` in MATLAB pixel coordinates: the centre of the
  top-left pixel is (1, 1). Boxes are `[x y width height]` with (x, y) the
  top-left pixel. mylab converts to and from OpenCV's 0-based coordinates.
- Intrinsic matrices are `K = [fx s cx; 0 fy cy; 0 0 1]` in those
  coordinates; `IntrinsicMatrix` is its transpose. Newer functions use
  `rigidtform3d` extrinsics (world to camera); older ones use rotation matrices
  and translation rows in the row-vector convention.
- Random sampling (RANSAC, MSAC, `pcdownsample(..., 'random', ...)`,
  `pcfitplane`) uses the global generator, so `rng(seed)` makes it repeatable.

## Differences from MATLAB

- `detectSURFFeatures` and SURF descriptors are computed by mylab itself
  (determinant of the Hessian with Gaussian derivatives), because OpenCV's
  SURF is in its non-free module. Points and descriptors are SURF-like but do
  not match other SURF implementations value for value.
- `extractFeatures` with `'Method', 'Auto'` uses BRISK descriptors for corner
  points (MATLAB uses FREAK, which needs OpenCV's extra modules here).
- `estimateCameraParameters` does not estimate skew.
- `detectCheckerboardPoints` numbers the corners starting nearest the image's
  top-left corner; a board seen upside down is numbered from its other end.
- `insertText` uses OpenCV's Hershey font; the `'Font'` option is ignored.
- Every estimator draws its own random samples, so inlier sets near the
  threshold can differ from MATLAB's.

## Not supported

- Training: `trainCascadeObjectDetector` (OpenCV's cascade trainer is a
  separate program not shipped with opencv-python), `trainYOLOv4ObjectDetector`
  and the other deep-learning training functions.
- `ocr` (it needs a text recognition engine with its own models), the
  pretrained deep-learning detectors and segmenters, `bagOfFeatures`,
  `imageviewset`, bundle adjustment, visual SLAM, `pcregisterndt`,
  `pcregistercpd`, Lidar and fisheye camera models, and the `vision.*`
  video I/O and blob-analysis System objects.
