"""Help text for mylab/builtins/vision_detect.py (written from scratch for mylab)."""

_BOX = "Boxes are M-by-4 [x y width height] with (x, y) the top-left pixel (1-based)."
_ONNX = ("mylab ships no pretrained networks and cannot download MATLAB's; export a detector to ONNX yourself and pass "
         "the file. Options: 'InputSize' [h w] of the network (default [640 640]), 'OutputFormat' ('auto', 'yolov5' "
         "rows of [cx cy w h objectness classScores...], 'yolov8' [cx cy w h classScores...] in either orientation, "
         "or 'yolox' raw grid outputs with strides 8, 16 and 32), 'Normalize' (scale pixels to 0..1) and "
         "'PreprocessMode' ('resize' or 'letterbox' with grey padding). Run it with "
         "[bboxes, scores, labels] = detect(detector, I, 'Threshold', t, 'SelectStrongest', tf, 'MinSize', sz, "
         "'MaxSize', sz): scores are class probability times objectness, labels a categorical array of the class "
         "names, and boxes are scaled back to the image. The network runs through OpenCV's dnn module, which "
         "supports most ONNX operators used by YOLO exports. Needs OpenCV.")

HELP = {
    "bboxOverlapRatio": (["ratio = bboxOverlapRatio(bboxA, bboxB)", "ratio = bboxOverlapRatio(bboxA, bboxB, 'Min')"],
                         "M-by-N matrix of overlap ratios between the boxes of bboxA (M-by-4) and bboxB (N-by-4): "
                         "intersection over union (default) or intersection over the smaller area ('Min'). " + _BOX),
    "selectStrongestBbox": (["[selectedBbox, selectedScore, index] = selectStrongestBbox(bbox, score)",
                             "[...] = selectStrongestBbox(..., 'RatioType', 'Union'|'Min', 'OverlapThreshold', t, "
                             "'NumStrongest', n)"],
                            "Non-maximum suppression: keeps the highest-scoring box and removes boxes overlapping it "
                            "by more than OverlapThreshold (default 0.5), repeatedly. index gives the kept rows in "
                            "their original order. " + _BOX),
    "selectStrongestBboxMulticlass": (["[selectedBbox, selectedScore, selectedLabel, index] = "
                                       "selectStrongestBboxMulticlass(bbox, score, label, ...)"],
                                      "Non-maximum suppression within each class: boxes of different labels "
                                      "(categorical, numeric or text) never suppress each other. Options as for "
                                      "selectStrongestBbox."),
    "vision.CascadeObjectDetector": (["detector = vision.CascadeObjectDetector",
                                      "detector = vision.CascadeObjectDetector(model)",
                                      "detector = vision.CascadeObjectDetector(xmlFile, 'MinSize', [h w], 'MaxSize', "
                                      "[h w], 'ScaleFactor', s, 'MergeThreshold', n, 'UseROI', tf)",
                                      "bboxes = detector(I)", "bboxes = step(detector, I, roi)"],
                                     "Viola-Jones detector using OpenCV's cascade classifier. model may be the path "
                                     "of an OpenCV cascade XML file or one of these names, which use the "
                                     "BSD-licensed cascades from opencv-python 4.x (not the MATLAB models, so results "
                                     "differ): 'FrontalFaceCART' (haarcascade_frontalface_alt2), 'FrontalFaceLBP' "
                                     "(haarcascade_frontalface_default; no LBP model is bundled), 'ProfileFace', "
                                     "'UpperBody', 'LeftEye'/'LeftEyeCART' and 'RightEye'/'RightEyeCART' (the "
                                     "2splits eye models), and 'EyePairBig'/'EyePairSmall' (single eyes from "
                                     "haarcascade_eye joined into pair boxes). 'Mouth' and 'Nose' are not available. "
                                     "OpenCV 5 wheels no longer include the XML files: set MYLAB_HAARCASCADES to a "
                                     "folder holding them. ScaleFactor defaults to 1.1 and MergeThreshold (the "
                                     "neighbours a detection needs) to 4. " + _BOX + " Needs OpenCV."),
    "vision.PeopleDetector": (["detector = vision.PeopleDetector", "detector = vision.PeopleDetector(model, ...)",
                               "[bboxes, scores] = detector(I)"],
                              "Finds upright people with a HOG descriptor and linear SVM. The models are OpenCV's "
                              "built-in SVMs (Dalal-Triggs for 'UprightPeople_128x64', the default, and Daimler for "
                              "'UprightPeople_96x48'), not MATLAB's, so scores differ. Properties: "
                              "ClassificationThreshold (default 1; higher finds fewer people), MinSize, MaxSize, "
                              "ScaleFactor (default 1.05), WindowStride (default [8 8]), MergeDetections (default "
                              "true) and UseROI. " + _BOX + " Needs OpenCV."),
    "peopleDetector": (["detector = peopleDetector", "[bboxes, scores] = detect(detector, I, 'Threshold', t)"],
                       "A HOG people detector with the same OpenCV model and properties as vision.PeopleDetector. "
                       "MATLAB's peopleDetector uses a deep network; this one does not."),
    "yolov4ObjectDetector": (["detector = yolov4ObjectDetector(onnxFile, classNames, ...)",
                              "[bboxes, scores, labels] = detect(detector, I, ...)"],
                             "A YOLO-style detector from a user-supplied ONNX file. The default OutputFormat is "
                             "'auto' with plain resizing and 0..1 pixel scaling. " + _ONNX),
    "yoloxObjectDetector": (["detector = yoloxObjectDetector(onnxFile, classNames, ...)",
                             "[bboxes, scores, labels] = detect(detector, I, ...)"],
                            "A YOLOX detector from a user-supplied ONNX file: grid decoding, letterbox input and no "
                            "pixel scaling by default. " + _ONNX),
    "yolov8ObjectDetector": (["detector = yolov8ObjectDetector(onnxFile, classNames, ...)",
                              "[bboxes, scores, labels] = detect(detector, I, ...)"],
                             "A YOLOv8-style detector (Ultralytics export) from a user-supplied ONNX file, with "
                             "letterbox input. " + _ONNX),
    "bboxPrecisionRecall": (["[precision, recall] = bboxPrecisionRecall(bboxes, groundTruthBboxes)",
                             "[precision, recall] = bboxPrecisionRecall(bboxes, groundTruthBboxes, threshold)"],
                            "Precision and recall of detected boxes against ground truth at an overlap (IoU) "
                            "threshold (default 0.5). Each ground-truth box can be matched once. Cell arrays give "
                            "one set of boxes per image."),
    "evaluateDetectionPrecision": (["[averagePrecision, recall, precision] = evaluateDetectionPrecision("
                                    "detectionResults, groundTruthData)",
                                    "[...] = evaluateDetectionPrecision(detectionResults, groundTruthData, threshold)"],
                                   "Average precision of a detector. detectionResults is a table with Boxes and "
                                   "Scores variables (and Labels for several classes), one row per image; "
                                   "groundTruthData is a table with one variable of boxes per class. Detections are "
                                   "matched greedily by score to ground truth with IoU >= threshold (default 0.5); "
                                   "AP is the area under the precision envelope. For several classes the outputs are "
                                   "per-class (cell arrays for recall and precision)."),
    "evaluateDetectionMissRate": (["[logAverageMissRate, fppi, missRate] = evaluateDetectionMissRate("
                                   "detectionResults, groundTruthData, threshold)"],
                                  "Miss rate against false positives per image for a single-class detector, and the "
                                  "log-average miss rate over nine FPPI values from 0.01 to 1."),
}
