"""Help text for mylab/builtins/vision_tracking.py (written from scratch for mylab)."""

_FLOW = ("Call flow = estimateFlow(obj, frame) once per grayscale frame: the first call returns zero flow, later "
         "calls the motion from the previous frame, as an opticalFlow object (Vx, Vy in pixels per frame, "
         "Orientation, Magnitude). reset(obj) forgets the previous frame.")

HELP = {
    "vision.PointTracker": (["tracker = vision.PointTracker", "tracker = vision.PointTracker('NumPyramidLevels', n, "
                             "'MaxBidirectionalError', e, 'BlockSize', [h w], 'MaxIterations', k)",
                             "initialize(tracker, points, I)", "[points, validity, scores] = tracker(J)",
                             "setPoints(tracker, points)", "release(tracker)"],
                            "Tracks points from frame to frame with the pyramidal Kanade-Lucas-Tomasi method "
                            "(OpenCV). initialize takes M-by-2 [x y] points and the first frame. Each call with a new "
                            "frame returns the new locations, a logical validity per point (lost points stay "
                            "invalid until setPoints) and a 0..1 score. With a finite MaxBidirectionalError (default "
                            "Inf) each point is also tracked backwards and dropped if it does not return within that "
                            "many pixels. Needs OpenCV."),
    "opticalFlow": (["flow = opticalFlow(Vx, Vy)", "plot(flow, 'DecimationFactor', [dy dx], 'ScaleFactor', s)"],
                    "A dense motion field with properties Vx, Vy, Orientation (radians) and Magnitude. plot draws "
                    "arrows on the current axes."),
    "opticalFlowLK": (["obj = opticalFlowLK", "obj = opticalFlowLK('NoiseThreshold', t)",
                       "flow = estimateFlow(obj, I)"],
                      "Dense Lucas-Kanade optical flow (5-by-5 windows on lightly smoothed frames, intensities "
                      "taken in 0..255 units); pixels whose smaller structure-tensor eigenvalue is below "
                      "NoiseThreshold (default 0.0039) get zero flow. " + _FLOW),
    "opticalFlowHS": (["obj = opticalFlowHS('Smoothness', a, 'MaxIteration', n, 'VelocityDifference', d)",
                       "flow = estimateFlow(obj, I)"],
                      "Horn-Schunck optical flow: a smooth flow field from MaxIteration (default 10) Jacobi "
                      "iterations with smoothness weight Smoothness (default 1), stopping early when no velocity "
                      "changes by more than VelocityDifference. " + _FLOW),
    "opticalFlowFarneback": (["obj = opticalFlowFarneback('NumPyramidLevels', n, 'PyramidScale', s, "
                              "'NumIterations', k, 'NeighborhoodSize', m, 'FilterSize', f)",
                              "flow = estimateFlow(obj, I)"],
                             "Farneback polynomial-expansion optical flow through OpenCV (defaults 3, 0.5, 3, 5, "
                             "15). " + _FLOW + " Needs OpenCV."),
    "vision.KalmanFilter": (["kf = vision.KalmanFilter(StateTransitionModel, MeasurementModel)",
                             "kf = vision.KalmanFilter(A, H, B, 'State', x, 'StateCovariance', P, 'ProcessNoise', Q, "
                             "'MeasurementNoise', R)",
                             "[zPred, xPred, PPred] = predict(kf)", "[zCorr, xCorr, PCorr] = correct(kf, z)",
                             "d = distance(kf, zMatrix)", "kf2 = clone(kf)"],
                            "A linear Kalman filter (a handle object). predict advances the state (optionally with a "
                            "control input u: predict(kf, u)), correct folds in a measurement, and distance gives a "
                            "Mahalanobis-based distance of each measurement row from the prediction (useful to "
                            "assign detections to tracks). Scalars given for the covariances mean multiples of the "
                            "identity."),
    "configureKalmanFilter": (["kf = configureKalmanFilter(MotionModel, InitialLocation, InitialEstimateError, "
                               "MotionNoise, MeasurementNoise)"],
                              "A vision.KalmanFilter for tracking a point moving with 'ConstantVelocity' or "
                              "'ConstantAcceleration'. The state holds, for each coordinate of InitialLocation, the "
                              "position followed by its velocity (and acceleration); InitialEstimateError and "
                              "MotionNoise give one variance per state component of a coordinate (2 or 3 values), "
                              "and MeasurementNoise is the variance of a measured position."),
}
