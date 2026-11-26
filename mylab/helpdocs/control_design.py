"""Help text for control design: Lyapunov and Riccati equations, LQ design, estimators, PID tuning."""

_APP = (
    " This interactive app is not available; the function only reports that and points to the "
    "command-line functions that do the same work."
)
_SYSFORM = " A model can be given instead of A and B, as in lqr(sys, Q, R)."

HELP = {
    "lyap": (
        ["X = lyap(A, Q)", "X = lyap(A, B, C)", "X = lyap(A, Q, [], E)"],
        "Solves the continuous Lyapunov equation A*X + X*A' + Q = 0, the Sylvester equation "
        "A*X + X*B + C = 0, or the generalized equation A*X*E' + E*X*A' + Q = 0. A unique solution needs "
        "that no eigenvalue of A is the negative of another (for example a stable A).",
    ),
    "dlyap": (
        ["X = dlyap(A, Q)", "X = dlyap(A, B, C)", "X = dlyap(A, Q, [], E)"],
        "Solves the discrete Lyapunov equation A*X*A' - X + Q = 0, the Sylvester equation "
        "A*X*B - X + C = 0, or A*X*A' - E*X*E' + Q = 0.",
    ),
    "lyapchol": (
        "R = lyapchol(A, B)",
        "Cholesky factor R of the solution X = R'*R of A*X + X*A' + B*B' = 0 for a stable A.",
    ),
    "dlyapchol": (
        "R = dlyapchol(A, B)",
        "Cholesky factor R of the solution X = R'*R of A*X*A' - X + B*B' = 0 for a stable A.",
    ),
    "care": (
        ["X = care(A, B, Q)", "X = care(A, B, Q, R, S, E)", "[X, L, G] = care(...)"],
        "Stabilizing solution of the continuous algebraic Riccati equation "
        "A'*X*E + E'*X*A - (E'*X*B + S)*inv(R)*(B'*X*E + S') + Q = 0. R defaults to the identity, S to zero "
        "and E to the identity. G = inv(R)*(B'*X*E + S') is the feedback gain and L the closed-loop poles "
        "of A - B*G.",
    ),
    "dare": (
        ["X = dare(A, B, Q)", "X = dare(A, B, Q, R, S, E)", "[X, L, G] = dare(...)"],
        "Stabilizing solution of the discrete algebraic Riccati equation "
        "A'*X*A - E'*X*E - (A'*X*B + S)*inv(B'*X*B + R)*(B'*X*A + S') + Q = 0, with the gain "
        "G = inv(B'*X*B + R)*(B'*X*A + S') and the closed-loop poles L.",
    ),
    "icare": (
        ["X = icare(A, B, Q)", "X = icare(A, B, Q, R, S, E)", "[X, K, L, info] = icare(...)"],
        "Stabilizing solution of the continuous algebraic Riccati equation (see care), with the outputs in "
        "the order solution, gain K, closed-loop poles L. [] selects a default (R = I, S = 0, E = I). info "
        "has Report = 0 on success; a nonzero G term is not supported.",
    ),
    "idare": (
        ["X = idare(A, B, Q)", "X = idare(A, B, Q, R, S, E)", "[X, K, L, info] = idare(...)"],
        "Stabilizing solution of the discrete algebraic Riccati equation (see dare), with the outputs in "
        "the order solution, gain K, closed-loop poles L.",
    ),
    "lqr": (
        ["K = lqr(A, B, Q, R)", "K = lqr(A, B, Q, R, N)", "[K, S, P] = lqr(...)", "K = lqr(sys, Q, R, N)"],
        "Linear-quadratic regulator: the state feedback u = -K*x that minimizes the integral of "
        "x'*Q*x + u'*R*u + 2*x'*N*u. S is the Riccati solution and P the closed-loop poles of A - B*K. R must "
        "be positive definite and (A, B) stabilizable. For a discrete-time model the discrete problem is "
        "solved." + _SYSFORM,
    ),
    "dlqr": (
        ["K = dlqr(A, B, Q, R)", "K = dlqr(A, B, Q, R, N)", "[K, S, P] = dlqr(...)"],
        "Discrete-time linear-quadratic regulator: u[k] = -K*x[k] minimizes the sum of "
        "x'*Q*x + u'*R*u + 2*x'*N*u for x[k+1] = A*x[k] + B*u[k].",
    ),
    "lqrd": (
        ["K = lqrd(A, B, Q, R, Ts)", "K = lqrd(A, B, Q, R, N, Ts)", "[K, S, P] = lqrd(...)"],
        "Discrete regulator for a continuous plant: the gain for u[k] = -K*x[k] with a zero-order hold that "
        "minimizes the continuous-time cost of lqr. The plant and the cost are both discretized exactly.",
    ),
    "lqry": (
        ["K = lqry(sys, Q, R)", "K = lqry(sys, Q, R, N)", "[K, S, P] = lqry(...)"],
        "Linear-quadratic regulator with a weight on the outputs: minimizes y'*Q*y + u'*R*u + 2*y'*N*u.",
    ),
    "lqi": (
        ["K = lqi(sys, Q, R)", "K = lqi(sys, Q, R, N)", "[K, S, e] = lqi(...)"],
        "Linear-quadratic regulator with integral action. The state is extended with the integral xi of "
        "r - y, and K = [Kx, Ki] is used as u = -K*[x; xi]. Q weights the extended state, so it has one row "
        "per state plus one per output.",
    ),
    "lqe": (
        ["L = lqe(A, G, C, Q, R)", "L = lqe(A, G, C, Q, R, N)", "[L, P, E] = lqe(...)"],
        "Kalman estimator gain for dx/dt = A*x + B*u + G*w, y = C*x + v with process-noise covariance Q, "
        "measurement-noise covariance R and cross-covariance N. P is the error covariance and E the "
        "estimator poles of A - L*C.",
    ),
    "dlqe": (
        "[M, P, Z, E] = dlqe(A, G, C, Q, R)",
        "Discrete Kalman estimator: the innovation gain M, the prediction error covariance P, the error "
        "covariance after the measurement update Z and the estimator poles E.",
    ),
    "kalman": (
        ["[kest, L, P] = kalman(sys, Qn, Rn)", "[kest, L, P] = kalman(sys, Qn, Rn, Nn)",
         "[kest, L, P] = kalman(sys, Qn, Rn, Nn, sensors, known)", "[kest, L, P, M, Z] = kalman(sys, ...)"],
        "Kalman estimator for a state-space plant. The last inputs of sys are the process noise w (as many "
        "as Qn has rows) unless known lists the deterministic inputs; sensors lists the measured outputs. "
        "Qn, Rn and Nn are the covariances of w, of the measurement noise v and between them. kest takes "
        "[known inputs; measurements] and returns [output estimates; state estimates]. L is the estimator "
        "gain and P the error covariance; for a discrete-time plant M and Z are the innovation gain and the "
        "updated covariance, and a last argument 'current' (default) or 'delayed' chooses whether the "
        "estimate uses the newest measurement.",
    ),
    "estim": (
        ["est = estim(sys, L)", "est = estim(sys, L, sensors, known)"],
        "State estimator with a given gain L: its inputs are the known inputs and the measured outputs of "
        "sys, its outputs the estimated outputs followed by the estimated states.",
    ),
    "reg": (
        ["rsys = reg(sys, K, L)", "rsys = reg(sys, K, L, sensors, known, controls)"],
        "Regulator made of a state-feedback gain K and an estimator gain L (u = -K*xhat). rsys maps the "
        "measurements to the control inputs and is meant to be connected with positive feedback: "
        "feedback(sys, rsys, +1).",
    ),
    "lqg": (
        "rsys = lqg(sys, QXU, QWV)",
        "Linear-quadratic-Gaussian regulator. QXU weights [x; u] in the cost and QWV is the covariance of "
        "[w; v] for dx/dt = A*x + B*u + w, y = C*x + D*u + v. The result maps y to u and is connected with "
        "positive feedback: feedback(sys, rsys, +1). Options for integral action or two degrees of freedom "
        "are not supported.",
    ),
    "lqgreg": (
        "rlqg = lqgreg(kest, K)",
        "Regulator formed from a Kalman estimator (from kalman) and a state-feedback gain: u = -K*xhat. "
        "The result maps the measurements to the controls, for positive feedback.",
    ),
    "place": (
        ["K = place(A, B, p)", "[K, prec] = place(A, B, p)"],
        "State-feedback gain that puts the eigenvalues of A - B*K at the values in p. A pole may not be "
        "repeated more often than B has independent columns. prec estimates how many digits of the "
        "requested poles were achieved. For an observer gain use L = place(A', C', p)'.",
    ),
    "acker": (
        "K = acker(A, B, p)",
        "Pole placement for a single-input system with Ackermann's formula; repeated poles are allowed. "
        "It is less accurate than place for systems of higher order.",
    ),
    "pidtune": (
        ["C = pidtune(sys, type)", "C = pidtune(sys, type, wc)", "C = pidtune(sys, C0)",
         "C = pidtune(sys, type, opts)", "[C, info] = pidtune(...)"],
        "Tunes a PID controller for the SISO plant sys in a unity-feedback loop. type is 'P', 'I', 'PI', "
        "'PD', 'PDF', 'PID' or 'PIDF' (F adds a filter on the derivative); a pid or pidstd object C0 gives "
        "the type, form and discrete formulas instead. The controller is designed in the frequency domain: "
        "at the crossover frequency wc the loop gain is 1 and the phase margin equals the target (60 "
        "degrees unless pidtuneOptions says otherwise). When wc is not given it is chosen to give the "
        "largest integral gain (the fastest disturbance rejection) among the designs that are stable and "
        "keep the target margin; for plants where that has no limit, a crossover of twice the slowest plant "
        "pole is used. If the margin cannot be reached at a requested wc the best achievable design is "
        "returned. info reports Stable, CrossoverFrequency and PhaseMargin. Two-degree-of-freedom types are "
        "not supported.",
    ),
    "pidtuneOptions": (
        ["opts = pidtuneOptions", "opts = pidtuneOptions('PhaseMargin', 45, 'DesignFocus', 'reference-tracking')"],
        "Options for pidtune: PhaseMargin in degrees (default 60), DesignFocus ('balanced', "
        "'reference-tracking' for weaker integral action and less overshoot, or 'disturbance-rejection' for "
        "stronger integral action), CrossoverFrequency and NumUnstablePoles.",
    ),
    "sisotool": ("sisotool(sys)", "SISO design app." + _APP),
    "rltool": ("rltool(sys)", "Root-locus design app." + _APP),
    "controlSystemDesigner": ("controlSystemDesigner(sys)", "Control system design app." + _APP),
    "pidTuner": ("pidTuner(sys, type)", "PID tuning app." + _APP),
    "pidtool": ("pidtool(sys, type)", "PID tuning app (older name of pidTuner)." + _APP),
    "ltiview": ("ltiview(sys)", "Response viewer app." + _APP),
    "linearSystemAnalyzer": ("linearSystemAnalyzer(sys)", "Response viewer app." + _APP),
    "ss2tf": (
        ["[b, a] = ss2tf(A, B, C, D)", "[b, a] = ss2tf(A, B, C, D, iu)"],
        "Transfer function coefficients of a state-space system from input iu: a is the characteristic "
        "polynomial of A and b has one row of numerator coefficients per output, of the same length as a.",
    ),
    "tf2ss": (
        "[A, B, C, D] = tf2ss(num, den)",
        "Controller canonical state-space form of a proper transfer function. num may have one row per "
        "output; den is shared.",
    ),
    "zp2tf": (
        "[num, den] = zp2tf(z, p, k)",
        "Polynomial coefficients from zeros, poles and gain. z has one column per output (use Inf to pad) "
        "and k one gain per output; num has one row per output.",
    ),
    "tf2zp": (
        "[z, p, k] = tf2zp(num, den)",
        "Zeros, poles and gain of a transfer function given by coefficients in descending powers of s. "
        "With several rows in num, z has one column per output, padded with Inf.",
    ),
    "zp2ss": ("[A, B, C, D] = zp2ss(z, p, k)", "State-space form of a system given by zeros, poles and gain."),
    "ss2zp": (
        ["[z, p, k] = ss2zp(A, B, C, D)", "[z, p, k] = ss2zp(A, B, C, D, iu)"],
        "Zeros, poles and gains of a state-space system from input iu, one column of zeros and one gain per "
        "output.",
    ),
}
