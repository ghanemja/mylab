"""Help text for model reduction, discretization, delays and canonical forms."""

_METHODS = (
    " Methods: 'zoh' (zero-order hold on the input, the default; exact for inputs that are constant "
    "between samples), 'foh' (first-order hold: straight lines between samples), 'impulse' (impulse "
    "invariant), 'tustin' (bilinear transformation; add a critical frequency to match the response there "
    "exactly, as in c2d(sys, Ts, 'prewarp', w0)) and 'matched' (poles and zeros mapped with z = exp(s*Ts), "
    "SISO only)."
)

HELP = {
    "minreal": (
        ["sysr = minreal(sys)", "sysr = minreal(sys, tol)"],
        "Minimal realization. For a state-space model the states that cannot be reached from the inputs or "
        "seen at the outputs are removed (and the number removed is printed); for a transfer function or "
        "zero-pole-gain model every zero that coincides with a pole within tol (default about 1.5e-8, "
        "relative) is cancelled with it.",
    ),
    "sminreal": (
        "sysr = sminreal(sys)",
        "Removes the states of a state-space model that are not connected to any input or output through "
        "nonzero matrix entries. Only the pattern of zeros matters, so no cancellation by rounding can occur.",
    ),
    "gram": (
        ["Wc = gram(sys, 'c')", "Wo = gram(sys, 'o')", "Rc = gram(sys, 'cf')", "Ro = gram(sys, 'of')"],
        "Controllability or observability gramian of a stable model: the solution of A*Wc + Wc*A' + B*B' = 0 "
        "(or A'*Wo + Wo*A + C'*C = 0), and of the corresponding discrete equations for a discrete-time "
        "model. 'cf' and 'of' return the Cholesky factor R with W = R'*R.",
    ),
    "covar": (
        ["P = covar(sys, W)", "[P, Q] = covar(sys, W)"],
        "Output covariance P and state covariance Q of a stable model driven by white noise of intensity W. "
        "A continuous-time model with direct feedthrough has an infinite output covariance.",
    ),
    "balreal": (
        ["sysb = balreal(sys)", "[sysb, g] = balreal(sys)", "[sysb, g, T, Ti] = balreal(sys)"],
        "Balanced realization: a state transformation that makes the controllability and observability "
        "gramians equal and diagonal. g holds their diagonal, the Hankel singular values, in decreasing "
        "order; T maps the old state to the new one (xb = T*x) and Ti is its inverse. Unstable modes are "
        "kept apart, come first and get g = Inf. The model must be minimal.",
    ),
    "hsvd": (
        "g = hsvd(sys)",
        "Hankel singular values of a model: large values belong to states that matter for the input/output "
        "behaviour, small ones to states that balred can remove. Unstable modes give Inf.",
    ),
    "balred": (
        ["rsys = balred(sys, order)", "rsys = balred(sys, order, 'StateProjection', 'Truncate')",
         "rsys = balred(sys, order, opts)"],
        "Reduced-order approximation with the given number of states, obtained by balancing the model and "
        "dropping the states with the smallest Hankel singular values. By default the DC gain is preserved "
        "('MatchDC'); 'Truncate' simply deletes the states, which matches better at high frequency. The "
        "option is also accepted under its older name 'StateElimMethod' or in a balredOptions structure. "
        "Unstable poles are always kept. The result has the type of sys.",
    ),
    "balredOptions": (
        ["opts = balredOptions", "opts = balredOptions('StateProjection', 'Truncate')"],
        "Options for balred: how the discarded states are removed, 'MatchDC' (default) or 'Truncate'.",
    ),
    "modred": (
        ["rsys = modred(sys, elim)", "rsys = modred(sys, elim, 'matchdc')", "rsys = modred(sys, elim, 'truncate')"],
        "Removes the states listed in elim from a state-space model, either keeping the DC gain (the "
        "default, which assumes the removed states settle instantly) or by plain deletion.",
    ),
    "stabsep": (
        "[Gs, Gu] = stabsep(sys)",
        "Splits a model into a stable part Gs and an unstable part Gu with sys = Gs + Gu. Poles on the "
        "stability boundary go to Gu; the direct feedthrough goes to Gs.",
    ),
    "pade": (
        ["[num, den] = pade(T, N)", "sysx = pade(sys, N)", "sysx = pade(sys, NU, NY, NINT)"],
        "Pade approximation of a time delay. [num, den] = pade(T, N) gives the order-N rational "
        "approximation of exp(-T*s). pade(sys, N) replaces every delay of a continuous-time model by such an "
        "approximation, so that functions that need a rational model (feedback, rlocus, ss conversions) can "
        "be used; NU, NY and NINT set separate orders for input, output and input/output delays, and Inf "
        "keeps a delay as it is. For a discrete-time model the delays become poles at z = 0.",
    ),
    "c2d": (
        ["sysd = c2d(sys, Ts)", "sysd = c2d(sys, Ts, method)", "sysd = c2d(sys, Ts, 'prewarp', w0)",
         "sysd = c2d(sys, Ts, opts)", "[sysd, G] = c2d(sys, Ts, method)"],
        "Discretizes a continuous-time model with sample time Ts." + _METHODS + " Delays become whole "
        "sample delays; with 'zoh' a delay that is not a multiple of Ts is treated exactly (the model gets "
        "extra states), with the other methods it is rounded and a warning is given. For a state-space model "
        "and 'zoh' or 'foh', G maps the continuous initial conditions [x0; u0] to the discrete initial state.",
    ),
    "d2c": (
        ["sysc = d2c(sysd)", "sysc = d2c(sysd, method)", "sysc = d2c(sysd, 'prewarp', w0)"],
        "Continuous-time model whose discretization is sysd. Methods: 'zoh' (default; uses a matrix "
        "logarithm and fails for poles at z = 0 or on the negative real axis), 'tustin' and 'matched'.",
    ),
    "d2d": (
        ["sys2 = d2d(sys1, Ts)", "sys2 = d2d(sys1, Ts, method)"],
        "Changes the sample time of a discrete-time model, with the 'zoh' (default) or 'tustin' method. "
        "When the new sample time is a multiple of the old one and the method is 'zoh', the result is exact "
        "for inputs held over the longer period.",
    ),
    "c2dOptions": (
        ["opts = c2dOptions", "opts = c2dOptions('Method', 'tustin', 'PrewarpFrequency', w0)"],
        "Options for c2d: the method and, for 'tustin', the frequency at which the response is matched.",
    ),
    "d2cOptions": (
        ["opts = d2cOptions", "opts = d2cOptions('Method', 'tustin', 'PrewarpFrequency', w0)"],
        "Options for d2c: the method and, for 'tustin', the prewarp frequency.",
    ),
    "d2dOptions": (
        ["opts = d2dOptions", "opts = d2dOptions('Method', 'tustin', 'PrewarpFrequency', w0)"],
        "Options for d2d: the method and, for 'tustin', the prewarp frequency.",
    ),
    "ctrb": (
        ["Co = ctrb(A, B)", "Co = ctrb(sys)"],
        "Controllability matrix [B, A*B, A^2*B, ...]. The system is controllable when its rank equals the "
        "number of states.",
    ),
    "obsv": (
        ["Ob = obsv(A, C)", "Ob = obsv(sys)"],
        "Observability matrix [C; C*A; C*A^2; ...]. The system is observable when its rank equals the "
        "number of states.",
    ),
    "ctrbf": (
        ["[Abar, Bbar, Cbar, T, k] = ctrbf(A, B, C)", "[...] = ctrbf(A, B, C, tol)"],
        "Controllability staircase form: an orthogonal T with Abar = T*A*T' = [Auc 0; A21 Ac] and "
        "Bbar = T*B = [0; Bc], where (Ac, Bc) is controllable and Auc holds the uncontrollable modes. k "
        "lists how many controllable states each step found, so sum(k) is the controllable dimension.",
    ),
    "obsvf": (
        ["[Abar, Bbar, Cbar, T, k] = obsvf(A, B, C)", "[...] = obsvf(A, B, C, tol)"],
        "Observability staircase form: an orthogonal T with Abar = T*A*T' = [Ano A12; 0 Ao] and "
        "Cbar = C*T' = [0 Co], where (Co, Ao) is observable. sum(k) is the observable dimension.",
    ),
    "canon": (
        ["csys = canon(sys, 'modal')", "csys = canon(sys, 'companion')", "[csys, T] = canon(sys, type)"],
        "Canonical state-space forms. 'modal' (default) makes A block diagonal, with real poles on the "
        "diagonal and a 2-by-2 block [re im; -im re] per complex pair; 'companion' puts the coefficients of "
        "the characteristic polynomial in the last column of A. T is the state transformation xc = T*x. "
        "The modal form needs a full set of eigenvectors and the companion form controllability from the "
        "first input.",
    ),
    "ss2ss": (
        "sysT = ss2ss(sys, T)",
        "Change of state coordinates xnew = T*x: the model with T*A/T, T*B and C/T.",
    ),
    "xperm": ("sysP = xperm(sys, P)", "Reorders the states of a state-space model in the order given by P."),
}
