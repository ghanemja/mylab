"""Help text for poles, zeros and time responses of control-system models."""

_TIME = (
    " The optional time argument is either a final time or an evenly spaced vector such as 0:0.01:5 (for a "
    "discrete-time model its step must equal Ts). Without it a duration and step are chosen from the poles "
    "so that the transient is shown until it has settled."
)
_PLOT = (
    " Without output arguments the response is plotted; several models, each optionally followed by a line "
    "style such as 'r--', are drawn in the same axes."
)

HELP = {
    "pole": (
        "p = pole(sys)",
        "Poles of a model as a column vector: the roots of the denominator of a transfer function, or the "
        "eigenvalues of A for a state-space model. Delays are not counted.",
    ),
    "zero": (
        ["z = zero(sys)", "[z, gain] = zero(sys)"],
        "Zeros of a model: the numerator roots for one input and one output (gain is the zero-pole-gain "
        "factor), and the transmission zeros for a model with several inputs or outputs.",
    ),
    "tzero": (
        ["z = tzero(sys)", "z = tzero(A, B, C, D)", "[z, nrank] = tzero(...)"],
        "Invariant zeros of a state-space model: the values of s where the matrix [A - s*I, B; C, D] loses "
        "rank. nrank is the normal rank of the transfer matrix. For a non-square system the zeros are the "
        "ones common to two random square combinations of its channels.",
    ),
    "dcgain": (
        "k = dcgain(sys)",
        "Steady-state gain: the transfer matrix at s = 0 (continuous time) or z = 1 (discrete time). A "
        "channel with an integrator has an infinite DC gain.",
    ),
    "damp": (
        ["damp(sys)", "[wn, zeta] = damp(sys)", "[wn, zeta, p] = damp(sys)", "[wn, zeta, p, tau] = damp(sys)"],
        "Natural frequency, damping ratio and time constant of each pole, sorted by natural frequency. For a "
        "discrete-time model the poles are first mapped to the s-plane with log(z)/Ts. Without outputs a "
        "table is printed.",
    ),
    "impulse": (
        ["impulse(sys)", "impulse(sys, tFinal)", "impulse(sys, t)", "[y, t] = impulse(sys, ...)",
         "[y, t, x] = impulse(sys, ...)"],
        "Impulse response. y has one row per time, one column per output and one page per input; x holds the "
        "states of a state-space model. For a discrete-time model the input is a pulse of height 1/Ts lasting "
        "one sample. A direct feedthrough term (an impulse at t = 0) is not shown." + _TIME + _PLOT,
    ),
    "impulseplot": ("impulseplot(sys, ...)", "Plots the impulse response; same arguments as impulse."),
    "initial": (
        ["initial(sys, x0)", "initial(sys, x0, tFinal)", "initial(sys, x0, t)", "[y, t, x] = initial(sys, x0, ...)"],
        "Response of a state-space model to the initial state x0 with zero input." + _TIME + _PLOT,
    ),
    "initialplot": ("initialplot(sys, x0, ...)", "Plots the response to an initial state; same arguments as "
                    "initial."),
    "lsim": (
        ["lsim(sys, u, t)", "y = lsim(sys, u, t)", "[y, t, x] = lsim(sys, u, t)", "[y, t, x] = lsim(sys, u, t, x0)",
         "y = lsim(sys, u, t, 'zoh')", "y = lsim(sys, u)"],
        "Response to an arbitrary input. u has one row per time sample and one column per input, and t is "
        "evenly spaced (it can be left out for a discrete-time model). x0 is the initial state of a "
        "state-space model. Between samples the input of a continuous-time model is held constant ('zoh') or "
        "joined by straight lines ('foh'); by default 'foh' is used unless the signal is mostly flat with "
        "jumps, like a step or a square wave. The state is advanced exactly for that input, so the result "
        "does not depend on an integration tolerance. Without outputs the response is plotted over the input.",
    ),
    "lsimplot": ("lsimplot(sys, u, t, ...)", "Plots a simulated response; same arguments as lsim."),
    "lsiminfo": (
        ["S = lsiminfo(y, t)", "S = lsiminfo(y, t, yfinal)", "S = lsiminfo(y, t, yfinal, yinit)",
         "S = lsiminfo(..., 'SettlingTimeThreshold', ST)"],
        "Characteristics of a response given as data: TransientTime, SettlingTime, Min, MinTime, Max and "
        "MaxTime. yfinal defaults to the last sample and the settling band is 2 % unless ST says otherwise.",
    ),
    "stepinfo": (
        ["S = stepinfo(sys)", "S = stepinfo(y, t)", "S = stepinfo(y, t, yfinal)", "S = stepinfo(y, t, yfinal, yinit)",
         "S = stepinfo(..., 'SettlingTimeThreshold', ST)", "S = stepinfo(..., 'RiseTimeThreshold', [lo hi])"],
        "Step-response characteristics as a struct: RiseTime (10 % to 90 % of the change), TransientTime, "
        "SettlingTime (the response stays within 2 % of the final value), SettlingMin, SettlingMax, Overshoot "
        "and Undershoot in percent, Peak and PeakTime. For a model the final value is its DC gain and the "
        "crossing times are located on the exact response, not read off a grid; for data they are "
        "interpolated between samples. An unstable model gives NaN (and an infinite peak). A model with "
        "several inputs or outputs gives a struct array.",
    ),
    "gensig": (
        ["[u, t] = gensig(type, tau)", "[u, t] = gensig(type, tau, Tf)", "[u, t] = gensig(type, tau, Tf, Ts)"],
        "Test signal for lsim: 'sine', 'square' or 'pulse' with period tau, lasting Tf (default 5 periods) "
        "and sampled every Ts (default tau/64).",
    ),
    "stepDataOptions": (
        ["opt = stepDataOptions", "opt = stepDataOptions('StepAmplitude', a, 'InputOffset', u0)"],
        "Options for step: the size of the step and the input level before it. With an offset the response "
        "starts from the steady state that belongs to u0. Pass the result as last argument: step(sys, opt).",
    ),
    "RespConfig": (
        ["opt = RespConfig", "opt = RespConfig('Amplitude', a, 'Bias', u0)"],
        "Options for step and impulse under their newer name: Amplitude is the step size and Bias the input "
        "level before the step. Delay and InitialState are not supported.",
    ),
}
