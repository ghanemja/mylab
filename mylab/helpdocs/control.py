"""Help text for control-system model objects: tf, zpk, ss, frd, pid and how to connect them."""

_MODELS = (
    " Models are value objects. Arithmetic (+ - * / \\ ^), [a, b] and [a; b] concatenation, sys(i, j) "
    "indexing and the functions series, parallel, feedback, append, connect and lft combine them; a result "
    "takes the type with the highest precedence: frd, then ss, zpk, tf and pid. Numbers act as static gains."
)
_PROPS = (
    " Properties are read and written with dot syntax or get/set, e.g. sys.Ts, sys.InputDelay, "
    "sys.InputName, sys.OutputName, sys.Name; names are not case sensitive and may be shortened."
)
_DELAY = (
    " Delays are in the time unit of the model for continuous time and in sample periods for discrete time."
)

HELP = {
    "tf": (
        ["sys = tf(num, den)", "sys = tf(num, den, Ts)", "sys = tf(num, den, 'InputDelay', tau, ...)",
         "s = tf('s')", "z = tf('z', Ts)", "sys = tf(othersys)", "sys = tf(gain)"],
        "Transfer function model. num and den are coefficient vectors in descending powers of s (or z); for "
        "several inputs and outputs they are cell arrays with one vector per output/input pair, and a single "
        "den is shared by all pairs. Ts is the sample time (0 or omitted: continuous time, -1: unspecified). "
        "tf('s') is the Laplace variable, so G = 1/(s^2 + 2*s + 1) builds a model with ordinary algebra, and "
        "exp(-tau*s) adds a time delay. tf(othersys) converts a zpk, ss or pid model. Properties: Numerator and "
        "Denominator (also num, den; cell arrays), Variable ('s', 'p', 'z', 'q', 'z^-1', 'q^-1'), Ts, "
        "InputDelay, OutputDelay, IODelay, InputName, OutputName, Name." + _MODELS + _PROPS + _DELAY,
    ),
    "filt": (
        ["sys = filt(num, den)", "sys = filt(num, den, Ts)"],
        "Discrete-time transfer function written the way digital filters are: num and den hold coefficients "
        "in ascending powers of z^-1, so filt([1 2], [1 0.5]) is (1 + 2 z^-1)/(1 + 0.5 z^-1). The sample time "
        "is unspecified (-1) unless Ts is given. The result is a tf model whose Variable is 'z^-1'.",
    ),
    "zpk": (
        ["sys = zpk(z, p, k)", "sys = zpk(z, p, k, Ts)", "s = zpk('s')", "sys = zpk(othersys)"],
        "Zero-pole-gain model k*(s - z1)(s - z2).../((s - p1)(s - p2)...). z and p are vectors ([] for none); "
        "for several inputs and outputs they are cell arrays and k is a matrix. zpk(othersys) converts a tf, "
        "ss or pid model by computing its zeros and poles. Properties: Z, P (cell arrays of column vectors), K, "
        "DisplayFormat ('roots', 'frequency' or 'time constant'), Variable, Ts and the delay and name "
        "properties." + _MODELS + _PROPS,
    ),
    "ss": (
        ["sys = ss(A, B, C, D)", "sys = ss(A, B, C, D, Ts)", "sys = ss(D)", "sys = ss(othersys)",
         "sys = ss(othersys, 'minimal')"],
        "State-space model dx/dt = A x + B u, y = C x + D u (x[k+1] = A x[k] + B u[k] with a sample time Ts). "
        "D = 0 stands for a zero matrix of the right size. ss(othersys) builds a realization of a tf, zpk or "
        "pid model: controller canonical form for one input and one output, one block per column (or row) "
        "otherwise; 'minimal' also removes states that cannot be reached or observed. Properties: A, B, C, D, "
        "StateName, Ts, InputDelay, OutputDelay, InputName, OutputName." + _MODELS + _PROPS,
    ),
    "dss": (
        ["sys = dss(A, B, C, D, E)", "sys = dss(A, B, C, D, E, Ts)"],
        "State-space model given in descriptor form E dx/dt = A x + B u, y = C x + D u. E must be "
        "nonsingular; the model is stored in the ordinary form with E\\A and E\\B.",
    ),
    "frd": (
        ["sys = frd(response, freq)", "sys = frd(response, freq, Ts)", "sys = frd(othersys, freq)",
         "sys = frd(..., 'FrequencyUnit', 'Hz')"],
        "Frequency-response data model: complex response samples at the frequencies freq (rad per time unit "
        "unless FrequencyUnit says otherwise). response is a vector for one input and one output, or an "
        "ny-by-nu-by-nf array. frd(othersys, freq) samples another model. frd models can be added, multiplied, "
        "inverted and put in feedback loops point by point (the frequencies must agree), and bode, nyquist, "
        "nichols, sigma, margin and getPeakGain work on the samples. Properties: Frequency, ResponseData, "
        "FrequencyUnit, Ts.",
    ),
    "pid": (
        ["C = pid(Kp)", "C = pid(Kp, Ki)", "C = pid(Kp, Ki, Kd)", "C = pid(Kp, Ki, Kd, Tf)",
         "C = pid(Kp, Ki, Kd, Tf, Ts)", "C = pid(sys)", "C = pid(..., 'IFormula', 'Trapezoidal')"],
        "PID controller in parallel form, Kp + Ki/s + Kd*s/(Tf*s + 1). Omitted gains are 0, so pid(2, 1) is a "
        "PI controller. With a sample time Ts the integrators use the formulas IFormula and DFormula: "
        "'ForwardEuler' (default), 'BackwardEuler' or 'Trapezoidal'. pid(sys) converts a transfer function "
        "that has PID structure. Properties: Kp, Ki, Kd, Tf, IFormula, DFormula, Ts. Sums of PID controllers "
        "and products with a number stay pid objects; other combinations give a tf, zpk or ss model, e.g. "
        "feedback(C*G, 1).",
    ),
    "pidstd": (
        ["C = pidstd(Kp)", "C = pidstd(Kp, Ti)", "C = pidstd(Kp, Ti, Td)", "C = pidstd(Kp, Ti, Td, N)",
         "C = pidstd(Kp, Ti, Td, N, Ts)", "C = pidstd(sys)"],
        "PID controller in standard form, Kp*(1 + 1/(Ti*s) + Td*s/(Td/N*s + 1)). Ti = Inf turns the integral "
        "action off and N = Inf removes the derivative filter. pidstd(sys) converts a pid object or a "
        "transfer function with PID structure. Properties: Kp, Ti, Td, N, IFormula, DFormula, Ts.",
    ),
    "tfdata": (
        ["[num, den] = tfdata(sys)", "[num, den, Ts] = tfdata(sys)", "[num, den] = tfdata(sys, 'v')"],
        "Numerator and denominator coefficients of a model (converted to a transfer function if necessary). "
        "The outputs are cell arrays with one row vector per output/input pair; with 'v' and a SISO model "
        "they are plain row vectors. Both vectors of a pair have the same length (the shorter one is padded "
        "with leading zeros), which is the form the Transfer Fcn block and filter functions expect.",
    ),
    "zpkdata": (
        ["[z, p, k] = zpkdata(sys)", "[z, p, k, Ts] = zpkdata(sys)", "[z, p, k] = zpkdata(sys, 'v')"],
        "Zeros, poles and gain of a model. z and p are cell arrays of column vectors, one per output/input "
        "pair, and k is a matrix; with 'v' and a SISO model z and p are column vectors.",
    ),
    "ssdata": (
        ["[A, B, C, D] = ssdata(sys)", "[A, B, C, D, Ts] = ssdata(sys)"],
        "State-space matrices of a model. A tf, zpk or pid model is first converted with ss(sys).",
    ),
    "dssdata": (
        ["[A, B, C, D, E] = dssdata(sys)", "[A, B, C, D, E, Ts] = dssdata(sys)"],
        "State-space matrices in descriptor form. Models are stored with E equal to the identity, which is "
        "what is returned.",
    ),
    "frdata": (
        ["[response, freq] = frdata(sys)", "[response, freq, Ts] = frdata(sys)",
         "[response, freq] = frdata(sys, 'v')"],
        "Response samples and frequencies of an frd model. response is ny-by-nu-by-nf; with 'v' and a SISO "
        "model it is a column vector.",
    ),
    "piddata": (
        ["[Kp, Ki, Kd, Tf] = piddata(sys)", "[Kp, Ki, Kd, Tf, Ts] = piddata(sys)"],
        "Gains of a controller in parallel form. sys may be a pid or pidstd object or a transfer function "
        "with PID structure.",
    ),
    "pidstddata": (
        ["[Kp, Ti, Td, N] = pidstddata(sys)", "[Kp, Ti, Td, N, Ts] = pidstddata(sys)"],
        "Parameters of a controller in standard form: gain, integral time, derivative time and filter "
        "divisor. An error is raised if the controller has no proportional gain.",
    ),
    "order": (
        "n = order(sys)",
        "Order of a model: the number of states of a state-space model, the number of poles of a SISO "
        "transfer function or zero-pole-gain model, and the number of states of its realization for a model "
        "with several inputs or outputs.",
    ),
    "isct": ("tf = isct(sys)", "True for a continuous-time model (sample time 0)."),
    "isdt": (
        "tf = isdt(sys)",
        "True for a discrete-time model (nonzero sample time). A static gain counts as both continuous and "
        "discrete.",
    ),
    "issiso": ("tf = issiso(sys)", "True for a model with one input and one output."),
    "isstatic": ("tf = isstatic(sys)", "True for a model without dynamics or delays (a pure gain)."),
    "isstable": (
        "tf = isstable(sys)",
        "True when all poles lie in the open left half-plane (continuous time) or strictly inside the unit "
        "circle (discrete time). Delays do not affect the answer.",
    ),
    "isproper": (
        ["tf = isproper(sys)", "tf = isproper(sys, 'elem')"],
        "True when no channel has more zeros than poles. With 'elem' the result is a logical matrix with one "
        "entry per output/input pair.",
    ),
    "hasdelay": ("tf = hasdelay(sys)", "True when the model has an input, output or input/output delay."),
    "totaldelay": (
        "td = totaldelay(sys)",
        "Matrix of the total delay between each input and output: input delay + output delay + I/O delay.",
    ),
    "absorbDelay": (
        "sysnd = absorbDelay(sys)",
        "Turns the delays of a discrete-time model into poles at z = 0 (a delay of d samples becomes z^-d in "
        "the model itself). For continuous time use pade to approximate delays.",
    ),
    "feedback": (
        ["T = feedback(sys1, sys2)", "T = feedback(sys1, sys2, sign)",
         "T = feedback(sys1, sys2, feedin, feedout)", "T = feedback(sys1, sys2, feedin, feedout, sign)"],
        "Closed loop of sys1 with sys2 in the return path: T = sys1/(1 + sys1*sys2) for negative feedback "
        "(the default) and sys1/(1 - sys1*sys2) for sign = +1. feedback(L, 1) closes a unity-feedback loop. "
        "feedin and feedout select which inputs and outputs of sys1 take part in the loop; the result keeps all "
        "inputs and outputs of sys1. Continuous-time delays inside the loop are not supported: approximate "
        "them first with pade.",
    ),
    "series": (
        ["sys = series(sys1, sys2)", "sys = series(sys1, sys2, outputs1, inputs2)"],
        "Series connection: the outputs of sys1 drive the inputs of sys2, which is sys2*sys1. The index "
        "vectors connect only the listed outputs of sys1 to the listed inputs of sys2.",
    ),
    "parallel": (
        ["sys = parallel(sys1, sys2)", "sys = parallel(sys1, sys2, inp1, inp2, out1, out2)"],
        "Parallel connection: both models receive the same input and their outputs are added, which is "
        "sys1 + sys2. With index vectors only the listed inputs are shared and only the listed outputs are "
        "summed; the other channels of both models are kept (those of sys1 first).",
    ),
    "append": (
        "sys = append(sys1, sys2, ...)",
        "Block-diagonal model that keeps the models side by side without connecting them: its inputs and "
        "outputs are those of sys1 followed by those of sys2, and so on. Numbers are static gains.",
    ),
    "lft": (
        ["sys = lft(sys1, sys2)", "sys = lft(sys1, sys2, nu, ny)"],
        "Linear fractional (star) connection: the last nu inputs and ny outputs of sys1 are closed with sys2 "
        "in positive feedback. Without nu and ny all channels of the smaller model are used: sys2 closes the "
        "lower channels of sys1, or sys1 closes the upper channels of sys2.",
    ),
    "connect": (
        ["sys = connect(sys1, ..., sysN, inputs, outputs)", "sys = connect(blksys, Q, inputs, outputs)"],
        "Builds a model from a block diagram. In the first form every block has InputName and OutputName "
        "set, and blocks are wired wherever an input name equals an output name; inputs and outputs name the "
        "external signals of the result (summing junctions come from sumblk). In the second form blksys is an "
        "appended model and each row of Q is [input, outputs feeding it], with a minus sign for a subtracted "
        "output; inputs and outputs are index vectors. The result is a state-space model.",
    ),
    "sumblk": (
        ["S = sumblk('e = r - y')", "S = sumblk(formula, n)"],
        "Summing junction for connect: a static-gain model whose output and inputs carry the signal names of "
        "the formula, with gains +1 or -1. A second argument n makes every signal a vector of n channels.",
    ),
}
