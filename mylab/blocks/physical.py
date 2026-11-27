"""Physical networks: electrical and mechanical components joined by two-way connections.

A physical block has *conserving ports* (named, such as ``p`` and ``n``)
instead of, or besides, signal ports. Lines between conserving ports are not
directed: they join the ports into *nodes*. At a node the *across* variable
(voltage, velocity, angular velocity) is the same for every port, and the
*through* variables (current, force, torque) flowing into the components add
up to zero.

When a model is compiled, all physical blocks that are connected form one
network, which becomes one block of the simulation (:class:`PhysicalNetwork`).
Its unknowns are the across variable of each node and the through and
internal variables of the components; its continuous states are the ones the
components declare (a capacitor's voltage, an inductor's current, a mass's
velocity, a spring's stretch). Each component adds as many equations as
unknowns, and each node adds the balance of its through variables, which
gives a differential-algebraic system

    dx/dt = f(x, z, u, t),    0 = g(x, z, u, t)

with states x, algebraic unknowns z and input signals u. mylab requires
index 1: the matrix dg/dz must be invertible, so that z follows from x and u
(by Newton's method; in one step when the network is linear). The ordinary
solvers then integrate x. A network that is not index 1 (a capacitor
directly across a voltage source, two masses joined rigidly) or that leaves
a variable undetermined (a circuit without a reference) is an error that
names the components involved.

Sign conventions (positive directions):

- two-port components carry their through variable from the first port to
  the second: current from ``p`` to ``n``; force and torque from ``R`` to
  ``C`` (translational) and from ``F`` to ``B`` (rotational); the across
  variable of the component is first port minus second (``v = v_p - v_n``,
  ``v = v_R - v_C``, ``w = w_F - w_B``);
- a source with a positive input drives the external circuit from its second
  port to its first: a current source pushes current out of ``p``, a force
  source pushes port ``R`` forward relative to ``C``;
- a mass or inertia has one port; the force or torque flowing into it
  accelerates it.

Physical signals are ordinary signals here: sensors output signals and
controlled sources take them. The PS-Simulink and Simulink-PS converter blocks
pass a signal on unchanged (they exist so that diagrams drawn with them keep
their shape). The Solver Configuration block is accepted and has no effect.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .library import BlockError, Node, P, block

DOMAINS = {
    "electrical": "voltage / current",
    "translational": "velocity / force",
    "rotational": "angular velocity / torque",
}


class PhysBlock(Node):
    """A component of a physical network. ``ports`` lists its conserving ports as (name, domain)."""

    physical = True
    ports: list = []
    nin: Any = 0
    nout: Any = 0
    nonlinear = False

    def define(self, net: "Network", node: dict, u0: int, y0: int) -> None:
        """Add the component's unknowns, states and equations. ``node`` maps port name -> node number,
        ``u0``/``y0`` are where its input and output signals start in the network's lists."""


class _S:
    """The values the equations are evaluated at."""

    __slots__ = ("x", "z", "u", "t")


class Network:
    """Builds the equations of one physical network."""

    def __init__(self, n_nodes: int, node_names: list[str]):
        self.n_nodes = n_nodes
        self.node_names = node_names
        self.z_names: list[str] = [f"across variable of the node at {n}" for n in node_names]
        self.z_owner: list[str] = ["" for _ in node_names]
        self.eqs: list[tuple[Callable, str, str]] = []  # (residual, owner path, what)
        self.kcl: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
        self.x0: list[float] = []
        self.x_names: list[str] = []
        self.ders: list[Callable] = []
        self.outs: list[Callable] = []
        self.zcs: list[Callable] = []
        self.grounded: set[int] = set()
        self.modes: list[bool] = []  # switches of piecewise-linear components (a diode conducts or not)
        self.mode_tests: list[Callable] = []

    def mode(self, test: Callable) -> int:
        """A switch decided by ``test`` (true or false at the solution); the equations may read net.modes[k]."""
        self.modes.append(False)
        self.mode_tests.append(test)
        return len(self.modes) - 1

    # unknowns
    def alg(self, owner: str, what: str) -> int:
        self.z_names.append(f"{what} of '{owner}'")
        self.z_owner.append(owner)
        return len(self.z_names) - 1

    def state(self, owner: str, what: str, x0: float, der: Callable) -> int:
        self.x0.append(float(x0))
        self.x_names.append(f"{what} of '{owner}'")
        self.ders.append(der)
        return len(self.x0) - 1

    def e(self, node: int) -> Callable:
        return lambda s: s.z[node]

    # equations
    def eq(self, owner: str, what: str, fn: Callable) -> None:
        self.eqs.append((fn, owner, what))

    def flow(self, node: int, zi: int, sign: float = 1.0) -> None:
        """The unknown z[zi] flows from the node into the component (or out of it, with sign -1)."""
        self.kcl[node].append((zi, sign))

    def branch(self, owner: str, a: int, b: int, what: str) -> int:
        """A through variable that flows into the component at node a and out of it at node b."""
        zi = self.alg(owner, what)
        self.flow(a, zi, 1.0)
        self.flow(b, zi, -1.0)
        return zi

    def output(self, fn: Callable) -> None:
        self.outs.append(fn)


def _scalar(node: Node, name: str) -> float:
    return node.scalar(name)


# -------------------------------------------------------------------------------------
# Components
# -------------------------------------------------------------------------------------


def phys(type_name: str, doc: str, title: str = "", icon: str = "", aliases: tuple = ()):
    return block(type_name, "Physical", doc, title, aliases, icon)


class TwoPort(PhysBlock):
    domain = "electrical"
    first, second = "p", "n"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "ports" not in cls.__dict__:
            cls.ports = [(cls.first, cls.domain), (cls.second, cls.domain)]


class _Electrical(TwoPort):
    domain, first, second = "electrical", "p", "n"


class _Translational(TwoPort):
    domain, first, second = "translational", "R", "C"


class _Rotational(TwoPort):
    domain, first, second = "rotational", "F", "B"


def _rel(net: Network, a: int, b: int) -> Callable:
    return lambda s: s.z[a] - s.z[b]


@phys("Resistor", "Electrical resistor: v = R i between its ports p and n.", icon="R")
class Resistor(_Electrical):
    params = [P("R", "1", "Resistance (ohm)")]

    def define(self, net, node, u0, y0):
        r = _scalar(self, "R")
        a, b = node["p"], node["n"]
        i = net.branch(self.path, a, b, "current")
        net.eq(self.path, "Ohm's law", lambda s: (s.z[a] - s.z[b]) - r * s.z[i])


@phys("Capacitor", "Electrical capacitor: i = C dv/dt; its voltage is a state.", icon="C")
class Capacitor(_Electrical):
    params = [P("C", "1e-6", "Capacitance (F)"), P("v0", "0", "Initial voltage (V)")]

    def define(self, net, node, u0, y0):
        c = _scalar(self, "C")
        if c <= 0:
            raise BlockError("the capacitance must be positive")
        a, b = node["p"], node["n"]
        i = net.branch(self.path, a, b, "current")
        k = net.state(self.path, "voltage", _scalar(self, "v0"), lambda s: s.z[i] / c)
        net.eq(self.path, "capacitor voltage", lambda s: (s.z[a] - s.z[b]) - s.x[k])


@phys("Inductor", "Electrical inductor: v = L di/dt; its current is a state.", icon="L")
class Inductor(_Electrical):
    params = [P("L", "1e-3", "Inductance (H)"), P("i0", "0", "Initial current (A)")]

    def define(self, net, node, u0, y0):
        ind = _scalar(self, "L")
        if ind <= 0:
            raise BlockError("the inductance must be positive")
        a, b = node["p"], node["n"]
        i = net.branch(self.path, a, b, "current")
        k = net.state(self.path, "current", _scalar(self, "i0"), lambda s: (s.z[a] - s.z[b]) / ind)
        net.eq(self.path, "inductor current", lambda s: s.z[i] - s.x[k])


@phys(
    "Diode",
    "Piecewise-linear diode: conducts with resistance Ron above the forward voltage Vf, else conductance Goff.",
    icon="D",
)
class Diode(_Electrical):
    params = [P("Vf", "0.6", "Forward voltage (V)"), P("Ron", "0.01", "On resistance (ohm)"),
              P("Goff", "1e-8", "Off conductance (1/ohm)")]

    def define(self, net, node, u0, y0):
        vf, ron, goff = _scalar(self, "Vf"), _scalar(self, "Ron"), _scalar(self, "Goff")
        if ron <= 0 or goff < 0:
            raise BlockError("Ron must be positive and Goff not negative")
        a, b = node["p"], node["n"]
        i = net.branch(self.path, a, b, "current")
        k = net.mode(lambda s: (s.z[a] - s.z[b]) >= vf)  # conducting or not: linear in each mode

        def law(s):
            v = s.z[a] - s.z[b]
            cur = goff * vf + (v - vf) / ron if net.modes[k] else goff * v
            return s.z[i] - cur

        net.eq(self.path, "diode law", law)
        net.zcs.append(lambda s: (s.z[a] - s.z[b]) - vf)


@phys("DCVoltageSource", "Constant voltage between p and n: v_p - v_n = V.", title="DC Voltage Source", icon="V")
class DCVoltageSource(_Electrical):
    params = [P("V", "1", "Voltage (V)")]

    def define(self, net, node, u0, y0):
        v = _scalar(self, "V")
        a, b = node["p"], node["n"]
        net.branch(self.path, a, b, "current")
        net.eq(self.path, "source voltage", lambda s: (s.z[a] - s.z[b]) - v)


@phys("ControlledVoltageSource", "Voltage between p and n given by the input signal.", icon="V(u)")
class ControlledVoltageSource(_Electrical):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        a, b = node["p"], node["n"]
        net.branch(self.path, a, b, "current")
        net.eq(self.path, "source voltage", lambda s: (s.z[a] - s.z[b]) - s.u[u0])


@phys("DCCurrentSource", "Constant current I pushed out of port p through the circuit back into n.",
      title="DC Current Source", icon="I")
class DCCurrentSource(_Electrical):
    params = [P("I", "1", "Current (A)")]

    def define(self, net, node, u0, y0):
        cur = _scalar(self, "I")
        i = net.branch(self.path, node["p"], node["n"], "current")
        net.eq(self.path, "source current", lambda s: s.z[i] + cur)


@phys("ControlledCurrentSource", "Current given by the input signal, pushed out of port p.", icon="I(u)")
class ControlledCurrentSource(_Electrical):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        i = net.branch(self.path, node["p"], node["n"], "current")
        net.eq(self.path, "source current", lambda s: s.z[i] + s.u[u0])


@phys("ElectricalReference", "Electrical ground: the voltage of the node it is connected to is zero.", icon="gnd")
class ElectricalReference(PhysBlock):
    params = []
    ports = [("p", "electrical")]

    def define(self, net, node, u0, y0):
        _reference(self, net, node["p"])


@phys("VoltageSensor", "Outputs the voltage v_p - v_n; no current flows through it.", icon="V?")
class VoltageSensor(_Electrical):
    params = []
    nout = 1

    def define(self, net, node, u0, y0):
        a, b = node["p"], node["n"]
        net.output(lambda s: s.z[a] - s.z[b])


@phys("CurrentSensor", "Outputs the current flowing from p to n through it (it joins p and n).", icon="I?")
class CurrentSensor(_Electrical):
    params = []
    nout = 1

    def define(self, net, node, u0, y0):
        a, b = node["p"], node["n"]
        i = net.branch(self.path, a, b, "current")
        net.eq(self.path, "no voltage drop", lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.z[i])


def _reference(blk: PhysBlock, net: Network, n: int) -> None:
    if n in net.grounded:
        return  # one reference per node is enough
    net.grounded.add(n)
    r = net.alg(blk.path, "reaction")
    net.flow(n, r, 1.0)
    net.eq(blk.path, "reference", lambda s: s.z[n])


# -- mechanical translational -----------------------------------------------------------


@phys("Mass", "A mass m: the force flowing into its port accelerates it; its velocity is a state.", icon="m")
class Mass(PhysBlock):
    params = [P("m", "1", "Mass (kg)"), P("v0", "0", "Initial velocity (m/s)")]
    ports = [("M", "translational")]

    def define(self, net, node, u0, y0):
        m = _scalar(self, "m")
        if m <= 0:
            raise BlockError("the mass must be positive")
        a = node["M"]
        f = net.alg(self.path, "force")
        net.flow(a, f, 1.0)
        k = net.state(self.path, "velocity", _scalar(self, "v0"), lambda s: s.z[f] / m)
        net.eq(self.path, "mass velocity", lambda s: s.z[a] - s.x[k])


@phys("TranslationalSpring", "Spring: force k x, where x is how far R has moved relative to C (a state).",
      title="Translational Spring", icon="k")
class TranslationalSpring(_Translational):
    params = [P("k", "1", "Spring rate (N/m)"), P("x0", "0", "Initial deformation (m)")]

    def define(self, net, node, u0, y0):
        kk = _scalar(self, "k")
        a, b = node["R"], node["C"]
        f = net.branch(self.path, a, b, "force")
        x = net.state(self.path, "deformation", _scalar(self, "x0"), lambda s: s.z[a] - s.z[b])
        net.eq(self.path, "spring law", lambda s: s.z[f] - kk * s.x[x])


@phys("TranslationalDamper", "Viscous damper: force b (v_R - v_C).", title="Translational Damper", icon="b")
class TranslationalDamper(_Translational):
    params = [P("b", "1", "Damping (N s/m)")]

    def define(self, net, node, u0, y0):
        bb = _scalar(self, "b")
        a, c = node["R"], node["C"]
        f = net.branch(self.path, a, c, "force")
        net.eq(self.path, "damper law", lambda s: s.z[f] - bb * (s.z[a] - s.z[c]))


@phys("IdealForceSource", "Force given by the input signal; a positive force pushes R forward relative to C.",
      title="Ideal Force Source", icon="F(u)")
class IdealForceSource(_Translational):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        f = net.branch(self.path, node["R"], node["C"], "force")
        net.eq(self.path, "source force", lambda s: s.z[f] + s.u[u0])


@phys("IdealTranslationalVelocitySource", "Velocity v_R - v_C given by the input signal.",
      title="Ideal Translational Velocity Source", icon="v(u)")
class IdealTranslationalVelocitySource(_Translational):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        a, b = node["R"], node["C"]
        net.branch(self.path, a, b, "force")
        net.eq(self.path, "source velocity", lambda s: (s.z[a] - s.z[b]) - s.u[u0])


@phys("MechanicalTranslationalReference", "Fixed frame for translational motion: velocity zero.",
      title="Mechanical Translational Reference", icon="gnd")
class MechanicalTranslationalReference(PhysBlock):
    params = []
    ports = [("R", "translational")]

    def define(self, net, node, u0, y0):
        _reference(self, net, node["R"])


@phys("IdealTranslationalMotionSensor", "Outputs the velocity and the position of R relative to C; no force.",
      title="Ideal Translational Motion Sensor", icon="x?")
class IdealTranslationalMotionSensor(_Translational):
    params = [P("x0", "0", "Initial position (m)")]
    nout = 2

    def define(self, net, node, u0, y0):
        a, b = node["R"], node["C"]
        x = net.state(self.path, "position", _scalar(self, "x0"), lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.x[x])


@phys("IdealForceSensor", "Outputs the force flowing from R to C through it (it joins R and C rigidly).",
      title="Ideal Force Sensor", icon="F?")
class IdealForceSensor(_Translational):
    params = []
    nout = 1

    def define(self, net, node, u0, y0):
        a, b = node["R"], node["C"]
        f = net.branch(self.path, a, b, "force")
        net.eq(self.path, "rigid joint", lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.z[f])


# -- mechanical rotational --------------------------------------------------------------


@phys("Inertia", "Rotational inertia J: the torque flowing into its port accelerates it.", icon="J")
class Inertia(PhysBlock):
    params = [P("J", "1", "Inertia (kg m^2)"), P("w0", "0", "Initial angular velocity (rad/s)")]
    ports = [("I", "rotational")]

    def define(self, net, node, u0, y0):
        j = _scalar(self, "J")
        if j <= 0:
            raise BlockError("the inertia must be positive")
        a = node["I"]
        tq = net.alg(self.path, "torque")
        net.flow(a, tq, 1.0)
        k = net.state(self.path, "angular velocity", _scalar(self, "w0"), lambda s: s.z[tq] / j)
        net.eq(self.path, "inertia velocity", lambda s: s.z[a] - s.x[k])


@phys("RotationalSpring", "Torsion spring: torque k theta, theta = angle of F relative to B (a state).",
      title="Rotational Spring", icon="k")
class RotationalSpring(_Rotational):
    params = [P("k", "1", "Spring rate (N m/rad)"), P("theta0", "0", "Initial deflection (rad)")]

    def define(self, net, node, u0, y0):
        kk = _scalar(self, "k")
        a, b = node["F"], node["B"]
        tq = net.branch(self.path, a, b, "torque")
        th = net.state(self.path, "deflection", _scalar(self, "theta0"), lambda s: s.z[a] - s.z[b])
        net.eq(self.path, "spring law", lambda s: s.z[tq] - kk * s.x[th])


@phys("RotationalDamper", "Rotational viscous damper: torque b (w_F - w_B).", title="Rotational Damper", icon="b")
class RotationalDamper(_Rotational):
    params = [P("b", "1", "Damping (N m s/rad)")]

    def define(self, net, node, u0, y0):
        bb = _scalar(self, "b")
        a, c = node["F"], node["B"]
        tq = net.branch(self.path, a, c, "torque")
        net.eq(self.path, "damper law", lambda s: s.z[tq] - bb * (s.z[a] - s.z[c]))


@phys("IdealTorqueSource", "Torque given by the input signal; a positive torque turns F forward relative to B.",
      title="Ideal Torque Source", icon="T(u)")
class IdealTorqueSource(_Rotational):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        tq = net.branch(self.path, node["F"], node["B"], "torque")
        net.eq(self.path, "source torque", lambda s: s.z[tq] + s.u[u0])


@phys("IdealAngularVelocitySource", "Angular velocity w_F - w_B given by the input signal.",
      title="Ideal Angular Velocity Source", icon="w(u)")
class IdealAngularVelocitySource(_Rotational):
    params = []
    nin = 1

    def define(self, net, node, u0, y0):
        a, b = node["F"], node["B"]
        net.branch(self.path, a, b, "torque")
        net.eq(self.path, "source velocity", lambda s: (s.z[a] - s.z[b]) - s.u[u0])


@phys("MechanicalRotationalReference", "Fixed frame for rotation: angular velocity zero.",
      title="Mechanical Rotational Reference", icon="gnd")
class MechanicalRotationalReference(PhysBlock):
    params = []
    ports = [("R", "rotational")]

    def define(self, net, node, u0, y0):
        _reference(self, net, node["R"])


@phys("IdealRotationalMotionSensor", "Outputs the angular velocity and angle of F relative to B; no torque.",
      title="Ideal Rotational Motion Sensor", icon="θ?")
class IdealRotationalMotionSensor(_Rotational):
    params = [P("theta0", "0", "Initial angle (rad)")]
    nout = 2

    def define(self, net, node, u0, y0):
        a, b = node["F"], node["B"]
        th = net.state(self.path, "angle", _scalar(self, "theta0"), lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.x[th])


@phys("IdealTorqueSensor", "Outputs the torque flowing from F to B through it (it joins F and B rigidly).",
      title="Ideal Torque Sensor", icon="T?")
class IdealTorqueSensor(_Rotational):
    params = []
    nout = 1

    def define(self, net, node, u0, y0):
        a, b = node["F"], node["B"]
        tq = net.branch(self.path, a, b, "torque")
        net.eq(self.path, "rigid joint", lambda s: s.z[a] - s.z[b])
        net.output(lambda s: s.z[tq])


# -- connecting to the rest of the model ------------------------------------------------


@phys("SolverConfiguration", "Marks a physical network (accepted for familiarity; mylab needs no settings here).",
      title="Solver Configuration", icon="f(x)=0")
class SolverConfiguration(PhysBlock):
    params = []
    ports = [("p", "any")]


@phys("ConnectionPort", "Put inside a subsystem: a physical port of the subsystem (L1, L2... or R1, R2...).",
      title="Connection Port", aliases=("PMIOPort",), icon="o")
class ConnectionPort(PhysBlock):
    params = [P("Port", "1", "Port number", "text"), P("Side", "left", "Side", options=["left", "right"])]
    ports = [("x", "any")]


@block("PSSimulinkConverter", "Physical", "Passes a sensor signal on to ordinary blocks (physical signals are "
       "ordinary signals in mylab).", title="PS-Simulink Converter", aliases=("PS-Simulink Converter",), icon="PS>")
class PSSimulinkConverter(Node):
    params = [P("Unit", "1", "Output unit (kept for information; no conversion)", "text")]
    same_shape = True

    def out(self, t, u):
        return [u[0]]


@block("SimulinkPSConverter", "Physical", "Passes an ordinary signal on to a controlled source (physical signals "
       "are ordinary signals in mylab).", title="Simulink-PS Converter", aliases=("Simulink-PS Converter",),
       icon=">PS")
class SimulinkPSConverter(PSSimulinkConverter):
    params = [P("Unit", "1", "Input unit (kept for information; no conversion)", "text")]


# -------------------------------------------------------------------------------------
# The network as one block of the simulation
# -------------------------------------------------------------------------------------


class PhysicalNetwork(Node):
    """All connected physical components, as one block with continuous states."""

    sample = "cont"

    def __init__(self, path: str, members: list, ctx: Any, node_of_port: dict, node_names: list):
        super().__init__(path, {}, ctx)
        self.members = members  # [(component node, {port: node number})]
        self.node_names = node_names
        self.n_nodes = len(node_names)
        self.n_u = sum(m.n_in() for m, _ in members)
        self.n_y = sum(m.n_out() for m, _ in members)

    def n_in(self):
        return self.n_u

    def n_out(self):
        return self.n_y

    def is_df(self, port):
        return bool(self.df_mask[port]) if hasattr(self, "df_mask") and port < len(self.df_mask) else True

    def init(self):
        net = Network(self.n_nodes, self.node_names)
        u0 = y0 = 0
        for m, ports in self.members:
            try:
                m.init()
                m.define(net, ports, u0, y0)
            except BlockError as exc:
                raise BlockError(f"'{m.path}': {exc}") from None
            u0 += m.n_in()
            y0 += m.n_out()
        for n in range(self.n_nodes):  # the balance of each node
            terms = net.kcl[n]
            if terms:
                net.eq(f"the node at {self.node_names[n]}", "balance of through variables",
                       lambda s, _t=tuple(terms): sum(sg * s.z[i] for i, sg in _t))
        self.net = net
        self.nz = len(net.z_names)
        if len(net.eqs) != self.nz:
            raise BlockError(
                f"the physical network has {len(net.eqs)} equations for {self.nz} unknowns; a node is connected to "
                "nothing but sensors, or a component is not connected"
            )
        self.nx = len(net.x0)
        self.x0 = np.array(net.x0, dtype=float)
        self.z = np.zeros(self.nz)
        self.linear = not any(m.nonlinear for m, _ in self.members)
        self.lus: dict = {}
        self.major_modes = None
        self.has_zc = bool(net.zcs)
        self.cache_key = None
        self.y_now = [np.zeros(1) for _ in range(self.n_y)]
        # which inputs act on the outputs at once (the others only through the states)
        self._check(np.zeros(self.nx) + self.x0, np.zeros(self.n_u), 0.0)
        base = self._outputs(self.x0, np.zeros(self.n_u), 0.0)
        mask = []
        for k in range(self.n_u):
            uk = np.zeros(self.n_u)
            uk[k] = 1.0
            mask.append(bool(np.any(np.abs(self._outputs(self.x0, uk, 0.0) - base) > 1e-12)))
        self.df_mask = mask

    def initial_state(self):
        return self.x0

    # -- solving ----------------------------------------------------------------------
    def _residual(self, s: _S) -> np.ndarray:
        return np.array([fn(s) for fn, _o, _w in self.net.eqs], dtype=float)

    def _jac(self, s: _S, r0: np.ndarray) -> np.ndarray:
        n = self.nz
        jac = np.zeros((n, n))
        z = s.z
        for i in range(n):
            h = 1e-7 * max(1.0, abs(z[i]))
            old = z[i]
            z[i] = old + h
            jac[:, i] = (self._residual(s) - r0) / h
            z[i] = old
        return jac

    def _state(self, x, u, t) -> _S:
        s = _S()
        s.x, s.u, s.t = np.asarray(x, dtype=float), [float(v) for v in u], float(t)
        s.z = self.z.copy()
        return s

    def _check(self, x, u, t) -> None:
        """Index 1: dg/dz must be invertible. Otherwise say which components are involved."""
        s = self._state(x, u, t)
        jac = self._jac(s, self._residual(s))
        _u, sv, vt = np.linalg.svd(jac)
        if sv.size and sv[-1] > 1e-9 * max(1.0, sv[0]):
            return
        rows = np.flatnonzero(np.abs(_u[:, -1]) > 1e-6)
        cols = np.flatnonzero(np.abs(vt[-1]) > 1e-6)
        owners = []
        for r in rows:
            o = self.net.eqs[r][1]
            if o and o not in owners:
                owners.append(o)
        unknowns = [self.net.z_names[c] for c in cols][:4]
        hints = []
        whats = {self.net.eqs[r][2] for r in rows}
        if whats & {"capacitor voltage", "inductor current", "mass velocity", "inertia velocity"}:
            hints.append(
                "a state is fixed by other components (for example a capacitor directly across a voltage source or "
                "another capacitor, an inductor in series with a current source, two masses joined rigidly or a "
                "mass moved by a velocity source); mylab solves index-1 networks only: put a resistor, spring or "
                "damper between them"
            )
        elif any(n.startswith("across variable") for n in unknowns) and not any(
            w == "reference" for w in whats
        ):
            hints.append("nothing fixes the level of the across variable: add a reference (ground) block")
        raise BlockError(
            "the equations of the physical network cannot be solved for its variables ("
            + "; ".join(hints or ["the network is not index 1, or a variable is not determined"])
            + "). Components involved: " + ", ".join(f"'{o}'" for o in owners[:8])
            + (". Undetermined: " + "; ".join(unknowns) if unknowns else "")
        )

    def solve(self, x, u, t) -> _S:
        """The algebraic unknowns for states x and inputs u. Switches (diodes) are latched between major
        steps while the solver locates their switching; otherwise they are made consistent by trying again."""
        net = self.net
        sim = self.sim
        if not net.modes:
            return self._solve(x, u, t)
        if sim is not None and sim.minor and sim.latch and self.major_modes is not None:
            net.modes[:] = self.major_modes
            return self._solve(x, u, t)
        for _ in range(2 * len(net.modes) + 4):
            s = self._solve(x, u, t)
            new = [bool(test(s)) for test in net.mode_tests]
            if new == net.modes:
                break
            net.modes[:] = new
        if sim is None or not sim.minor:
            self.major_modes = list(net.modes)
        return s

    def _solve(self, x, u, t) -> _S:
        key = (float(t), tuple(np.asarray(x, dtype=float).tolist()), tuple(float(v) for v in u), tuple(self.net.modes))
        if key == self.cache_key:
            return self.cache_s
        s = self._state(x, u, t)
        if self.linear:
            lu = self.lus.get(key[3])
            if lu is None:
                from scipy.linalg import lu_factor

                s.z = np.zeros(self.nz)
                jac = self._jac(s, self._residual(s))
                if np.linalg.matrix_rank(jac) < self.nz:  # a switch made the network singular
                    self._check(x, u, t)
                lu = self.lus[key[3]] = lu_factor(jac)
            from scipy.linalg import lu_solve

            r = self._residual(s)
            s.z = s.z - lu_solve(lu, r)
        else:
            for _ in range(60):
                r = self._residual(s)
                if np.max(np.abs(r), initial=0.0) < 1e-11 * (1.0 + np.max(np.abs(s.z), initial=0.0)):
                    break
                jac = self._jac(s, r)
                try:
                    dz = np.linalg.solve(jac, -r)
                except np.linalg.LinAlgError:
                    raise BlockError(f"the physical network has no unique solution at t = {t:g}") from None
                lam = 1.0
                base = np.max(np.abs(r))
                z0 = s.z.copy()
                while lam > 1e-6:
                    s.z = z0 + lam * dz
                    if np.max(np.abs(self._residual(s))) < base:
                        break
                    lam *= 0.5
            else:
                raise BlockError(f"the equations of the physical network did not converge at t = {t:g}")
        self.z = s.z.copy()
        self.cache_key, self.cache_s = key, s
        return s

    def _outputs(self, x, u, t) -> np.ndarray:
        s = self.solve(x, u, t)
        return np.array([fn(s) for fn in self.net.outs], dtype=float)

    def _inputs(self, u: list) -> list:
        vals = []
        for k, v in enumerate(u):
            a = np.ravel(v)
            if a.size != 1:
                raise BlockError(f"input {k + 1} of the physical network has {a.size} elements; one is needed")
            vals.append(float(a[0]))
        return vals

    def out(self, t, u):
        y = self._outputs(self.x, self._inputs(u), t)
        return [np.array([v]) for v in y]

    def der(self, t, u):
        s = self.solve(self.x, self._inputs(u), t)
        return np.array([fn(s) for fn in self.net.ders], dtype=float)

    def zc(self, t, u):
        if not self.net.zcs:
            return []
        s = self.solve(self.x, self._inputs(u), t)
        return [fn(s) for fn in self.net.zcs]


# -------------------------------------------------------------------------------------
# Finding the networks of a model
# -------------------------------------------------------------------------------------


def _a(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def networks(comp: Any, systems: list, phys_blocks: list, ctx: Any) -> list:
    """Group the physical blocks into connected networks. Returns [(PhysicalNetwork, [(sys, block)])]."""
    parent: dict = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    from .engine import _physical_end
    from .library import find_type

    domain: dict = {}
    for sy, b, cls in phys_blocks:
        for name, dom in cls.ports:
            key = (id(sy), b.name, name)
            find(key)
            domain[key] = dom
    for sy in systems:
        for ln in sy.system.lines:
            if _physical_end(ln.src) and _physical_end(ln.dst):
                union((id(sy), ln.src[0], ln.src[1]), (id(sy), ln.dst[0], ln.dst[1]))
            elif _physical_end(ln.src) != _physical_end(ln.dst):
                sb = sy.system.block(ln.src[0] if _physical_end(ln.src) else ln.dst[0])
                if sb is not None and getattr(find_type(sb.type), "physical", False):
                    comp.problem(
                        "error",
                        f"A line joins a physical port of '{sy.path(sb.name)}' with a signal port; physical ports "
                        "connect only to physical ports (sensors and controlled sources have signal ports for that).",
                        sy.path(sb.name),
                        kind="line",
                    )
        # connection ports: a port inside a subsystem is the subsystem's port L<k> or R<k> outside
        if sy.parent is not None and sy.block is not None:
            for b in sy.system.blocks:
                if getattr(find_type(b.type), "type", "") == "ConnectionPort":
                    side = "R" if str(b.params.get("Side", "left")).strip().lower().startswith("r") else "L"
                    k = str(b.params.get("Port", "1")).strip() or "1"
                    union((id(sy), b.name, "x"), (id(sy.parent), sy.block.name, f"{side}{k}"))
    # blocks that share a node are in the same network (a second grouping, over the nodes)
    group: dict = {}

    def gfind(a):
        group.setdefault(a, a)
        while group[a] != a:
            group[a] = group[group[a]]
            a = group[a]
        return a

    items_all = []
    for sy, b, cls in phys_blocks:
        if getattr(cls, "type", "") in ("ConnectionPort", "SolverConfiguration"):
            continue
        roots = [find((id(sy), b.name, name)) for name, _d in cls.ports]
        for r in roots[1:]:
            ra, rb = gfind(roots[0]), gfind(r)
            if ra != rb:
                group[ra] = rb
        items_all.append((sy, b, cls, roots))
    by_group: dict = {}
    for item in items_all:
        by_group.setdefault(gfind(item[3][0]), []).append(item)
    merged = list(by_group.values())
    out = []
    for k, items in enumerate(g for g in merged if g):
        roots = []
        for _sy, _b, _cls, rs in items:
            for r in sorted(rs, key=str):
                if r not in roots:
                    roots.append(r)
        number = {r: i for i, r in enumerate(roots)}
        names = []
        for r in roots:
            sy_id, bname, port = r
            sy = next(s for s in systems if id(s) == sy_id)
            names.append(f"{sy.path(bname)}.{port}")
        members = []
        sy_b = []
        wiring = []
        doms: dict = {}
        for sy, b, cls, _rs in items:
            node = cls(sy.path(b.name), b.params, sy.ctx)
            node.block = b  # type: ignore[attr-defined]
            ports = {}
            for name, dom in cls.ports:
                r = find((id(sy), b.name, name))
                ports[name] = number[r]
                if dom != "any":
                    if doms.setdefault(r, dom) != dom:
                        comp.problem(
                            "error",
                            f"'{sy.path(b.name)}': its port {name} ({dom}) is connected to "
                            f"{_a(doms[r])} {doms[r]} port; "
                            "ports of different physical domains cannot be joined.",
                            sy.path(b.name),
                            kind="line",
                        )
            members.append((node, ports))
            sy_b.append((sy, b))
            for j in range(node.n_in()):
                wiring.append((sy, b.name, j + 1))
        first = sy_b[0][0].path(sy_b[0][1].name)
        net = PhysicalNetwork(f"physical network {k + 1} (with '{first}')", members, ctx, number, names)
        net.wiring = wiring  # type: ignore[attr-defined]
        net.conds = []
        out.append((net, sy_b))
    return out
