"""Stand-ins for common library blocks that are not single mylab blocks.

Each builder returns a subsystem made of ordinary mylab blocks that does
what the named library block is publicly known to do. They were written
from the equations of the models (textbook kinematics), not from the
original library.
"""

from __future__ import annotations

from .model import Block, Line, System


def _sub(name: str, blocks: list, lines: list, mask: dict) -> Block:
    sy = System()
    for b in blocks:
        sy.blocks.append(Block(b[0], b[1], b[2], b[3]))
    for text in lines:
        a, _, b2 = text.partition("->")
        s, d = a.strip().rsplit("/", 1), b2.strip().rsplit("/", 1)
        sy.lines.append(Line((s[0], s[1]), (d[0], d[1])))
    blk = Block(name, "Subsystem", {}, [0, 0, 120, 70], system=sy)
    blk.extra["mask"] = mask
    return blk


def kinematic_model(kind: str, name: str, p: dict) -> Block | None:
    """Bicycle, unicycle, differential-drive or Ackermann vehicle: inputs -> state [x y theta ...] and its derivative."""  # noqa: E501
    inputs = p.get("VehicleInputs", "").lower()
    mask = {
        "L": p.get("WheelBase", "1"),
        "r": p.get("WheelRadius", "0.1"),
        "w": p.get("TrackWidth", "0.2"),
        "vrange": p.get("VehicleSpeedRange", p.get("WheelSpeedRange", "[-inf inf]")),
        "smax": p.get("MaxSteeringAngle", "pi/4"),
        "x0": p.get("InitialState", "[0; 0; 0]"),
    }
    lim2 = ("inf", "-inf")
    n = 3
    integ = {"InitialCondition": "reshape(x0, 1, [])"}
    th = "u(5)"
    if kind == "bicycle":
        speed = "u(1)"
        if "angular" in inputs:
            turn = "u(2)"
        else:
            turn, lim2 = "u(1)/L*tan(u(2))", ("smax", "-smax")
    elif kind == "unicycle":
        speed, turn = ("r*u(1)" if "wheel" in inputs else "u(1)"), "u(2)"
        if "wheel" in inputs:
            mask["vrange"] = p.get("WheelSpeedRange", "[-inf inf]")
    elif kind == "differentialdrive":
        if "vehicle speed" in inputs:
            speed, turn = "u(1)", "u(2)"
        else:
            speed, turn = "r*(u(1)+u(2))/2", "r*(u(2)-u(1))/w"
            mask["vrange"] = p.get("WheelSpeedRange", "[-inf inf]")
            lim2 = ("max(vrange)", "min(vrange)")
    elif kind == "ackermann":
        n, speed, turn = 4, "u(1)", "u(1)/L*tan(u(6))"
        integ.update(
            LimitOutput="on", UpperSaturationLimit="[inf inf inf smax]", LowerSaturationLimit="[-inf -inf -inf -smax]"
        )
    else:
        return None
    rows = [("dx", f"({speed})*cos({th})"), ("dy", f"({speed})*sin({th})"), ("dtheta", turn)]
    if n == 4:
        rows.append(("dsteer", "u(2)"))
    blocks = [
        ("In1", "Inport", {"Port": "1"}, [20, 40, 40, 24]),
        ("In2", "Inport", {"Port": "2"}, [20, 110, 40, 24]),
        ("Limit 1", "Saturation", {"UpperLimit": "max(vrange)", "LowerLimit": "min(vrange)"}, [90, 32, 60, 40]),
        ("Limit 2", "Saturation", {"UpperLimit": lim2[0], "LowerLimit": lim2[1]}, [90, 102, 60, 40]),
        ("Join", "Mux", {"Inputs": "3"}, [190, 30, 10, 130]),
        ("Rates", "Mux", {"Inputs": str(n)}, [420, 30, 10, 40 * n]),
        ("State", "Integrator", integ, [470, 60, 60, 44]),
        ("state", "Outport", {"Port": "1"}, [590, 70, 40, 24]),
        ("stateDot", "Outport", {"Port": "2"}, [590, 150, 40, 24]),
    ]
    lines = [
        "In1/1 -> Limit 1/1",
        "In2/1 -> Limit 2/1",
        "Limit 1/1 -> Join/1",
        "Limit 2/1 -> Join/2",
        "State/1 -> Join/3",
        "Rates/1 -> State/1",
        "State/1 -> state/1",
        "Rates/1 -> stateDot/1",
    ]
    for k, (nm, expr) in enumerate(rows):
        blocks.append((nm, "Fcn", {"Expr": expr}, [240, 30 + 46 * k, 150, 36]))
        lines += [f"Join/1 -> {nm}/1", f"{nm}/1 -> Rates/{k + 1}"]
    return _sub(name, blocks, lines, mask)


_REP = {
    "homogeneoustransformation": "tform",
    "eulerangles": "eul",
    "translationvector": "trvec",
    "quaternion": "quat",
    "rotationmatrix": "rotm",
    "axisangle": "axang",
}
_HAVE = {
    "eul2rotm",
    "eul2tform",
    "eul2quat",
    "rotm2eul",
    "tform2eul",
    "quat2eul",
    "quat2rotm",
    "quat2tform",
    "rotm2quat",
    "tform2quat",
    "rotm2tform",
    "tform2rotm",
    "trvec2tform",
    "tform2trvec",
    "axang2rotm",
    "rotm2axang",
}


def coordinate_conversion(p: dict) -> str | None:
    """Expression in u for a conversion between rotation and transform representations, or None."""
    key = lambda s: "".join(c for c in s.lower() if c.isalnum())  # noqa: E731
    a, b = _REP.get(key(p.get("InputRepresentation", ""))), _REP.get(key(p.get("OutputRepresentation", "")))
    if not a or not b or f"{a}2{b}" not in _HAVE:
        return None
    arg = "u" if a in ("tform", "rotm") else "u(:)'"
    seq = p.get("InputEulerSequence" if a == "eul" else "OutputEulerSequence", "ZYX").strip() or "ZYX"
    extra = f", '{seq}'" if "eul" in (a, b) else ""
    return f"{a}2{b}({arg}{extra})"


KINEMATIC = {
    "bicyclekinematicmodel": "bicycle",
    "unicyclekinematicmodel": "unicycle",
    "differentialdrivekinematicmodel": "differentialdrive",
    "ackermannkinematicmodel": "ackermann",
}
