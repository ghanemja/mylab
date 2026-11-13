"""Help text for rotations, rigid transforms, angle helpers and code-generation directives."""

_SEQ = (
    " seq is the axis order 'ZYX' (default), 'ZYZ' or 'XYZ'; the angles are in radians and listed in that order. "
    "Rotation matrices multiply column vectors."
)
_Q = " A quaternion is a row [w x y z] with the scalar part first."
_T = " A transform is a 4-by-4 matrix [R t; 0 0 0 1]."

HELP = {
    "angdiff": (
        ["d = angdiff(a, b)", "d = angdiff(a)"],
        "Difference b - a between angles in radians, wrapped to the range from -pi to pi. With one input, the "
        "differences between neighbouring elements of a vector.",
    ),
    "eul2rotm": (
        ["R = eul2rotm(eul)", "R = eul2rotm(eul, seq)"],
        "Rotation matrix from Euler angles (one row of three)." + _SEQ,
    ),
    "rotm2eul": (
        ["eul = rotm2eul(R)", "eul = rotm2eul(R, seq)"],
        "Euler angles of a 3-by-3 rotation matrix, as a row."
        + _SEQ
        + " At the singular orientation the first angle is set to 0.",
    ),
    "eul2tform": (
        ["T = eul2tform(eul)", "T = eul2tform(eul, seq)"],
        "4-by-4 transform with the rotation given by Euler angles and no translation." + _SEQ,
    ),
    "tform2eul": (
        ["eul = tform2eul(T)", "eul = tform2eul(T, seq)"],
        "Euler angles of the rotation part of a 4-by-4 transform." + _SEQ,
    ),
    "eul2quat": (["q = eul2quat(eul)", "q = eul2quat(eul, seq)"], "Quaternion from Euler angles." + _Q + _SEQ),
    "quat2eul": (["eul = quat2eul(q)", "eul = quat2eul(q, seq)"], "Euler angles of a quaternion." + _Q + _SEQ),
    "quat2rotm": ("R = quat2rotm(q)", "Rotation matrix of a quaternion; the quaternion is normalized first." + _Q),
    "rotm2quat": (
        "q = rotm2quat(R)",
        "Quaternion of a 3-by-3 rotation matrix, with a scalar part that is not negative." + _Q,
    ),
    "quat2tform": ("T = quat2tform(q)", "4-by-4 transform with the rotation of a quaternion and no translation." + _Q),
    "tform2quat": ("q = tform2quat(T)", "Quaternion of the rotation part of a 4-by-4 transform." + _Q),
    "rotm2tform": (
        "T = rotm2tform(R)",
        "Puts a 3-by-3 rotation matrix into a 4-by-4 transform with no translation." + _T,
    ),
    "tform2rotm": ("R = tform2rotm(T)", "The 3-by-3 rotation part of a 4-by-4 transform." + _T),
    "trvec2tform": ("T = trvec2tform(t)", "4-by-4 transform that only shifts by the vector t = [x y z]." + _T),
    "tform2trvec": ("t = tform2trvec(T)", "The translation [x y z] of a 4-by-4 transform, as a row." + _T),
    "axang2rotm": (
        "R = axang2rotm([x y z angle])",
        "Rotation matrix for a turn by angle (radians) about the axis [x y z]. The axis need not have length 1.",
    ),
    "rotm2axang": (
        "axang = rotm2axang(R)",
        "Axis and angle [x y z angle] of a rotation matrix; the angle is between 0 and pi. For no rotation the axis is [0 0 1].",  # noqa: E501
    ),
    "rotx": ("R = rotx(deg)", "3-by-3 matrix for a rotation by deg degrees about the x axis."),
    "roty": ("R = roty(deg)", "3-by-3 matrix for a rotation by deg degrees about the y axis."),
    "rotz": ("R = rotz(deg)", "3-by-3 matrix for a rotation by deg degrees about the z axis."),
    "coder.extrinsic": (
        "coder.extrinsic('name', ...)",
        "A code-generation directive. mylab runs code directly, so this does nothing.",
    ),
    "coder.inline": ("coder.inline('always')", "A code-generation directive; does nothing in mylab."),
    "coder.varsize": ("coder.varsize('x', dims)", "A code-generation directive; does nothing in mylab."),
    "coder.cinclude": ("coder.cinclude('file.h')", "A code-generation directive; does nothing in mylab."),
    "coder.updateBuildInfo": ("coder.updateBuildInfo(...)", "A code-generation directive; does nothing in mylab."),
    "coder.columnMajor": ("coder.columnMajor", "A code-generation directive; does nothing in mylab."),
    "coder.const": ("y = coder.const(x)", "Returns x. (For code generation it marks a value as constant.)"),
    "coder.nullcopy": ("y = coder.nullcopy(x)", "Returns x. (For code generation it skips an initialization.)"),
    "coder.ignoreConst": ("y = coder.ignoreConst(x)", "Returns x; a code-generation hint."),
    "coder.ignoreSize": ("y = coder.ignoreSize(x)", "Returns x; a code-generation hint."),
    "coder.target": (
        ["tf = coder.target('MATLAB')", "name = coder.target"],
        "True for 'MATLAB' (the code is being run, not turned into C) and false for every other target name. Without an input: ''.",  # noqa: E501
    ),
    "coder.ceval": (
        "coder.ceval('cfunction', ...)",
        "Calls compiled C code, which mylab cannot do: it raises an error. Guard it with if ~coder.target('MATLAB').",
    ),
    "simsizes": (
        ["sizes = simsizes", "sys = simsizes(sizes)"],
        "For level-1 S-functions used in block diagrams. Without an input: a struct with the fields NumContStates, "
        "NumDiscStates, NumOutputs, NumInputs, DirFeedthrough and NumSampleTimes, all 0, to fill in. With that struct: "
        "the vector the S-function returns for flag 0.",
    ),
}
