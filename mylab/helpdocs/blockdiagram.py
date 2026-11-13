"""Help text for mylab Blocks: building and simulating block-diagram models from code."""

_PATH = (
    "Blocks are named by path: 'model/Block', or 'model/Subsystem/Block' inside a subsystem. Library names such as "
    "'blocks/Gain' work, and so do the familiar ones like 'simulink/Continuous/Integrator' and 'built-in/Gain'."
)

HELP = {
    "new_system": (
        ["new_system(name)", "h = new_system(name)"],
        "Creates an empty block-diagram model in memory and makes it the current model. Add blocks with "
        "add_block, connect them with add_line, run it with sim, and write it to name.mlb.json with save_system.",
    ),
    "open_system": (
        "open_system(name)",
        "Loads a model (from name.mlb.json or name.slx in the current folder or on the path) and makes it the "
        "current model. In the mylab web app the model also opens in the diagram editor; elsewhere a note is "
        "printed once and the model can be used from code.",
    ),
    "load_system": (
        ["load_system(name)", "h = load_system(name)"],
        "Loads a model file into memory without showing it. Models are also loaded automatically the first "
        "time their name is used, so sim('mymodel') works when mymodel.mlb.json is in the current folder. "
        "A .slx file is converted on loading; blocks that have no equivalent are listed in a warning.",
    ),
    "save_system": (
        ["save_system(name)", "save_system(name, newname)", "file = save_system(...)"],
        "Writes a model to name.mlb.json, a readable JSON file, in the current folder (or back to the file it "
        "was loaded from). With a second name the model is renamed and saved under that name. mylab does not "
        "write .slx files.",
    ),
    "close_system": (
        ["close_system(name)", "close_system(name, 1)", "bdclose('all')"],
        "Removes a model from memory. Changes are discarded unless the second argument is 1, which saves the "
        "model first. bdclose('all') closes every model.",
    ),
    "bdclose": ("bdclose('all')", "Closes block-diagram models; the same as close_system."),
    "bdIsLoaded": ("tf = bdIsLoaded(name)", "True when a model with this name is in memory."),
    "bdroot": (
        ["name = bdroot", "name = bdroot(path)"],
        "Name of the current model, or of the model that a block path or handle belongs to.",
    ),
    "gcs": ("name = gcs", "Name of the current model: the one most recently created, loaded or changed."),
    "gcb": ("path = gcb", "Path of the block most recently added with add_block."),
    "add_block": (
        [
            "add_block(source, destination)",
            "add_block(source, destination, 'Param', value, ...)",
            "h = add_block(..., 'MakeNameUnique', 'on')",
        ],
        "Adds a block to a model. source names a block type in the library, destination is the path of the new "
        "block, and the remaining pairs set its parameters. Parameter values are text holding a MATLAB "
        "expression ('2*K'); numbers are accepted too. source may also be the path of an existing block to "
        "copy. " + _PATH + " Example: add_block('blocks/Gain', 'm/K', 'Gain', '2.5').",
    ),
    "add_line": (
        [
            "add_line(system, 'Src/port', 'Dst/port')",
            "add_line(system, outHandle, inHandle)",
            "h = add_line(system, 'Src/1', {'A/1', 'B/2'}, 'autorouting', 'on')",
        ],
        "Connects an output port to an input port of blocks in the same system. Ports are written as "
        "'BlockName/number', or given as the handles from get_param(block, 'PortHandles'). The control inputs "
        "of conditionally run subsystems are 'Name/Enable' and 'Name/Trigger'. One output may feed several "
        "inputs (call add_line again, or pass a cell array of destinations); an input takes one line. "
        "'autorouting' is accepted: the editor always routes lines by itself.",
    ),
    "delete_block": ("delete_block(path)", "Removes a block and the lines attached to it."),
    "delete_line": (
        ["delete_line(system, 'Src/port', 'Dst/port')", "delete_line(h)"],
        "Removes the line between two ports (or the line with handle h).",
    ),
    "set_param": (
        ["set_param(path, 'Param', value, ...)", "set_param(model, 'StopTime', '20', 'Solver', 'ode4')"],
        "Changes parameters of a block, or settings of a model. Block values are text holding a MATLAB "
        "expression that is evaluated when the model runs; numbers are converted to text. Besides its own "
        "parameters every block has Name, Position ([left top right bottom]) and Orientation. Model settings: "
        "StartTime, StopTime, Solver, MaxStep, RelTol, AbsTol, FixedStep, ZeroCrossings. An unknown "
        "parameter name is an error that lists the names the block has.",
    ),
    "get_param": (
        [
            "value = get_param(path, 'Param')",
            "names = get_param(model, 'Blocks')",
            "ph = get_param(block, 'PortHandles')",
        ],
        "Reads a parameter of a block or a setting of a model, as text. Also available for blocks: Name, "
        "BlockType, Parent, Position, Orientation, Ports ([inputs outputs]), Handle, PortHandles (a struct with "
        "the fields Inport and Outport for add_line) and DialogParameters; for models and subsystems: Blocks "
        "and Lines.",
    ),
    "find_system": (
        [
            "paths = find_system(model)",
            "paths = find_system(model, 'BlockType', 'Gain')",
            "paths = find_system(model, 'SearchDepth', 1, 'Name', 'K')",
        ],
        "Lists the model and the paths of its blocks as a cell array, including blocks inside subsystems. "
        "Name, value pairs keep only blocks whose parameter has that value (BlockType, Name, Parent or any "
        "block parameter); 'SearchDepth', n limits how deep subsystems are searched.",
    ),
    "sim": (
        [
            "out = sim(model)",
            "out = sim(model, 'StopTime', '10', 'Solver', 'ode45', ...)",
            "out = sim(model, tfinal)",
            "[t, x, y] = sim(model, [t0 tf], options)",
        ],
        "Simulates a block-diagram model over time. out.tout holds the times of the solver's steps and out.yout "
        "the signals of the top-level Out blocks, one column per element. Each To Workspace block adds a field "
        "named after its variable: a struct with Time and Data (or a plain array, depending on the block's format). "
        "out.scopes has one such struct per Scope, out.logsout one per logged line. Settings given as name, value "
        "pairs (or a struct, or options from simset) apply to this run only: StartTime, StopTime, Solver, MaxStep, "
        "RelTol, AbsTol, FixedStep, ZeroCrossings, ExternalInput ([t u1 u2 ...] for the top-level In blocks). "
        "Variable-step solvers: ode45 (default), ode23, ode113, ode15s and ode23s for stiff models. Fixed-step: ode1 "
        "(Euler), ode2, ode3, ode4 (Runge-Kutta), ode5. Block parameters are looked up in the base workspace, then "
        "in the workspace of the function that calls sim. Called without an output, sim stores out, tout, yout and "
        "the To Workspace variables in the workspace. Errors name the block they come from.",
    ),
    "Simulink.Variant": (
        "v = Simulink.Variant('MODE == 2')",
        "A variant control for imported block diagrams: a struct with the field Condition. A variant subsystem "
        "whose choice names v as its control runs that choice when the condition is true in the workspace.",
    ),
    "simset": (
        ["options = simset('Solver', 'ode4', 'FixedStep', 0.01)", "options = simset(options, name, value, ...)"],
        "Makes an options struct for sim(model, tspan, options). The fields are the settings sim accepts.",
    ),
    "simget": (
        ["settings = simget(model)", "value = simget(model, name)"],
        "The simulation settings stored in a model, as a struct of text values, or one of them.",
    ),
    "linmod": (
        ["[A, B, C, D] = linmod(model)", "[A, B, C, D] = linmod(model, x0, u0)", "sys = linmod(model)"],
        "Linearizes a model around an operating point: dx/dt = A x + B u, y = C x + D u. The inputs u are the "
        "top-level In blocks, the outputs y the top-level Out blocks, and x the continuous states in the order of "
        "the blocks. The matrices are computed by central differences at the states x0 (default: the initial "
        "values) and inputs u0 (default: zero). Discrete states are held at their initial values. With one "
        "output the result is a struct with the fields a, b, c, d and StateName.",
    ),
    "linearize": (
        "[A, B, C, D] = linearize(model, x0, u0)",
        "The same as linmod: the matrices A, B, C, D of the model around an operating point.",
    ),
    "linmod2": ("[A, B, C, D] = linmod2(model, x0, u0)", "The same as linmod (both use central differences)."),
    "trim": (
        ["[x, u, y, dx] = trim(model)", "[x, u, y, dx] = trim(model, x0, u0)"],
        "Finds a steady state of a model: states x and inputs u for which the derivatives dx are zero, starting "
        "the search (least squares) from the guess x0, u0. y holds the outputs there. Check dx: when no steady "
        "state exists near the guess, the result is only the closest point found.",
    ),
}
