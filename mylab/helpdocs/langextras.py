"""Help text for onCleanup, inputParser, RandStream, timer, message, events and small environment functions."""

_NOOP = "Accepted so that scripts written for the MATLAB desktop keep running; it does nothing in mylab."
_PARSER = "p = inputParser; addRequired(p, 'n'); addParameter(p, 'Tol', 1e-3); parse(p, varargin{:}); p.Results"

HELP = {
    # -- cleanup ------------------------------------------------------------------------------------
    "onCleanup": ("c = onCleanup(@() cleanupCode)",
                  "Runs the given function (no inputs) when the variable c goes away: when the function "
                  "that created it returns or stops with an error, or when c is cleared. Typical use: "
                  "fid = fopen(name); c = onCleanup(@() fclose(fid)); The function does not run when c is "
                  "merely overwritten, and objects stored inside cells or structs are not tracked. An "
                  "error inside the cleanup function is reported as a warning."),

    # -- inputParser --------------------------------------------------------------------------------
    "inputParser": ("p = inputParser",
                    "Creates an object that describes and checks the inputs of a function. Declare "
                    "arguments with addRequired, addOptional and addParameter, then call parse(p, "
                    "varargin{:}). Afterwards p.Results is a struct with one field per argument (fields in "
                    "alphabetical order), p.UsingDefaults lists the arguments that were not given and "
                    "p.Unmatched holds unknown name-value pairs when p.KeepUnmatched is true. Other "
                    "settings: FunctionName (used in error messages), CaseSensitive (default false), "
                    "PartialMatching (default true), StructExpand (default true). Example: " + _PARSER),
    "addRequired": (["addRequired(p, name)", "addRequired(p, name, validator)"],
                    "Adds a required positional argument to the inputParser p. validator is a function "
                    "handle that must return true (or not raise an error) for acceptable values, e.g. "
                    "@isnumeric or @(x) x > 0."),
    "addOptional": (["addOptional(p, name, default)", "addOptional(p, name, default, validator)"],
                    "Adds an optional positional argument with a default value to the inputParser p. "
                    "Optional arguments come after the required ones. A text value that is exactly the "
                    "name of a parameter and starts a complete list of name-value pairs is read as that "
                    "parameter name, not as the optional value."),
    "addParameter": (["addParameter(p, name, default)", "addParameter(p, name, default, validator)"],
                     "Adds a name-value argument with a default value to the inputParser p. Callers pass "
                     "it as 'Name', value after the positional arguments, or as a field of a struct. "
                     "Names match without regard to case and may be shortened as long as they stay unique."),
    "addParamValue": (["addParamValue(p, name, default)", "addParamValue(p, name, default, validator)"],
                      "Older name of addParameter: adds a name-value argument with a default value to the "
                      "inputParser p."),
    "addSwitch": ("addSwitch(p, name)",
                  "Adds a switch to the inputParser p: p.Results.name is true when the caller passes the "
                  "name on its own, and false otherwise."),
    "parse": (["parse(p, arg1, arg2, ...)", "p.parse(varargin{:})"],
              "Checks the given arguments against the declarations of the inputParser p and fills "
              "p.Results, p.Unmatched and p.UsingDefaults. Raises an error that names the argument when "
              "a required argument is missing, a validator rejects a value or a parameter name is "
              "unknown. Example: " + _PARSER),

    # -- random streams -------------------------------------------------------------------------------
    "RandStream": (["s = RandStream(type)", "s = RandStream(type, 'Seed', seed)"],
                   "Creates a random number stream. Use it with rand(s, ...), randn(s, ...), randi(s, ...) "
                   "and randperm(s, n), or make it the source for rand/randn/randi with "
                   "RandStream.setGlobalStream(s). reset(s) starts the sequence again. Properties: Type, "
                   "Seed, State. Every generator type ('mt19937ar', 'mlfg6331_64', ...) is accepted, but "
                   "all of them use NumPy's default generator, so the numbers differ from MATLAB's. "
                   "'Seed' may be 'shuffle'."),
    "RandStream.create": ("s = RandStream.create(type, 'Seed', seed)",
                          "Creates a random number stream; the same as RandStream(type, 'Seed', seed)."),
    "RandStream.setGlobalStream": ("old = RandStream.setGlobalStream(s)",
                                   "Makes s the stream that rand, randn, randi and randperm draw from, and "
                                   "returns the previous one. Example: "
                                   "RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 1))."),
    "RandStream.setDefaultStream": ("old = RandStream.setDefaultStream(s)",
                                    "Older name of RandStream.setGlobalStream."),
    "RandStream.getGlobalStream": ("s = RandStream.getGlobalStream",
                                   "Returns the stream that rand, randn and randi currently draw from."),
    "RandStream.getDefaultStream": ("s = RandStream.getDefaultStream",
                                    "Older name of RandStream.getGlobalStream."),
    "RandStream.list": ("RandStream.list",
                        "Lists the generator type names RandStream accepts. They all use the same "
                        "generator in mylab."),
    "rng": (["rng(seed)", "rng('default')", "rng('shuffle')", "rng(seed, generator)", "s = rng", "rng(s)"],
            "Seeds the random number generator so that rand, randn, randi and randperm give repeatable "
            "results. s = rng returns the current settings (fields Type, Seed and State), and rng(s) "
            "restores them, so the same numbers come again. Generator names such as 'twister' are "
            "accepted. The sequences differ from MATLAB's for the same seed."),

    # -- timer ----------------------------------------------------------------------------------------
    "timer": (["t = timer", "t = timer('TimerFcn', fcn, 'Period', p, ...)"],
              "Creates a timer object. Properties: TimerFcn, StartFcn, StopFcn, ErrorFcn (function handles "
              "called as fcn(t, event), text to evaluate, or a cell {fcn, extra...}), ExecutionMode "
              "('singleShot', 'fixedRate', 'fixedDelay', 'fixedSpacing'), Period, StartDelay, "
              "TasksToExecute, TasksExecuted, Running, Name, Tag, UserData. mylab has no background "
              "thread: start(t) runs the callbacks at once, one after the other, without waiting for "
              "Period or StartDelay. A timer with a limited TasksToExecute (or 'singleShot') finishes "
              "inside start. An endless timer runs once in start and keeps running inside wait(t) until "
              "a callback calls stop(t)."),
    "start": ("start(t)",
              "Starts the timer t: runs StartFcn, then TimerFcn (see timer for how mylab emulates timers)."),
    "stop": ("stop(t)", "Stops the timer t and runs its StopFcn."),
    "wait": ("wait(t)",
             "Blocks until the timer t has stopped. For an endless timer this is where the remaining "
             "TimerFcn calls happen; it gives up with a warning after 100000 calls."),
    "timerfind": ("timerfind", "Returns an empty array: mylab keeps no list of timers."),
    "timerfindall": ("timerfindall", "Returns an empty array: mylab keeps no list of timers."),
    "step": (["y = step(obj, x)", "y = obj.step(x)"],
             "Runs one step of a System object: calls the object's stepImpl method (and setupImpl before "
             "the first step). Step responses of control-system models are not supported."),

    # -- messages and errors ---------------------------------------------------------------------------
    "message": ("msg = message(id, arg1, ...)",
                "Creates a message object from a message identifier such as 'MyPkg:myId'. mylab has no "
                "message catalog, so getString(msg) returns the identifier followed by the arguments. "
                "error(message(id)) raises an error with that identifier."),
    "getString": ("s = getString(msg)", "Returns the text of a message object created with message."),
    "getReport": (["s = getReport(err)", "s = getReport(err, 'basic')"],
                  "Returns the text of an error caught with try/catch, including the function and line "
                  "where it happened. 'basic' gives the message only. Example: try, f(); catch err, "
                  "disp(getReport(err)), end"),
    "throwAsCaller": ("throwAsCaller(err)", "Raises the caught error err again; the same as rethrow(err)."),

    # -- containers.Map in function form ----------------------------------------------------------------
    "isKey": ("tf = isKey(m, key)",
              "True when the containers.Map m has the key. With a cell array of keys, returns a logical "
              "array. The same as m.isKey(key)."),
    "keys": ("k = keys(m)", "Returns the keys of the containers.Map m as a cell array, in sorted order."),
    "values": (["v = values(m)", "v = values(m, keys)"],
               "Returns the values of the containers.Map m as a cell array, in the order of its sorted "
               "keys, or the values for the given cell array of keys."),
    "remove": ("remove(m, key)", "Removes a key (or a cell array of keys) and its value from the "
                                 "containers.Map m."),

    # -- events ---------------------------------------------------------------------------------------
    "notify": (["notify(obj, 'EventName')", "notify(obj, 'EventName', eventData)"],
               "Fires an event of the handle object obj: every listener added with addlistener is called "
               "as callback(obj, eventData). The event must be declared in an events block of the class. "
               "Without eventData the callback gets an object with the properties Source and EventName."),
    "addlistener": (["L = addlistener(obj, 'EventName', @callback)",
                     "L = addlistener(obj, 'PropName', 'PostSet', @callback)"],
                    "Calls callback(source, eventData) whenever obj fires the event with notify. The "
                    "second form runs the callback after ('PostSet') or before ('PreSet') the property is "
                    "assigned; eventData.AffectedObject is the object. Set L.Enabled = false or call "
                    "delete(L) to stop listening."),
    "listener": ("L = listener(obj, 'EventName', @callback)",
                 "Like addlistener: calls callback(source, eventData) when obj fires the event. Keep L; "
                 "delete(L) stops listening."),
    "event.listener": ("L = event.listener(obj, 'EventName', @callback)", "The same as listener."),
    "event.proplistener": ("L = event.proplistener(obj, 'PropName', 'PostSet', @callback)",
                           "The same as addlistener(obj, 'PropName', 'PostSet', @callback)."),
    "event.EventData": ("data = event.EventData",
                        "Creates an empty event data object with the properties Source and EventName. "
                        "Classes that carry extra data with an event derive from it: classdef MyData < "
                        "event.EventData."),
    "enumeration": (["enumeration ClassName", "[members, names] = enumeration('ClassName')",
                     "[members, names] = enumeration(obj)"],
                    "Lists the members of an enumeration class (a classdef with an enumeration block), or "
                    "returns them as an object array together with their names."),
    "isenum": ("tf = isenum(x)", "True when x is a member (or an array of members) of an enumeration class."),

    # -- environment ------------------------------------------------------------------------------------
    "uistack": ("uistack(h, 'top')", _NOOP),
    "movegui": ("movegui(fig, 'center')", _NOOP),
    "workspace": ("workspace", _NOOP),
    "figurepalette": ("figurepalette", _NOOP),
    "plotbrowser": ("plotbrowser", _NOOP),
    "propertyeditor": ("propertyeditor", _NOOP),
    "plottools": ("plottools", _NOOP),
    "brush": ("brush on", _NOOP),
    "linkdata": ("linkdata on", _NOOP),
    "linkprop": ("linkprop(handles, 'XLim')", _NOOP + " Properties are not kept in sync."),
    "web": (["web(url)", "status = web(url)"],
            "Does not open a browser. Prints a short note with the address, or returns status 0 when an "
            "output is requested."),
    "feature": (["n = feature('numcores')", "v = feature(name)"],
                "Reads an internal setting. 'numcores' gives the number of processor cores, 'getpid' the "
                "process number, 'DefaultCharacterSet' is 'UTF-8'. Other names return 0."),
    "javachk": ("err = javachk(feature)",
                "Returns a struct with a message saying that Java is not available in mylab."),
}
