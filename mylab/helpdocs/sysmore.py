"""Help text for the system functions of mylab/builtins/sysmore.py, and for functions in older
modules whose behaviour this area extended (saving and loading, text import, random numbers,
warnings, audio and video files). Entries here replace the earlier ones of the same name.

Written from scratch for mylab.
"""

_PY = (" Python calls can be switched off by the program that hosts mylab (the interpreter's python_enabled "
       "setting, or the environment variable MYLAB_PYTHON=0); they then stop with an error.")
_NOJAVA = ("mylab runs on Python and has no Java virtual machine, so this function always stops with an error "
           "that says so. Use MATLAB-language types (string, containers.Map, cell arrays, datetime) or a "
           "Python library through py.name(...) instead.")
_STREAM = (" After rng(seed) the numbers are the ones MATLAB's default generator (the Mersenne Twister) "
           "returns for that seed, and arrays are filled column by column from the same stream.")
_IMPORT = ("Options (name-value pairs): 'Delimiter' (a character, a name such as 'tab' or 'comma', or a cell "
           "array), 'NumHeaderLines', 'Range' (a corner 'B2', a block 'A2:C5', rows '2:5' or [row1 col1 row2 "
           "col2]), 'DecimalSeparator', 'ThousandsSeparator', 'TreatAsMissing', 'TrimNonNumeric', "
           "'CommentStyle', 'Encoding', 'LineEnding', 'EmptyLineRule', 'ConsecutiveDelimitersRule', "
           "'LeadingDelimitersRule', 'TrailingDelimitersRule', 'ExtraColumnsRule', 'ExpectedNumVariables', "
           "'DataLines' and 'VariableNamesLine'. An unknown option is an error. Without 'Delimiter' the "
           "delimiter is found by trying comma, tab, semicolon, bar and white space; lines above the table "
           "that do not have its number of fields are skipped.")

HELP = {
    # -- function handles, memoize, names ---------------------------------------------------------------
    "functions": ("s = functions(fh)",
                  "Describes a function handle as a struct: function (the name, or the text of an anonymous "
                  "function), type ('simple', 'anonymous', 'scopedfunction' for a local function, 'nested') "
                  "and file. For an anonymous function, workspace{1} is a struct of the variables it "
                  "captured when it was created."),
    "memoize": (["m = memoize(fh)", "y = m(x, ...)", "clearCache(m)", "s = stats(m)", "m.CacheSize = n",
                 "m.Enabled = false"],
                "Returns a version of the function that remembers its results: a call with the same inputs "
                "(and the same number of outputs) returns the stored outputs without running the function "
                "again. CacheSize (10 by default) is the number of input combinations kept; when it is "
                "full the one used longest ago is dropped. Enabled false calls the function every time. "
                "clearCache empties the cache, and stats(m) reports its contents and hit counts (Cache, "
                "MostHitCachedInput, CacheHitRatePercent, CacheOccupancyPercent). Memoizing the same named "
                "function twice gives the same object. Only memoize functions without side effects."),
    "clearAllMemoizedCaches": ("clearAllMemoizedCaches",
                               "Empties the cache of every memoized function made with memoize."),
    "matlab.lang.makeValidName": (
        ["N = matlab.lang.makeValidName(S)", "N = matlab.lang.makeValidName(S, 'ReplacementStyle', style)",
         "N = matlab.lang.makeValidName(S, 'Prefix', p)", "[N, modified] = matlab.lang.makeValidName(...)"],
        "Turns text into valid variable names. A letter that follows white space becomes a capital and the "
        "white space is removed; other invalid characters become underscores ('underscore', the default), "
        "are removed ('delete') or are written as their hexadecimal code ('hex'); a name that does not "
        "start with a letter, and a keyword, gets the prefix 'x' (or p). Names are cut to 63 characters. "
        "S is a character vector, a string array or a cell array; modified tells which names changed. "
        "Example: matlab.lang.makeValidName('1 bad name') is 'x1BadName'."),
    "matlab.lang.makeUniqueStrings": (
        ["U = matlab.lang.makeUniqueStrings(S)", "U = matlab.lang.makeUniqueStrings(S, excluded)",
         "U = matlab.lang.makeUniqueStrings(S, whichToCheck)",
         "U = matlab.lang.makeUniqueStrings(S, excluded, maxlength)", "[U, modified] = ..."],
        "Makes the texts in S different from each other, and from those in excluded, by appending _1, _2 "
        "and so on to later duplicates. A logical or index vector instead of excluded says which elements "
        "may be changed. maxlength limits the length of the results (63 by default)."),
    "namelengthmax": ("n = namelengthmax", "The longest allowed name of a variable or function: 63 characters."),
    "rats": (["s = rats(x)", "s = rats(x, len)"],
             "Text with each element of x written as a ratio of small integers that approximates it, for "
             "example rats(0.75) shows 3/4 and rats(pi) shows 355/113. len is the room given to each "
             "element (13 by default); a larger len allows closer approximations. A value too small to "
             "show next to the others is shown as an asterisk. See rat for the numerator and denominator "
             "as numbers."),
    # -- threads, locks, profiler, futures ------------------------------------------------------------------
    "maxNumCompThreads": (["n = maxNumCompThreads", "old = maxNumCompThreads(n)",
                           "old = maxNumCompThreads('automatic')"],
                          "The number of computational threads: by default the number of processor cores. "
                          "Setting a number returns the previous value. mylab records the setting; the "
                          "threads NumPy uses for linear algebra are chosen by its own libraries."),
    "mlock": ("mlock", "Called inside a function: marks that function as locked. mislocked reports it and "
              "munlock removes the mark."),
    "munlock": (["munlock", "munlock(name)"], "Removes the lock that mlock put on the running function, or on "
                "the function called name."),
    "mislocked": (["tf = mislocked", "tf = mislocked(name)"],
                  "True if the running function, or the function called name, was locked with mlock."),
    "profile": (["profile on", "profile off", "profile resume", "profile clear", "profile viewer",
                 "p = profile('info')", "s = profile('status')"],
                "Measures where a program spends its time. Between profile on and profile off every "
                "statement of scripts and functions is timed (built-in functions count towards the "
                "statement that calls them). profile('info') returns a struct whose FunctionTable has one "
                "element per function with FunctionName, FileName, NumCalls, TotalTime (including the "
                "functions it calls), ExecutedLines (line, count, time), Children and Parents; profile "
                "viewer prints a summary table instead of opening a window. profile resume continues "
                "without clearing, profile clear discards the measurements, profile('status') tells "
                "whether the profiler is on. Programs run slower while it is on."),
    "profsave": (["profsave", "profsave(info, folder)"],
                 "Writes the profiler's summary table to profile.txt in the folder (profile_results by "
                 "default). mylab writes plain text, not the HTML pages MATLAB produces."),
    "parfeval": (["f = parfeval(fcn, nout, in1, ...)", "f = parfeval(pool, fcn, nout, in1, ...)",
                  "[out1, ...] = fetchOutputs(f)", "wait(f)", "cancel(f)"],
                 "Evaluates fcn(in1, ...) for nout outputs and returns a future that holds the result. "
                 "mylab has no background workers: the function runs at once, and the future is finished "
                 "when parfeval returns. fetchOutputs(f) returns the outputs, or raises the error the "
                 "function raised. Properties: State ('finished'), Error, OutputArguments, InputArguments, "
                 "Function, NumOutputArguments, Diary (what the function printed), Read and ID."),
    "parfevalOnAll": ("f = parfevalOnAll(fcn, nout, in1, ...)",
                      "Like parfeval: with no background workers the function runs once, in this session."),
    "fetchOutputs": ("[out1, out2, ...] = fetchOutputs(f)",
                     "The outputs of the function a future from parfeval ran. If that function failed, its "
                     "error is raised here."),
    "fetchNext": ("[idx, out1, ...] = fetchNext(f)",
                  "The outputs of a future that has not been read yet, with its index (1). An error if the "
                  "future was already read."),
    "backgroundPool": ("pool = backgroundPool",
                       "A stand-in for MATLAB's background pool, accepted as the first argument of parfeval. "
                       "Work given to it runs at once in this session."),
    # -- Python ----------------------------------------------------------------------------------------------
    "py.*": (["y = py.module.function(args)", "obj = py.list({1, 2, 3})", "n = py.len(obj)",
              "x = py.math.pi", "obj.method(args)", "d = double(obj)", "c = cell(obj)", "s = char(obj)",
              "t = struct(pydict)", "f = @py.math.sqrt"],
             "Calls Python from MATLAB code: py. followed by a built-in name (py.list, py.len, py.dict) or a "
             "module and a name in it (py.math.sqrt, py.os.path.join, py.numpy.linalg.norm). Arguments are "
             "converted: numeric scalars to float, int (integer classes), bool or complex; text to str; a "
             "real row vector to an array.array; other arrays to NumPy arrays; a cell vector to a tuple; a "
             "scalar struct or containers.Map to a dict; a function handle to a callable; keyword arguments "
             "come from pyargs. Results that are float, complex or bool become double, complex double or "
             "logical, int becomes int64, and everything else stays a Python object (its class shows as "
             "py.list, py.str, py.numpy.ndarray, ...). Convert such an object with double, single, an "
             "integer class, logical, char, string, cell (elements keep their Python type unless they are "
             "numbers) or struct (for a dict); numel and length give len(obj). obj.name reads an attribute "
             "or calls a method, obj{k} takes element k (counting from 1) or a dict entry, and obj(args) "
             "calls a callable object. A Python exception becomes a MATLAB error whose message starts "
             "with 'Python Error:'. What Python prints appears in the command window." + _PY),
    "pyargs": ("kw = pyargs('name1', value1, 'name2', value2, ...)",
               "Keyword arguments for a Python call: py.sorted(x, pyargs('reverse', true)) calls "
               "sorted(x, reverse=True). Put it after the positional arguments."),
    "pyrun": (["pyrun(code)", "out = pyrun(code, outvar)", "[o1, o2] = pyrun(code, [\"a\" \"b\"])",
               "out = pyrun(code, outvar, name=value, ...)"],
              "Runs Python statements (a string, or a string array with one line each) and returns the "
              "variables named in outvar. name=value pairs create Python variables first. Variables stay "
              "defined from one pyrun call to the next, so a later call can use what an earlier one "
              "computed. Results are converted as described under py.*." + _PY),
    "pyrunfile": (["pyrunfile(file)", "out = pyrunfile(file, outvar)", "out = pyrunfile(\"script.py arg1 arg2\", "
                   "outvar, name=value)"],
                  "Runs a Python script file, with the words after the file name as its command-line "
                  "arguments (sys.argv) and the name=value pairs as variables, and returns the variables "
                  "named in outvar. Each run starts with fresh variables." + _PY),
    "pyenv": ("pe = pyenv",
              "Describes the Python that Python calls use: Version, Executable, Library, Home, Status "
              "('Loaded') and ExecutionMode ('InProcess'). It is the Python that mylab itself runs on and "
              "cannot be changed with pyenv; to use another Python, start mylab with it." + _PY),
    # -- Java: not available ------------------------------------------------------------------------------------
    "java.*": ("java.package.Class(args)", "Java classes cannot be used. " + _NOJAVA),
    "javax.*": ("javax.package.Class(args)", "Java classes cannot be used. " + _NOJAVA),
    "javaObject": ("obj = javaObject(className, args)", "Would create a Java object. " + _NOJAVA),
    "javaObjectEDT": ("obj = javaObjectEDT(className, args)", "Would create a Java object. " + _NOJAVA),
    "javaMethod": ("out = javaMethod(methodName, object, args)", "Would call a Java method. " + _NOJAVA),
    "javaMethodEDT": ("out = javaMethodEDT(methodName, object, args)", "Would call a Java method. " + _NOJAVA),
    "javaArray": ("a = javaArray(className, m, n)", "Would create a Java array. " + _NOJAVA),
    "javaaddpath": ("javaaddpath(folder)", "Would add to the Java class path. " + _NOJAVA),
    "javarmpath": ("javarmpath(folder)", "Would remove from the Java class path. " + _NOJAVA),
    "javaclasspath": ("p = javaclasspath", "Would return the Java class path. " + _NOJAVA),
    "isjava": ("tf = isjava(x)", "Whether x is a Java object: always false, because mylab has no Java."),
    # -- functions of older modules that changed -------------------------------------------------------------
    "save": (["save", "save(filename)", "save(filename, 'var1', 'var2', ...)", "save filename var1 var2",
              "save(filename, 'a*', '-regexp', '^x\\d')", "save(filename, '-struct', 's')",
              "save(filename, 'x', '-append')", "save(filename, 'x', '-v7.3')", "save(filename, 'x', '-ascii')"],
             "Saves variables of the workspace to a MAT-file (matlab.mat when no name is given; .mat is "
             "added to a name without extension). Names may contain * and ? wildcards, and -regexp is "
             "followed by regular expressions. -struct saves the fields of a scalar struct as separate "
             "variables. -append adds to an existing file, replacing variables of the same name. Formats: "
             "-v7 (the default, compressed), -v6 and -nocompression (uncompressed), -v4, -v7.3 (an HDF5 "
             "file; needs the optional Python package h5py) and -ascii (text with 8 digits, or 16 with "
             "-double, tab-separated with -tabs; numeric and char arrays only). All numeric classes, "
             "logical, char, complex and sparse arrays, cells, structs and struct arrays are stored in the "
             "standard way, so MATLAB and Octave read them. String arrays, tables, datetime, duration, "
             "calendarDuration, categorical, containers.Map, function handles and classdef objects (through "
             "saveobj when the class defines it) are stored as structs marked with the field mylab_class_: "
             "mylab restores them on load, other programs see the struct. A value that cannot be stored "
             "(a figure handle, for example) is an error that names the variable; nothing is skipped."),
    "load": (["load(filename)", "S = load(filename)", "load filename", "load(filename, 'var1', 'a*')",
              "load(filename, '-regexp', '^x\\d')", "load(filename, '-mat')", "A = load(filename, '-ascii')"],
             "Loads variables from a MAT-file into the workspace, or returns them as the fields of a struct. "
             "Names (with * and ? wildcards) or -regexp patterns select variables; asking for a name that "
             "is not in the file gives a warning. Files of versions 4, 5, 6 and 7 written by MATLAB or "
             "Octave are read directly, including big-endian files, objects, function handles and MATLAB "
             "string arrays; version 7.3 files need the optional Python package h5py. Variables saved as "
             "global become global again. A file that is not a MAT-file (or any file with -ascii) is read "
             "as rows of numbers separated by spaces, commas or tabs, with % comments, into one matrix "
             "named after the file. .mat is added to a name without extension, and the file is also looked "
             "for on the search path."),
    "who": (["who", "who a* b", "who -regexp pattern", "who global", "who -file filename", "names = who(...)"],
            "Lists the variables of the workspace, of the global workspace, or of a MAT-file, optionally "
            "only those that match the names, wildcards or regular expressions. With an output it returns "
            "the names as a cell column."),
    "whos": (["whos", "whos a* b", "whos -regexp pattern", "whos global", "whos -file filename", "s = whos(...)"],
             "Like who, with the size, bytes and class of each variable and whether it is global, sparse or "
             "complex. With an output it returns a struct array with the fields name, size, bytes, class, "
             "global, sparse, complex, nesting and persistent instead of printing. Bytes are counted for "
             "the data as MATLAB counts them (two bytes per character, 16 per complex double) and are "
             "approximate for cells, structs and objects."),
    "warning": (["warning(msg)", "warning(fmt, a1, ...)", "warning(id, fmt, a1, ...)", "warning('off', id)",
                 "warning('on', id)", "warning('error', id)", "s = warning('off', 'all')", "warning(s)",
                 "s = warning('query', id)", "s = warning"],
                "Prints a warning and records it for lastwarn, or controls which warnings are shown. "
                "warning('off', id) hides one warning and warning('off', 'all') all of them; 'on' shows "
                "them again; 'error' makes warning(id, ...) raise an error instead. Each of these returns "
                "the previous state as a struct with the fields identifier and state (a struct array for "
                "'all'), and warning(s) puts that state back, which is the usual way to silence a warning "
                "for a few lines only. warning('query', id) returns the current state, and warning with "
                "no input returns or lists all states. 'backtrace' and 'verbose' are accepted as modes."),
    "computer": (["str = computer", "arch = computer('arch')", "[str, maxsize, endian] = computer"],
                 "The kind of computer: 'GLNXA64', 'PCWIN64', 'MACI64' or 'MACA64' (in the browser "
                 "'WASM32'); with 'arch' the lower-case architecture name such as 'glnxa64'. maxsize is the "
                 "largest number of elements an array may have and endian is 'L' or 'B'."),
    "dbstack": (["dbstack", "s = dbstack", "s = dbstack(n)", "s = dbstack('-completenames')",
                 "[s, k] = dbstack"],
                "The functions that are running, innermost first. As an output it is a struct array with "
                "the fields file, name and line (the line each function is at); n leaves out the first n "
                "entries and '-completenames' gives full file paths. k is the index of the current "
                "workspace (1)."),
    "inputname": (["name = inputname(k)", "name = inputname(k, false)"],
                  "Inside a function: the name of the caller's variable that was passed as argument k, or '' "
                  "when the argument was an expression or a constant (an error instead when the second "
                  "argument is true). It is known for functions called by name with arguments in "
                  "parentheses; at the prompt, outside any function, it returns ''."),
    "isMATLABReleaseOlderThan": (
        ["tf = isMATLABReleaseOlderThan(release)", "tf = isMATLABReleaseOlderThan(release, stage)",
         "tf = isMATLABReleaseOlderThan(release, stage, update)"],
        "True if the MATLAB release mylab reports itself as (see version) is older than the given release, "
        "written like 'R2021b'. stage is 'release' (the default) or 'prerelease', and update an update "
        "number; mylab counts as the general release without updates."),
    "rand": (["r = rand", "R = rand(n)", "R = rand(m, n)", "R = rand([m n])", "R = rand(..., 'single')",
              "R = rand(..., 'like', p)"],
             "Uniformly distributed random numbers between 0 and 1. The class is double, or single when "
             "asked for by name or through a 'like' prototype." + _STREAM),
    "randn": (["r = randn", "R = randn(n)", "R = randn(m, n)", "R = randn([m n])", "R = randn(..., 'single')",
               "R = randn(..., 'like', p)"],
              "Normally distributed random numbers with mean 0 and standard deviation 1." + _STREAM),
    "randi": (["r = randi(imax)", "R = randi(imax, n)", "R = randi(imax, m, n)", "R = randi([imin imax], m, n)",
               "R = randi(..., classname)"],
              "Uniformly distributed random integers from 1 to imax, or from imin to imax. The result is "
              "double unless a numeric class name is given." + _STREAM),
    "randperm": (["p = randperm(n)", "p = randperm(n, k)"],
                 "A random permutation of 1:n, or k different numbers drawn from it. The permutation of all "
                 "n numbers is the one MATLAB returns for the same seed."),
    "rng": (["rng(seed)", "rng('default')", "rng('shuffle')", "rng(seed, generator)", "s = rng", "rng(s)"],
            "Seeds the random number generator. rng(seed) and rng('default') (seed 0) make rand, randn, "
            "randi and randperm repeat the same numbers, and they are the numbers MATLAB gives for that "
            "seed with its default generator, the Mersenne Twister. A new session starts as after "
            "rng('default'). rng('shuffle') seeds from the clock. s = rng returns the settings (Type, "
            "Seed, State) and rng(s) restores them. Other generator names are accepted and recorded, but "
            "the numbers always come from the Mersenne Twister."),
    "zeros": (["A = zeros(n)", "A = zeros(m, n)", "A = zeros(m, n, p, ...)", "A = zeros([m n])",
               "A = zeros(..., classname)", "A = zeros(..., 'like', p)"],
              "An array of zeros. The class is double unless a class name ('single', 'int8', ..., "
              "'logical') is given; 'like', p takes the class and complexity from the value p."),
    "ones": (["A = ones(n)", "A = ones(m, n)", "A = ones(m, n, p, ...)", "A = ones([m n])",
              "A = ones(..., classname)", "A = ones(..., 'like', p)"],
             "An array of ones. The class is double unless a class name is given; 'like', p takes the class "
             "and complexity from the value p."),
    "readtable": (["T = readtable(filename)", "T = readtable(filename, opts)", "T = readtable(filename, Name, Value)"],
                  "Reads a delimited text file (or a spreadsheet, see the Sheet and Range options there) "
                  "into a table. A first line of text gives the variable names, made into valid names "
                  "unless 'VariableNamingRule' is 'preserve'. Each column becomes numbers, datetime, "
                  "duration or text, whichever fits all its fields; empty fields are NaN, NaT or ''. "
                  + _IMPORT + " Further options: 'ReadVariableNames', 'ReadRowNames', 'TextType' ('char' or "
                  "'string'), 'DatetimeType' and 'DurationType' ('text' to keep them as text), "
                  "'MissingRule' and 'ImportErrorRule' ('fill', 'error', 'omitrow', 'omitvar'), and "
                  "'Format' with one conversion per column (%f, %d, %s, %q, %D, %C, %T, %*s to skip). "
                  "opts is an object from detectImportOptions."),
    "readmatrix": (["A = readmatrix(filename)", "A = readmatrix(filename, opts)",
                    "A = readmatrix(filename, Name, Value)"],
                   "Reads a delimited text file (or a spreadsheet) into a numeric matrix. Text lines at the "
                   "top are skipped, and fields that are empty or not numbers become NaN. " + _IMPORT
                   + " 'OutputType' returns another class instead of double: 'single', an integer class, "
                   "'string', 'char' (a cell array), 'datetime', 'duration' or 'categorical'."),
    "readcell": (["C = readcell(filename)", "C = readcell(filename, opts)", "C = readcell(filename, Name, Value)"],
                 "Reads a delimited text file (or a spreadsheet) into a cell array with one cell per field: "
                 "numbers as numbers, dates and times as datetime and duration values, other text as "
                 "character vectors (strings with 'TextType', 'string'), and empty fields as missing. "
                 + _IMPORT + " 'DatetimeType' and 'DurationType' set to 'text' keep dates and times as text."),
    "textscan": (["C = textscan(fid, format)", "C = textscan(text, format)", "C = textscan(..., N)",
                  "C = textscan(..., Name, Value)", "[C, position] = textscan(...)"],
                 "Reads formatted text from an open file or from a character vector into a cell array with "
                 "one cell per conversion. The format is applied again and again, N times at most. "
                 "Conversions: %f and %n (double; %f32 single), %d and %u (int32 and uint32, or with 8, "
                 "16, 32, 64 appended), %s (a word, or a whole field when a delimiter is set), %q (text "
                 "that may be in double quotes), %c (characters, %5c for five), %[abc] and %[^abc] "
                 "(characters in or not in a set), %D and %{format}D (datetime), %T (duration) and %C "
                 "(categorical). A number after % limits the field width, %*f reads and discards, and "
                 "other text in the format must match the input. Options: 'Delimiter' (one or a cell "
                 "array; without it fields are separated by white space, including line ends), "
                 "'EndOfLine', 'Whitespace', 'HeaderLines', 'CommentStyle' (a marker, or {open, close}), "
                 "'MultipleDelimsAsOne', 'EmptyValue' (for empty numeric fields; NaN by default), "
                 "'TreatAsEmpty', 'CollectOutput', 'TextType' and 'ReturnOnError' (false: stop with an "
                 "error where the text does not fit the format, instead of returning what was read). With "
                 "a delimiter, a line that ends early gives empty values for its remaining fields. Reading "
                 "from a file leaves it positioned after the text that was used; position is that place."),
    "fread": (["A = fread(fid)", "A = fread(fid, size)", "A = fread(fid, size, precision)",
               "A = fread(fid, size, precision, skip)", "A = fread(fid, size, precision, skip, machinefmt)",
               "[A, count] = fread(...)"],
              "Reads binary data. size is n, Inf or [m n] (n may be Inf). precision names the type in the "
              "file: 'uint8' (the default), 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', "
              "'single', 'double', 'char' and C-style names such as 'short', 'long' and 'float', or a bit "
              "field 'ubitN' or 'bitN' (N from 1 to 64; signed for bitN). The result is double unless the "
              "precision says otherwise: '*uint8' keeps the file's type and 'uint8=>char' converts. "
              "'N*type' with skip reads N values, then passes over skip bytes (bits for a bit field). "
              "Bit fields are packed from the least significant bit of each byte ('ieee-le'), or from the "
              "most significant with the 'ieee-be' byte order. count is the number of values read."),
    "fwrite": (["fwrite(fid, A)", "fwrite(fid, A, precision)", "fwrite(fid, A, precision, skip)",
                "fwrite(fid, A, precision, skip, machinefmt)", "count = fwrite(...)"],
               "Writes the elements of A, column by column, as binary data of the given precision ('uint8' by "
               "default; the names are those of fread, including the bit fields 'ubitN' and 'bitN'). Values "
               "out of range are set to the nearest limit. skip leaves that many bytes (bits for a bit "
               "field) before each value. Bit fields written one after another share bytes; the last byte "
               "is filled with zero bits when something else is written or the file is closed."),
    "audioread": (["[y, Fs] = audioread(file)", "[y, Fs] = audioread(file, [first last])",
                   "[y, Fs] = audioread(file, 'native')"],
                  "Reads a sound file: samples as doubles between -1 and 1, one column per channel, and the "
                  "sample rate. [first last] reads part of the file; 'native' returns the file's own "
                  "integer type. WAV and FLAC files are read by mylab itself; OGG, MP3, AIFF and AU files "
                  "need the optional Python package soundfile (which also makes FLAC faster)."),
    "audiowrite": (["audiowrite(file, y, Fs)", "audiowrite(file, y, Fs, 'BitsPerSample', n)",
                    "audiowrite(file, y, Fs, 'Quality', q)"],
                   "Writes samples (one column per channel, doubles between -1 and 1 or integers) to a sound "
                   "file whose format follows the extension. WAV (8, 16, 24, 32 or 64 bits per sample) and "
                   "FLAC (8, 16 or 24) are written by mylab itself; OGG, MP3, AIFF and AU need the optional "
                   "Python package soundfile. 'Quality' (0 to 100) and 'BitRate' apply to OGG and MP3. "
                   "Values outside -1..1 are clipped with a warning."),
    "audioinfo": ("info = audioinfo(file)",
                  "Describes a sound file: Filename, CompressionMethod, NumChannels, SampleRate, "
                  "TotalSamples, Duration and BitsPerSample (BitRate for OGG and MP3). The formats are "
                  "those of audioread."),
    "VideoWriter": (["v = VideoWriter(filename)", "v = VideoWriter(filename, profile)", "open(v)",
                     "writeVideo(v, frame)", "close(v)"],
                    "Creates an object that writes frames to a video file. Typical use: open(v); "
                    "writeVideo(v, getframe(gcf)) in a loop; close(v), which produces the file. Properties: "
                    "FrameRate (default 30), Quality (0-100), and read-only Filename, Path, FrameCount, "
                    "Width, Height, Duration. Profiles: 'Motion JPEG AVI' (the default for .avi), "
                    "'Uncompressed AVI', 'Grayscale AVI' and 'MPEG-4' (.mp4). AVI files are written by "
                    "mylab itself and can be read back with VideoReader. MPEG-4 needs the ffmpeg program; "
                    "when it is missing an animated GIF with the same base name is written and a warning "
                    "says so. A file name ending in .gif always writes an animated GIF."),
}
