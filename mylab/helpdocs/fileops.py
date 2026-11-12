"""Help text for file identifiers, binary files, paths, the operating system and compatibility
functions (mylab/builtins/fileops.py and the file functions of system.py and io.py).

Written from scratch for mylab.
"""

_STATUS = ("With outputs, [status, message, messageId] report success (true) or the reason for failure "
           "instead of raising an error.")
_PRECISION = ("Precision names: 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', "
              "'single' ('float32'), 'double' ('float64'), 'char' and C-style names such as 'short', 'int', "
              "'long', 'float' and 'uchar'. Bit-field precisions such as 'ubit3' are not supported.")
_MACHINE = "The byte order is 'l' or 'ieee-le' (little-endian), 'b' or 'ieee-be' (big-endian) or 'n' (native)."

HELP = {
    # -- file identifiers --------------------------------------------------------------------------
    "fopen": (["fid = fopen(filename)", "fid = fopen(filename, permission)",
               "fid = fopen(filename, permission, machinefmt, encoding)", "[fid, msg] = fopen(...)",
               "[filename, permission, machinefmt, encoding] = fopen(fid)", "fids = fopen('all')"],
              "Opens a file and returns a file id (3 or higher) for the reading and writing functions, or "
              "-1 on failure with the reason in msg. permission is 'r' (read, the default), 'w' (create or "
              "empty, then write), 'a' (append), or 'r+', 'w+', 'a+' for reading and writing; a 'b' or 't' "
              "suffix is accepted and makes no difference, since line endings are never translated. "
              "machinefmt sets the default byte order for fread and fwrite: " + _MACHINE + " encoding "
              "(default 'UTF-8') is used when text is read or written. A file opened for reading is also "
              "looked for on the search path. File ids 1 and 2 always mean the command window. Given a "
              "file id, fopen returns the file's name and settings; fopen('all') lists the open ids."),
    "fclose": (["fclose(fid)", "fclose('all')", "status = fclose(...)"],
               "Closes a file opened with fopen, or all of them. status is 0. An id that is not open is an "
               "error."),
    "fgetl": ("line = fgetl(fid)",
              "Reads the next line of a file as text, without its line ending. Returns the number -1 at the "
              "end of the file, so test the result with ischar."),
    "fgets": (["line = fgets(fid)", "line = fgets(fid, nchar)"],
              "Reads the next line of a file as text, keeping its line ending; nchar limits how many "
              "characters are returned. Returns -1 at the end of the file."),
    "feof": ("tf = feof(fid)",
             "True when there is nothing left to read from the file. Unlike some implementations this is "
             "already true right after the last byte has been read, without a further failed read."),
    "fread": (["A = fread(fid)", "A = fread(fid, size)", "A = fread(fid, size, precision)",
               "A = fread(fid, size, precision, skip)", "A = fread(fid, size, precision, skip, machinefmt)",
               "[A, count] = fread(...)"],
              "Reads binary data from a file. size is the number of values, Inf (the default) for the rest "
              "of the file, or [m n] to fill an m-by-n matrix column by column (n may be Inf). precision "
              "says how the bytes are stored and what class is returned: 'uint8' (the default) reads bytes "
              "and returns double; 'int16=>int32' reads int16 and returns int32; '*int16' is short for "
              "'int16=>int16'; 'uint8=>char' and '*char' return text. skip is a number of bytes to jump "
              "over after each value, or after each group of N values when the precision is written "
              "'N*type'. count is the number of values actually read; a short read fills an incomplete "
              "last column with zeros. " + _PRECISION + " " + _MACHINE),
    "fwrite": (["fwrite(fid, A)", "fwrite(fid, A, precision)", "fwrite(fid, A, precision, skip)",
                "fwrite(fid, A, precision, skip, machinefmt)", "count = fwrite(...)"],
               "Writes the elements of A to a file as binary data, column by column. precision is the type "
               "stored in the file ('uint8' by default); values are rounded and clipped to its range. Text "
               "is written as character codes, or in the file's encoding with precision 'char'. skip "
               "leaves that many bytes untouched before each value. count is the number of elements "
               "written. With fid 1 or 2 the values are shown in the command window as characters. "
               + _PRECISION + " " + _MACHINE),
    "fseek": (["fseek(fid, offset, origin)", "status = fseek(fid, offset, origin)"],
              "Moves the read/write position of a file by offset bytes from origin: 'bof' or -1 (start of "
              "file, the default), 'cof' or 0 (current position), 'eof' or 1 (end of file). Returns 0, or "
              "-1 (leaving the position unchanged) when the target lies outside the file."),
    "ftell": ("position = ftell(fid)", "Current position in a file, in bytes from its start."),
    "frewind": ("frewind(fid)", "Moves the position of a file back to its start."),
    "ferror": (["msg = ferror(fid)", "[msg, errnum] = ferror(fid)", "ferror(fid, 'clear')"],
               "Message of the most recent file error, such as a failed fseek, or '' when there was none "
               "(errnum is then 0). 'clear' resets it. mylab keeps one message for all files."),
    "fflush": ("fflush(fid)",
               "Writes any buffered output of a file to disk. Accepted with any argument, including stdout, "
               "so that Octave-style code runs."),
    "textread": (["[a, b, ...] = textread(filename, format)", "[a, b, ...] = textread(filename, format, n)",
                  "[...] = textread(..., 'delimiter', d, 'headerlines', k)", "A = textread(filename)"],
                 "Reads a whole text file with a textscan-style format, returning one output per "
                 "conversion: numeric columns for %f and %d, cell arrays of text for %s. n limits how many "
                 "times the format is used. Without a format the file is read as a numeric matrix. New code "
                 "should prefer textscan or readmatrix."),
    # -- search path -------------------------------------------------------------------------------
    "addpath": (["addpath(folder1, folder2, ...)", "addpath(genpath(folder))", "addpath(..., '-end')",
                 "oldpath = addpath(...)"],
                "Adds folders to the search path used to find .m files and data files. Each argument may "
                "hold several folders separated by the path separator (':' or ';'), as genpath returns "
                "them. Folders go to the front of the path in the order given, or to the back with '-end'; "
                "a folder that is already on the path is moved."),
    "genpath": (["p = genpath(folder)", "p = genpath"],
                "The folder and all folders below it, joined with the path separator, for use with addpath. "
                "Folders named private or resources, and those whose names start with '@', '+' or '.', are "
                "left out. Returns '' if the folder does not exist."),
    "filesep": ("s = filesep", "Separator between folders in a path: '/' on Linux and macOS, '\\' on Windows."),
    "pathsep": ("s = pathsep", "The character that separates folders in a search path: ':' (';' on Windows)."),
    "mfilename": (["name = mfilename", "p = mfilename('fullpath')", "p = mfilename('fullpathext')"],
                  "Name of the .m file that is currently running, without folder or extension. 'fullpath' "
                  "adds the folder, 'fullpathext' the folder and extension. Empty when called from the "
                  "prompt."),
    "which": (["which name", "p = which(name)", "c = which(name, '-all')"],
              "Tells what a name refers to: 'variable' for a workspace variable, the full path of the .m "
              "file for a file function or script, the path of a data file found on the search path, or "
              "'built-in (mylab/name)' for a function built into mylab. Returns '' (and prints a message "
              "when used as a command) for an unknown name. '-all' returns every match in a cell array."),
    "what": (["what", "what(folder)", "s = what(folder)"],
             "Lists the MATLAB files in a folder (the current one by default). With an output it returns a "
             "struct with the folder's path and cell arrays m, mat, mlx, classes and packages."),
    # -- operating system --------------------------------------------------------------------------
    "system": (["status = system(command)", "[status, output] = system(command)",
                "[status, output] = system(command, '-echo')"],
               "Runs an operating-system command through the shell and waits for it. status is its exit "
               "code (0 normally means success). With one output the command's output is shown in the "
               "command window; with two it is captured as text instead, unless '-echo' asks for both. Not "
               "available when mylab runs in the browser."),
    "unix": (["[status, output] = unix(command)"],
             "Same as system on Linux and macOS; on Windows it does nothing and returns status 0."),
    "dos": (["[status, output] = dos(command)"], "Same as system, kept for code written for Windows."),
    "setenv": (["setenv(name, value)", "setenv(name)"],
               "Sets an environment variable for this session and for commands started with system. "
               "Without a value the variable is set to ''."),
    "copyfile": (["copyfile(source, destination)", "[status, message, messageId] = copyfile(...)"],
                 "Copies a file or a whole folder. source may contain wildcards (*), and destination may be "
                 "a folder; existing files are overwritten. " + _STATUS),
    "movefile": (["movefile(source, destination)", "[status, message, messageId] = movefile(...)"],
                 "Moves or renames a file or folder. source may contain wildcards, and destination may be "
                 "an existing folder to move into. " + _STATUS),
    "ls": (["ls", "ls(folder_or_pattern)", "list = ls(...)"],
           "Lists the files in a folder, or those matching a wildcard pattern. With an output the names "
           "are returned as the rows of a char matrix, padded with spaces (on every platform)."),
    "memory": (["memory", "user = memory", "[user, sys] = memory"],
               "Rough memory figures: user has MaxPossibleArrayBytes, MemAvailableAllArrays and "
               "MemUsedMATLAB; sys.PhysicalMemory has Available and Total. Values are 0 where the platform "
               "does not report them."),
    # -- archives ----------------------------------------------------------------------------------
    "zip": (["zip(zipfile, files)", "zip(zipfile, files, rootfolder)", "names = zip(...)"],
            "Creates a zip archive from a file, folder, wildcard pattern or cell array of those. Names in "
            "the archive are relative to rootfolder (the current folder by default). '.zip' is added to "
            "the archive name if it has no extension."),
    "unzip": (["unzip(zipfile)", "unzip(zipfile, outputfolder)", "names = unzip(...)"],
              "Extracts a zip archive (a file or an http(s) address) into the current folder or "
              "outputfolder, and returns the names of the extracted files."),
    "gzip": (["gzip(files)", "gzip(files, outputfolder)", "names = gzip(...)"],
             "Compresses each file into a .gz file next to it, or in outputfolder. The originals are kept."),
    "gunzip": (["gunzip(files)", "gunzip(files, outputfolder)", "names = gunzip(...)"],
               "Decompresses .gz files (or an http(s) address) next to the originals, or into outputfolder."),
    "tar": (["tar(tarfile, files)", "tar(tarfile, files, rootfolder)", "names = tar(...)"],
            "Creates a tar archive from files, folders or wildcard patterns. A name ending in .tgz or .gz "
            "is compressed as well."),
    "untar": (["untar(tarfile)", "untar(tarfile, outputfolder)", "names = untar(...)"],
              "Extracts a tar archive, compressed or not, from a file or an http(s) address."),
    # -- the web -----------------------------------------------------------------------------------
    "webread": (["data = webread(url)", "data = webread(url, name1, value1, ...)", "data = webread(..., options)"],
                "Downloads the content of a web address. Text is returned as a char vector, JSON is "
                "decoded when jsondecode is available, and pictures are returned as image arrays. "
                "Name-value pairs are added to the address as query parameters. Needs network access: it "
                "raises a clear error when the address cannot be reached, and is unavailable when mylab "
                "runs in the browser."),
    "websave": (["websave(filename, url)", "websave(filename, url, name1, value1, ...)", "f = websave(...)"],
                "Downloads a web address into a file and returns the file's full path. Needs network "
                "access, like webread."),
    "urlread": (["str = urlread(url)", "[str, status] = urlread(url)",
                 "str = urlread(url, 'get', {name1, value1, ...})", "str = urlread(url, 'post', params)"],
                "Older form of webread that always returns text. With two outputs a failure gives '' and "
                "status false instead of an error."),
    "urlwrite": (["urlwrite(url, filename)", "[f, status] = urlwrite(url, filename)"],
                 "Older form of websave: downloads a web address into a file."),
    "weboptions": (["options = weboptions(name1, value1, ...)"],
                   "Options for webread and websave as a struct. Only Timeout (seconds) has an effect."),
    # -- versions and capabilities -----------------------------------------------------------------
    "version": (["v = version", "[v, d] = version", "r = version('-release')", "version('-date')",
                 "version('-description')"],
                "Version text. So that programs checking for a recent MATLAB take their modern code path, "
                "mylab answers as MATLAB 9.14 (R2023a): version returns '9.14.0 (R2023a) mylab x.y.z', "
                "'-release' gives '2023a', '-date' (and the second output) the date of that release, and "
                "'-description' mylab's own version."),
    "ver": (["ver", "v = ver", "v = ver(product)"],
            "Prints mylab's own version and the product versions it reports to programs. With an output it "
            "returns a struct array with fields Name, Version, Release and Date for MATLAB and for the "
            "toolboxes mylab covers in part (Image Processing, Signal Processing, Statistics and Machine "
            "Learning, Optimization); ver('signal') selects one, and an unknown product gives an empty "
            "struct array."),
    "verLessThan": (["tf = verLessThan(product, version)"],
                    "True if the installed version of a product is older than the given version text, e.g. "
                    "verLessThan('matlab', '9.4'). mylab compares against the versions listed by ver (MATLAB "
                    "9.14 and the matching 'images', 'signal', 'stats' and 'optim' versions). Any other "
                    "toolbox is not available and counts as older than every version, so the result is "
                    "true rather than an error."),
    "isMATLABReleaseOlderThan": (["tf = isMATLABReleaseOlderThan(release)"],
                                 "True if the release mylab reports (R2023a) is older than the given one, "
                                 "e.g. isMATLABReleaseOlderThan('R2024a')."),
    "license": (["tf = license('test', feature)", "license"],
                "Says whether a product is available: true for 'MATLAB' and for the toolboxes mylab covers "
                "in part ('image_toolbox', 'signal_toolbox', 'statistics_toolbox', 'optimization_toolbox'), "
                "false for everything else. mylab itself is free software under the MIT licence."),
    "usejava": (["tf = usejava(feature)"], "Always false: mylab has no Java virtual machine or Java-based desktop."),
    "mex": (["mex(...)"],
            "Raises an error: compiled MEX files cannot be built or run by mylab. Projects usually ship a "
            "plain .m version of such functions, which works."),
    "gpuArray": (["G = gpuArray(A)"],
                 "Returns A unchanged. mylab has no GPU arrays, so all calculations run on the CPU; this "
                 "exists so that GPU-enabled code runs without edits."),
    "gather": (["A = gather(G)"], "Returns its input unchanged (see gpuArray)."),
    "gpuDeviceCount": (["n = gpuDeviceCount"], "Always 0: mylab does not use GPUs."),
    "canUseGPU": (["tf = canUseGPU"], "Always false: mylab does not use GPUs."),
    "canUseParallelPool": (["tf = canUseParallelPool"], "Always false: mylab runs everything in one process."),
    "parpool": (["parpool", "parpool(n)", "pool = parpool(...)"],
                "Accepted for compatibility. mylab runs code in a single process, so no workers are started; "
                "the returned struct describes a pool with NumWorkers 1, and delete(pool) does nothing."),
    "gcp": (["pool = gcp", "pool = gcp('nocreate')"],
            "The current parallel pool: a struct with NumWorkers 1, or [] with 'nocreate' because no pool "
            "is ever running."),
    "matlabpool": (["matlabpool(...)", "n = matlabpool('size')"],
                   "Accepted for old code and does nothing; the pool size is 0."),
    # -- preferences -------------------------------------------------------------------------------
    "setpref": (["setpref(group, pref, value)", "setpref(group, {pref1, pref2}, {value1, value2})"],
                "Stores a preference value under a group name. In mylab preferences live in memory only and "
                "are forgotten when the session ends. addpref is the same."),
    "addpref": (["addpref(group, pref, value)"], "Same as setpref in mylab (an existing preference is replaced)."),
    "getpref": (["value = getpref(group, pref)", "value = getpref(group, pref, default)", "s = getpref(group)",
                 "s = getpref"],
                "Value of a stored preference. If it does not exist the default is stored and returned, or "
                "an error is raised when no default is given. With only a group, all its preferences are "
                "returned as a struct."),
    "ispref": (["tf = ispref(group)", "tf = ispref(group, pref)"],
               "True if the preference group, or the named preference in it, exists."),
    "rmpref": (["rmpref(group)", "rmpref(group, pref)"], "Removes a preference group or one preference from it."),
    # -- workspaces --------------------------------------------------------------------------------
    "assignin": (["assignin(ws, name, value)"],
                 "Assigns value to the variable name in another workspace: 'base' (the top-level workspace) "
                 "or 'caller' (the function that called the current one)."),
    "evalin": (["evalin(ws, expression)", "v = evalin(ws, expression)", "evalin(ws, try_code, catch_code)"],
               "Runs code or evaluates an expression in the 'base' or 'caller' workspace, so it sees and "
               "changes the variables there. A third argument is run if the first raises an error."),
    "inputname": (["name = inputname(k)"],
                  "Always returns ''. mylab does not record which variable names the caller used for a "
                  "function's arguments, so code should fall back to its own default name."),
    "diary": (["diary", "diary on", "diary off", "diary(filename)"],
              "Records everything printed in the command window in a text file. diary on appends to the "
              "file named 'diary' in the current folder (or the name set earlier), diary(filename) switches "
              "to that file and turns recording on, diary off stops, and diary alone toggles. Typed "
              "commands are not recorded, only output."),
    # -- compatibility -----------------------------------------------------------------------------
    "isstr": (["tf = isstr(x)"], "Old name of ischar: true if x is a char array."),
    "strmatch": (["idx = strmatch(str, strs)", "idx = strmatch(str, strs, 'exact')"],
                 "Rows of a char matrix, or elements of a cell array of text, that begin with str; a column "
                 "of indices. With 'exact' the whole entry must equal str (trailing padding of char-matrix "
                 "rows is ignored). An old function; strcmp, strncmp and startsWith are preferred."),
    "findstr": (["k = findstr(s1, s2)"],
                "Starting positions where the shorter of the two texts occurs in the longer one. An old "
                "function; strfind is preferred."),
    "nargchk": (["msg = nargchk(low, high, n)", "msg = nargchk(low, high, n, 'struct')"],
                "Returns an error message if n is outside low..high, or an empty value when it is in range; "
                "typically used as error(nargchk(1, 3, nargin)). With 'struct' the result is a struct with "
                "message and identifier. narginchk is the modern replacement."),
    "shiftdim": (["B = shiftdim(A, n)", "[B, nshifts] = shiftdim(A)"],
                 "Shifts the dimensions of an array. A positive n rotates them to the left (the first n "
                 "move to the end); a negative n adds leading dimensions of length 1. With one input, "
                 "leading dimensions of length 1 are removed and nshifts says how many."),
    "flipdim": (["B = flipdim(A, dim)"],
                "Old name of flip(A, dim): reverses the order of elements along dimension dim."),
    "swapbytes": (["Y = swapbytes(X)"],
                  "Reverses the byte order of every element, converting between little- and big-endian "
                  "data. Example: swapbytes(uint16(1)) is 256."),
    "strread": (["[a, b, ...] = strread(text, format)", "[...] = strread(text, format, n, 'delimiter', d)"],
                "Reads formatted data from text, like textread does from a file: one output per conversion "
                "in the format. An old function; textscan and sscanf are preferred."),
    "gpuDevice": (["g = gpuDevice", "gpuDevice(index)"],
                  "Describes the GPU in use. mylab has none, so the struct reports DeviceAvailable false and "
                  "no memory; selecting or resetting a device does nothing."),
    "blkdiag": (["B = blkdiag(A1, A2, ...)"],
                "Block-diagonal matrix with the given matrices along its diagonal and zeros elsewhere. "
                "Example: blkdiag([1 2; 3 4], 5) is a 3-by-3 matrix."),
    "getfield": (["v = getfield(s, name)", "v = getfield(s, name1, name2, ...)",
                  "v = getfield(s, {i, j}, name, {k})"],
                 "Value of a field, the same as s.name. Several names follow nested structs (s.a.b), and a "
                 "cell array of subscripts indexes the value before it, so getfield(s, {2}, 'a', {3}) is "
                 "s(2).a(3)."),
    "setfield": (["s = setfield(s, name, value)", "s = setfield(s, name1, name2, ..., value)",
                  "s = setfield(s, {i}, name, {k}, value)"],
                 "Returns a copy of the struct with a field set, the same as s.name = value. Several names "
                 "reach into nested structs, creating them as needed, and cell arrays of subscripts index "
                 "the value before them."),
    "substruct": (["S = substruct(type1, subs1, type2, subs2, ...)"],
                  "Builds the subscript description used by subsref and subsasgn: a struct array with "
                  "fields type ('()', '{}' or '.') and subs (a cell array of subscripts, or a field name)."),
    "subsref": (["B = subsref(A, S)"],
                "Applies the indexing described by S (from substruct) to A; for example "
                "subsref(A, substruct('()', {1, ':'})) is A(1, :). Classes cannot overload it."),
    "subsasgn": (["A = subsasgn(A, S, B)"],
                 "Returns A after the assignment described by S (from substruct); for example "
                 "subsasgn(A, substruct('.', 'x'), 5) is A.x = 5. Classes cannot overload it."),
}
