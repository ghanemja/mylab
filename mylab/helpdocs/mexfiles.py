"""Help text for building and running MEX files (mylab/builtins/mexfiles.py).

Written from scratch for mylab.
"""

HELP = {
    "mex": (["mex file.c", "mex file.c helper.c -output name", "mex -I<dir> -L<dir> -l<lib> -D<macro> file.c",
             "mex -R2018a file.c", "mex -g -v file.c", "mex -setup", "status = mex(...)"],
            "Compiles C or C++ source files into a MEX file, a function written in C that mylab calls like a "
            ".m function. The system compiler does the work (cc, gcc or clang; $CC and $CXX override it), and "
            "the result is written to the current folder (or -outdir) with the extension mexext returns. "
            "The sources include mylab's own mex.h, which provides the usual mx and mex functions; nothing "
            "links to MATLAB. Options: -output name, -outdir dir, -I, -L, -l, -D and -U as for a C compiler, "
            "-O (optimise, the default), -g (debug build, turns on mxAssert), -v (show the commands), -n (only "
            "show them), -silent, -R2018a (interleaved complex data for mxGetData) and CFLAGS=, CXXFLAGS=, "
            "LDFLAGS= to replace the default flags. Building replaces a loaded copy of the same MEX file. "
            "MEX files compiled for MATLAB cannot be loaded and must be rebuilt from source with this command; "
            "MEX files do not work in the browser version of mylab. See docs/MEX.md."),
    "mexext": (["ext = mexext", "s = mexext('all')"],
               "The file extension of MEX files on this computer, without the dot: 'mexa64' on Linux, "
               "'mexmaci64' or 'mexmaca64' on macOS (Intel or Apple silicon) and 'mexw64' on Windows. "
               "mexext('all') returns a struct array with the arch and ext of each platform."),
    "mylabMexCall__": (["mylabMexCall__(file, nargout, ...)"],
                       "Internal: runs a MEX file. mylab calls it for each MEX file it finds on the path; it "
                       "is not meant to be called directly."),
}
