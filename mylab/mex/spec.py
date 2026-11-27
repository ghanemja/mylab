"""The table of functions mylab hands to a compiled MEX file.

A MEX file built by mylab's ``mex`` never links against a matrix library.  Its
header (``include/mylab_mex_api.h``) declares one struct of function pointers,
and every ``mx*``/``mex*`` call in the user's C code goes through it.  mylab
fills the struct with ctypes callbacks written in Python (see ``runtime.py``)
and passes its address to the MEX file each time the file is called.

This module is the single description of that struct.  The header is
generated from it by ``scripts/gen_mex_header.py`` and the ctypes callback
types are built from it at run time, so the two sides cannot drift apart
(a test checks that the committed header matches).

Each line of :data:`TABLE` is a C prototype.  Names that start with
``mylab_`` are internal: they appear in the struct but user code reaches them
only through the hand-written inline functions in ``matrix.h`` and ``mex.h``.
Every other name also gets a generated inline wrapper of the same name, which
calls through the table and then checks whether mylab flagged an error.
"""

from __future__ import annotations

import re

# Bump when the layout of the table or the meaning of an entry changes: MEX files built for another
# version are refused with a request to rebuild them.
ABI_VERSION = 1

TABLE = """
void mylab_print(const char *text);
void mylab_error(const char *identifier, const char *message);
void mylab_warning(const char *identifier, const char *message);
void *mylab_data(const mxArray *pa, int which);
int mylab_set_data(mxArray *pa, void *data, int which);
void *mylab_typed_data(const mxArray *pa, mxClassID classid, int complex);
int mylab_set_typed_data(mxArray *pa, void *data, mxClassID classid, int complex);

mxArray *mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity complexity);
mxArray *mxCreateDoubleScalar(double value);
mxArray *mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID classid, mxComplexity complexity);
mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims, mxClassID classid, mxComplexity complexity);
mxArray *mxCreateUninitNumericMatrix(size_t m, size_t n, mxClassID classid, mxComplexity complexity);
mxArray *mxCreateUninitNumericArray(size_t ndim, const size_t *dims, mxClassID classid, mxComplexity complexity);
mxArray *mxCreateLogicalScalar(bool value);
mxArray *mxCreateLogicalMatrix(mwSize m, mwSize n);
mxArray *mxCreateLogicalArray(mwSize ndim, const mwSize *dims);
mxArray *mxCreateString(const char *text);
mxArray *mxCreateCharArray(mwSize ndim, const mwSize *dims);
mxArray *mxCreateCharMatrixFromStrings(mwSize m, const char **rows);
mxArray *mxCreateCellMatrix(mwSize m, mwSize n);
mxArray *mxCreateCellArray(mwSize ndim, const mwSize *dims);
mxArray *mxCreateStructMatrix(mwSize m, mwSize n, int nfields, const char **fieldnames);
mxArray *mxCreateStructArray(mwSize ndim, const mwSize *dims, int nfields, const char **fieldnames);
mxArray *mxCreateSparse(mwSize m, mwSize n, mwSize nzmax, mxComplexity complexity);
mxArray *mxCreateSparseLogicalMatrix(mwSize m, mwSize n, mwSize nzmax);
mxArray *mxDuplicateArray(const mxArray *pa);
void mxDestroyArray(mxArray *pa);

size_t mxGetM(const mxArray *pa);
size_t mxGetN(const mxArray *pa);
void mxSetM(mxArray *pa, mwSize m);
void mxSetN(mxArray *pa, mwSize n);
size_t mxGetNumberOfElements(const mxArray *pa);
mwSize mxGetNumberOfDimensions(const mxArray *pa);
const mwSize *mxGetDimensions(const mxArray *pa);
int mxSetDimensions(mxArray *pa, const mwSize *dims, mwSize ndim);
size_t mxGetElementSize(const mxArray *pa);
mwIndex mxCalcSingleSubscript(const mxArray *pa, mwSize nsubs, const mwIndex *subs);
bool mxIsEmpty(const mxArray *pa);
bool mxIsScalar(const mxArray *pa);

mxClassID mxGetClassID(const mxArray *pa);
const char *mxGetClassName(const mxArray *pa);
bool mxIsClass(const mxArray *pa, const char *name);
bool mxIsComplex(const mxArray *pa);
bool mxIsSparse(const mxArray *pa);
bool mxIsLogicalScalarTrue(const mxArray *pa);

double *mxGetPr(const mxArray *pa);
double *mxGetPi(const mxArray *pa);
void mxSetPr(mxArray *pa, double *pr);
void mxSetPi(mxArray *pa, double *pi);
double mxGetScalar(const mxArray *pa);
mwIndex *mxGetIr(const mxArray *pa);
mwIndex *mxGetJc(const mxArray *pa);
void mxSetIr(mxArray *pa, mwIndex *ir);
void mxSetJc(mxArray *pa, mwIndex *jc);
mwSize mxGetNzmax(const mxArray *pa);
void mxSetNzmax(mxArray *pa, mwSize nzmax);

char *mxArrayToString(const mxArray *pa);
char *mxArrayToUTF8String(const mxArray *pa);
int mxGetString(const mxArray *pa, char *buf, mwSize buflen);

mxArray *mxGetCell(const mxArray *pa, mwIndex index);
void mxSetCell(mxArray *pa, mwIndex index, mxArray *value);
mxArray *mxGetField(const mxArray *pa, mwIndex index, const char *fieldname);
void mxSetField(mxArray *pa, mwIndex index, const char *fieldname, mxArray *value);
mxArray *mxGetFieldByNumber(const mxArray *pa, mwIndex index, int fieldnumber);
void mxSetFieldByNumber(mxArray *pa, mwIndex index, int fieldnumber, mxArray *value);
int mxGetNumberOfFields(const mxArray *pa);
const char *mxGetFieldNameByNumber(const mxArray *pa, int fieldnumber);
int mxGetFieldNumber(const mxArray *pa, const char *fieldname);
int mxAddField(mxArray *pa, const char *fieldname);
void mxRemoveField(mxArray *pa, int fieldnumber);

void *mxMalloc(size_t n);
void *mxCalloc(size_t n, size_t size);
void *mxRealloc(void *ptr, size_t size);
void mxFree(void *ptr);

int mexCallMATLAB(int nlhs, mxArray **plhs, int nrhs, mxArray **prhs, const char *name);
mxArray *mexCallMATLABWithTrap(int nlhs, mxArray **plhs, int nrhs, mxArray **prhs, const char *name);
int mexEvalString(const char *command);
mxArray *mexEvalStringWithTrap(const char *command);
void mexSetTrapFlag(int flag);
mxArray *mexGetVariable(const char *workspace, const char *name);
const mxArray *mexGetVariablePtr(const char *workspace, const char *name);
int mexPutVariable(const char *workspace, const char *name, const mxArray *pa);
void mexLock(void);
void mexUnlock(void);
bool mexIsLocked(void);
const char *mexFunctionName(void);
void mexMakeArrayPersistent(mxArray *pa);
void mexMakeMemoryPersistent(void *ptr);
"""

# which-argument values of mylab_data / mylab_set_data
DATA_REAL = 0          # real part, split storage (mxGetPr, mxGetData in the separate-complex API)
DATA_IMAG = 1          # imaginary part, split storage (mxGetPi, mxGetImagData)
DATA_INTERLEAVED = 2   # all of the data, complex values interleaved (mxGetData in the interleaved API)


class Entry:
    """One function of the table."""

    def __init__(self, ret: str, name: str, args: list[tuple[str, str]]):
        self.ret = ret
        self.name = name
        self.args = args  # (type, parameter name)

    @property
    def internal(self) -> bool:
        return self.name.startswith("mylab_")


def _norm(t: str) -> str:
    t = re.sub(r"\s*\*\s*", " *", t.strip())
    return re.sub(r"\s+", " ", t).replace(" *", "*")


def entries() -> list[Entry]:
    out = []
    for line in TABLE.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"(.+?[\s*])(\w+)\((.*)\);", line)
        if m is None:
            raise ValueError(f"bad prototype: {line}")
        ret, name, arglist = _norm(m.group(1)), m.group(2), m.group(3).strip()
        args = []
        if arglist and arglist != "void":
            for part in arglist.split(","):
                am = re.fullmatch(r"(.+?[\s*])(\w+)", part.strip())
                if am is None:
                    raise ValueError(f"bad parameter {part!r} in {line}")
                args.append((_norm(am.group(1)), am.group(2)))
        out.append(Entry(ret, name, args))
    return out


def c_decl(t: str, name: str = "") -> str:
    """A C declaration of ``name`` with the normalised type ``t``."""
    base = t.rstrip("*")
    stars = t[len(base):]
    if stars:
        return f"{base} {stars}{name}"
    return f"{t} {name}".rstrip()
