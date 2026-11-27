/* mx_scale(x, k): x times k for any numeric class, keeping the class; a logical input is negated.
 * Second output: the class name.  Test fixture for mylab's MEX support. */
#include "mex.h"

#define SCALE(ctype) { ctype *p = (ctype *)mxGetData(out); for (i = 0; i < n; i++) p[i] = (ctype)(p[i] * k); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxArray *out;
    mwSize i, n;
    double k = 2;
    if (nrhs < 1)
        mexErrMsgIdAndTxt("fixture:scale:nargin", "mx_scale needs an input.");
    if (nrhs > 1)
        k = mxGetScalar(prhs[1]);
    out = mxDuplicateArray(prhs[0]);
    n = mxGetNumberOfElements(out);
    switch (mxGetClassID(out)) {
    case mxDOUBLE_CLASS: SCALE(double) break;
    case mxSINGLE_CLASS: SCALE(float) break;
    case mxINT8_CLASS: SCALE(mxInt8) break;
    case mxUINT8_CLASS: SCALE(mxUint8) break;
    case mxINT16_CLASS: SCALE(mxInt16) break;
    case mxUINT16_CLASS: SCALE(mxUint16) break;
    case mxINT32_CLASS: SCALE(mxInt32) break;
    case mxUINT32_CLASS: SCALE(mxUint32) break;
    case mxINT64_CLASS: SCALE(mxInt64) break;
    case mxUINT64_CLASS: SCALE(mxUint64) break;
    case mxLOGICAL_CLASS: {
        mxLogical *p = mxGetLogicals(out);
        for (i = 0; i < n; i++) p[i] = !p[i];
        break;
    }
    default:
        mexErrMsgIdAndTxt("fixture:scale:class", "mx_scale cannot scale a %s.", mxGetClassName(out));
    }
    plhs[0] = out;
    if (nlhs > 1)
        plhs[1] = mxCreateString(mxGetClassName(prhs[0]));
}
