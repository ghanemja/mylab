/* varargout = mx_nargs(varargin): output k is k*10 + nrhs; prints nlhs and nrhs.
 * mx_nargs('skip') with an output requested leaves it unassigned.
 * Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int k, n = nlhs > 0 ? nlhs : 1;
    mexPrintf("nlhs=%d nrhs=%d\n", nlhs, nrhs);
    if (nrhs == 1 && mxIsChar(prhs[0]))
        return;
    for (k = 0; k < n; k++)
        plhs[k] = mxCreateDoubleScalar((k + 1) * 10 + nrhs);
}
