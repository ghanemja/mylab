/* [y, z] = mx_callback(fname, x): y = fname(x) through mexCallMATLAB, z = [numel(sort(x)), workspace check].
 * Also prints with mexPrintf, evaluates a string in the caller and moves variables around.
 * Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char name[64];
    mxArray *in[1], *out[2], *v;
    (void)nlhs;
    if (nrhs != 2 || mxGetString(prhs[0], name, sizeof name) != 0)
        mexErrMsgIdAndTxt("fixture:callback:usage", "Usage: mx_callback(name, x)");
    in[0] = (mxArray *)prhs[1];
    mexCallMATLAB(1, out, 1, in, name);
    plhs[0] = out[0];
    mexPrintf("called %s from %s\n", name, mexFunctionName());
    mexEvalString("mex_seen_by_caller = 41;");
    v = mexGetVariable("caller", "mex_seen_by_caller");
    if (v == NULL)
        mexErrMsgTxt("variable not found");
    mxGetPr(v)[0] += 1;
    mexPutVariable("caller", "mex_put", v);
    mexPutVariable("base", "mex_put_base", mxCreateString("base!"));
    if (mexGetVariablePtr("caller", "no_such_variable_xyz") != NULL)
        mexErrMsgTxt("expected NULL for a missing variable");
    if (nlhs > 1) {
        mexCallMATLAB(2, out, 1, in, "sort");
        plhs[1] = mxCreateDoubleScalar((double)mxGetNumberOfElements(out[1]));
    }
}
