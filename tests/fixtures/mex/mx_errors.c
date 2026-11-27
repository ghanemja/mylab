/* mx_errors(mode): raises errors and warnings in different ways.  Test fixture for mylab's MEX support.
 *   1: mexErrMsgIdAndTxt with a formatted message    2: mexErrMsgTxt (no identifier)
 *   3: a warning, then returns 7                    4: an error from code run through mexCallMATLAB
 *   5: an invalid mx call (mxSetCell on a double)    6: mexCallMATLAB with the trap flag set
 *   7: mexCallMATLABWithTrap, returns the identifier of the caught error */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int mode = (int)mxGetScalar(prhs[0]);
    mxArray *in[1], *out[1];
    (void)nlhs; (void)nrhs;
    switch (mode) {
    case 1:
        mexErrMsgIdAndTxt("fixture:errors:bad", "Bad value %d (%s).", 42, "forty-two");
    case 2:
        mexErrMsgTxt("Plain failure.");
    case 3:
        mexWarnMsgIdAndTxt("fixture:errors:warn", "Careful with %d.", 3);
        plhs[0] = mxCreateDoubleScalar(7);
        return;
    case 4: {
        mxArray *args[2];
        args[0] = mxCreateString("fixture:errors:inner");
        args[1] = mxCreateString("raised inside mexCallMATLAB");
        mexCallMATLAB(0, NULL, 2, args, "error");
    }
        mexErrMsgTxt("not reached");
    case 5:
        mxSetCell(mxCreateDoubleScalar(1), 0, NULL);
        mexErrMsgTxt("not reached");
    case 6:
        mexSetTrapFlag(1);
        in[0] = mxCreateString("no_such_function_xyz");
        plhs[0] = mxCreateDoubleScalar(mexCallMATLAB(1, out, 1, in, "feval"));
        mexSetTrapFlag(0);
        return;
    case 7: {
        mxArray *e, *id;
        in[0] = mxCreateDoubleScalar(-1);
        e = mexCallMATLABWithTrap(1, out, 1, in, "fixture_fails");
        if (e == NULL)
            mexErrMsgTxt("expected an MException");
        mexCallMATLAB(1, &id, 1, &e, "fixture_identifier");
        plhs[0] = id;
        return;
    }
    default:
        plhs[0] = mxCreateDoubleScalar(0);
    }
}
