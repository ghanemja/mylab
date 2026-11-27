/* [B, I] = mx_sparse(A): B = 2*A for a sparse A (walking Ir/Jc/Pr), and I = speye(3) built with
 * mxCreateSparse.  Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwIndex j, k, *ir, *jc;
    double *pr;
    mwSize nnz;
    (void)nrhs;
    if (!mxIsSparse(prhs[0]))
        mexErrMsgIdAndTxt("fixture:sparse:full", "Input must be sparse.");
    plhs[0] = mxDuplicateArray(prhs[0]);
    jc = mxGetJc(plhs[0]);
    pr = mxGetPr(plhs[0]);
    nnz = jc[mxGetN(plhs[0])];
    for (k = 0; k < nnz; k++)
        pr[k] *= 2;
    if (nlhs > 1) {
        mxArray *e = mxCreateSparse(3, 3, 3, mxREAL);
        ir = mxGetIr(e);
        jc = mxGetJc(e);
        pr = mxGetPr(e);
        for (j = 0; j < 3; j++) {
            jc[j] = j;
            ir[j] = j;
            pr[j] = 1;
        }
        jc[3] = 3;
        plhs[1] = e;
    }
}
