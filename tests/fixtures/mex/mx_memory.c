/* y = mx_memory(n): builds 1..n in mxMalloc'd memory, grows it with mxRealloc, and hands it to the
 * result with mxSetPr; also makes and destroys scratch arrays.  Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize i, n = (mwSize)mxGetScalar(prhs[0]);
    double *p = (double *)mxMalloc(1 * sizeof(double));
    mxArray *scratch = mxCreateDoubleMatrix(100, 100, mxREAL);
    (void)nlhs; (void)nrhs;
    mxDestroyArray(scratch);
    scratch = mxCreateCellMatrix(1, 2);  /* not destroyed: mylab frees it after the call */
    mxSetCell(scratch, 0, mxCreateString("tmp"));
    p = (double *)mxRealloc(p, n * sizeof(double));
    for (i = 0; i < n; i++)
        p[i] = (double)(i + 1);
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
    mxSetPr(plhs[0], p);
    mxSetM(plhs[0], 1);
    mxSetN(plhs[0], n);
}
