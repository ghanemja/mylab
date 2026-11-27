/* [w, re, im] = mx_complex(z): w = 2*conj(z) using the separate real/imaginary API (mxGetPr/mxGetPi),
 * plus the real and imaginary parts as real arrays.  Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize i, n, m = mxGetM(prhs[0]), c = mxGetN(prhs[0]);
    double *zr, *zi, *wr, *wi;
    (void)nrhs;
    n = mxGetNumberOfElements(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(m, c, mxCOMPLEX);
    zr = mxGetPr(prhs[0]);
    zi = mxGetPi(prhs[0]);   /* NULL for a real input */
    wr = mxGetPr(plhs[0]);
    wi = mxGetPi(plhs[0]);
    for (i = 0; i < n; i++) {
        wr[i] = 2 * zr[i];
        wi[i] = zi ? -2 * zi[i] : 0;
    }
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(m, c, mxREAL);
        for (i = 0; i < n; i++) mxGetPr(plhs[1])[i] = zr[i];
    }
    if (nlhs > 2) {
        plhs[2] = mxCreateDoubleMatrix(m, c, mxREAL);
        for (i = 0; i < n; i++) mxGetPr(plhs[2])[i] = zi ? zi[i] : 0;
    }
}
