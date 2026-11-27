/* w = mx_complex2018(z): z.^2 with the interleaved complex API (build with -R2018a); a single input
 * gives a single result.  Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize i, n = mxGetNumberOfElements(prhs[0]);
    (void)nlhs; (void)nrhs;
    if (!mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("fixture:complex:real", "Input must be complex.");
    if (mxIsSingle(prhs[0])) {
        mxComplexSingle *z = mxGetComplexSingles(prhs[0]), *w;
        plhs[0] = mxCreateNumericMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxSINGLE_CLASS, mxCOMPLEX);
        w = mxGetComplexSingles(plhs[0]);
        for (i = 0; i < n; i++) {
            w[i].real = z[i].real * z[i].real - z[i].imag * z[i].imag;
            w[i].imag = 2 * z[i].real * z[i].imag;
        }
    } else {
        mxComplexDouble *z = mxGetComplexDoubles(prhs[0]), *w;
        if (mxGetDoubles(prhs[0]) != NULL)
            mexErrMsgTxt("mxGetDoubles should be NULL for a complex array.");
        plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]), mxCOMPLEX);
        w = (mxComplexDouble *)mxGetData(plhs[0]);
        for (i = 0; i < n; i++) {
            w[i].real = z[i].real * z[i].real - z[i].imag * z[i].imag;
            w[i].imag = 2 * z[i].real * z[i].imag;
        }
    }
}
