/*
 * colstats.c - column means and standard deviations, written in C as a MEX file.
 *
 *     [mu, sd] = colstats(X)
 *
 * X is a real double matrix; mu and sd are row vectors with one value per column
 * (sd uses n - 1, like std).  Build it with:  mex colstats.c
 *
 * Part of mylab's examples (MIT licence).
 */
#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize m, n, i, j;
    const double *x;
    double *mu, *sd;

    if (nrhs != 1)
        mexErrMsgIdAndTxt("colstats:nargin", "colstats takes exactly one input.");
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxIsSparse(prhs[0]))
        mexErrMsgIdAndTxt("colstats:type", "The input must be a real, full double matrix, not %s.",
                          mxGetClassName(prhs[0]));
    if (nlhs > 2)
        mexErrMsgIdAndTxt("colstats:nargout", "colstats returns at most two outputs.");

    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    x = mxGetPr(prhs[0]);   /* column-major: element (i, j) is x[i + j*m] */

    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    mu = mxGetPr(plhs[0]);
    sd = (double *)mxCalloc(n > 0 ? n : 1, sizeof(double));   /* scratch: released after the call */

    for (j = 0; j < n; j++) {
        double sum = 0, sq = 0;
        for (i = 0; i < m; i++)
            sum += x[i + j * m];
        mu[j] = m > 0 ? sum / (double)m : mxGetNaN();
        for (i = 0; i < m; i++)
            sq += (x[i + j * m] - mu[j]) * (x[i + j * m] - mu[j]);
        sd[j] = m > 1 ? sqrt(sq / (double)(m - 1)) : 0;
    }

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(1, n, mxREAL);
        for (j = 0; j < n; j++)
            mxGetPr(plhs[1])[j] = sd[j];
    }
    mxFree(sd);
}
