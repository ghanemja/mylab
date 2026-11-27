/* [r, s] = mx_cells(c): r is the cell c in reverse order (elements duplicated); s is a 1-by-numel(c)
 * struct array with fields index and value, and a 'name' field read from a struct input when given.
 * Test fixture for mylab's MEX support. */
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    static const char *fields[] = {"index", "value"};
    mwSize i, n;
    mxArray *r, *s;
    (void)nlhs;
    if (nrhs < 1 || !mxIsCell(prhs[0]))
        mexErrMsgIdAndTxt("fixture:cells:notCell", "First input must be a cell array.");
    n = mxGetNumberOfElements(prhs[0]);
    r = mxCreateCellMatrix(mxGetM(prhs[0]), mxGetN(prhs[0]));
    s = mxCreateStructMatrix(1, n, 2, fields);
    for (i = 0; i < n; i++) {
        const mxArray *e = mxGetCell(prhs[0], i);
        if (e)
            mxSetCell(r, n - 1 - i, mxDuplicateArray(e));
        mxSetField(s, i, "index", mxCreateDoubleScalar((double)(i + 1)));
        if (e)
            mxSetFieldByNumber(s, i, 1, mxDuplicateArray(e));
    }
    if (nrhs > 1 && mxIsStruct(prhs[1])) {
        int k, nf = mxGetNumberOfFields(prhs[1]);
        mxArray *names = mxCreateCellMatrix(1, nf);
        for (k = 0; k < nf; k++)
            mxSetCell(names, k, mxCreateString(mxGetFieldNameByNumber(prhs[1], k)));
        mxAddField(s, "names");
        for (i = 0; i < n; i++)
            mxSetField(s, i, "names", i == 0 ? names : mxDuplicateArray(names));
        if (mxGetField(prhs[1], 0, "missing") != NULL)
            mexErrMsgTxt("mxGetField should return NULL for a missing field.");
    }
    plhs[0] = r;
    if (nlhs > 1)
        plhs[1] = s;
}
