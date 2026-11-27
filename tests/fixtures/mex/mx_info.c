/* mx_info(x): a struct describing x (sizes, class, predicates), built with the mx API.
 * Test fixture for mylab's MEX support. */
#include "mex.h"

static void set_num(mxArray *s, const char *field, double v)
{
    mxSetField(s, 0, field, mxCreateDoubleScalar(v));
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    static const char *fields[] = {"m", "n", "numel", "ndims", "class", "dims"};
    const mxArray *x;
    mxArray *s, *dims;
    const mwSize *d;
    mwSize k, nd;
    double *pd;
    (void)nlhs;
    if (nrhs != 1)
        mexErrMsgTxt("mx_info takes one input.");
    x = prhs[0];
    s = mxCreateStructMatrix(1, 1, 6, fields);
    set_num(s, "m", (double)mxGetM(x));
    set_num(s, "n", (double)mxGetN(x));
    set_num(s, "numel", (double)mxGetNumberOfElements(x));
    nd = mxGetNumberOfDimensions(x);
    set_num(s, "ndims", (double)nd);
    mxSetField(s, 0, "class", mxCreateString(mxGetClassName(x)));
    d = mxGetDimensions(x);
    dims = mxCreateDoubleMatrix(1, nd, mxREAL);
    pd = mxGetPr(dims);
    for (k = 0; k < nd; k++)
        pd[k] = (double)d[k];
    mxSetFieldByNumber(s, 0, mxGetFieldNumber(s, "dims"), dims);
    k = mxAddField(s, "flags");
    {
        char flags[16];
        int j = 0;
        if (mxIsNumeric(x)) flags[j++] = 'N';
        if (mxIsComplex(x)) flags[j++] = 'C';
        if (mxIsSparse(x)) flags[j++] = 'S';
        if (mxIsChar(x)) flags[j++] = 'T';
        if (mxIsLogical(x)) flags[j++] = 'L';
        if (mxIsCell(x)) flags[j++] = 'c';
        if (mxIsStruct(x)) flags[j++] = 's';
        if (mxIsEmpty(x)) flags[j++] = 'E';
        if (mxIsScalar(x)) flags[j++] = '1';
        if (mxIsFunctionHandle(x)) flags[j++] = 'F';
        if (mxIsDouble(x)) flags[j++] = 'D';
        flags[j] = '\0';
        mxSetFieldByNumber(s, 0, (int)k, mxCreateString(flags));
    }
    plhs[0] = s;
}
