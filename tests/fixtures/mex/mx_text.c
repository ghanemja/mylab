/* [up, rows, trunc, n] = mx_text(s): s in upper case (via mxArrayToString), a char matrix made from
 * strings, the first 3 characters through mxGetString (and its return value).
 * Test fixture for mylab's MEX support. */
#include <ctype.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *text, small[4];
    size_t i;
    int status;
    static const char *rows[] = {"alpha", "be", "gamma!"};
    (void)nlhs;
    if (nrhs != 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("fixture:text:notChar", "Input must be a char array.");
    text = mxArrayToString(prhs[0]);
    for (i = 0; text[i]; i++)
        text[i] = (char)toupper((unsigned char)text[i]);
    plhs[0] = mxCreateString(text);
    mxFree(text);
    plhs[1] = mxCreateCharMatrixFromStrings(3, rows);
    status = mxGetString(prhs[0], small, sizeof small);
    plhs[2] = mxCreateString(small);
    plhs[3] = mxCreateDoubleScalar(status);
    if (nlhs > 4) {
        mwSize dims[2] = {1, 2};
        mxArray *c = mxCreateCharArray(2, dims);
        mxChar *p = mxGetChars(c);
        p[0] = 0x3b1;  /* Greek alpha */
        p[1] = 'z';
        plhs[4] = c;
    }
}
