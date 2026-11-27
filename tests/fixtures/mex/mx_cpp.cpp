// mx_cpp(mode): C++ MEX file.  mode 0: sum of the input with std::vector; 1: throws std::runtime_error;
// 2: mexErrMsgTxt with an object on the stack whose destructor prints.  Test fixture for mylab's MEX support.
#include <stdexcept>
#include <vector>
#include "mex.h"

struct Noisy {
    ~Noisy() { mexPrintf("destructor ran\n"); }
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)nrhs;
    int mode = static_cast<int>(mxGetScalar(prhs[0]));
    if (mode == 1)
        throw std::runtime_error("thrown from C++");
    if (mode == 2) {
        Noisy n;
        mexErrMsgIdAndTxt("fixture:cpp:err", "error with an object on the stack");
    }
    std::vector<double> v(mxGetPr(prhs[1]), mxGetPr(prhs[1]) + mxGetNumberOfElements(prhs[1]));
    double s = 0;
    for (double x : v) s += x;
    plhs[0] = mxCreateDoubleScalar(s);
}
