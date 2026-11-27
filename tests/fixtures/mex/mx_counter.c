/* n = mx_counter(cmd): counts calls with state kept between calls: a static counter, a persistent
 * mxArray and a block of persistent mxMalloc memory.  mexAtExit prints a message when it is cleared.
 * cmd: 'lock' / 'unlock' / 'locked' / anything else just counts.
 * Test fixture for mylab's MEX support. */
#include <string.h>
#include "mex.h"

static int calls = 0;
static mxArray *history = NULL;
static double *memory = NULL;

static void goodbye(void)
{
    mexPrintf("mx_counter: goodbye after %d calls\n", calls);
    if (history)
        mxDestroyArray(history);
    if (memory)
        mxFree(memory);
    history = NULL;
    memory = NULL;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char cmd[16] = "";
    (void)nlhs;
    if (nrhs > 0)
        mxGetString(prhs[0], cmd, sizeof cmd);
    if (strcmp(cmd, "lock") == 0) { mexLock(); return; }
    if (strcmp(cmd, "unlock") == 0) { mexUnlock(); return; }
    if (strcmp(cmd, "locked") == 0) { plhs[0] = mxCreateLogicalScalar(mexIsLocked()); return; }
    if (history == NULL) {
        history = mxCreateDoubleMatrix(1, 1, mxREAL);
        mexMakeArrayPersistent(history);
        memory = (double *)mxCalloc(1, sizeof(double));
        mexMakeMemoryPersistent(memory);
        mexAtExit(goodbye);
    }
    calls++;
    mxGetPr(history)[0] += 10;
    memory[0] += 100;
    plhs[0] = mxCreateDoubleMatrix(1, 3, mxREAL);
    mxGetPr(plhs[0])[0] = calls;
    mxGetPr(plhs[0])[1] = mxGetPr(history)[0];
    mxGetPr(plhs[0])[2] = memory[0];
}
