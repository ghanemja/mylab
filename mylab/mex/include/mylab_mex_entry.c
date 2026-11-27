/*
 * mylab_mex_entry.c - the entry points mylab calls in a MEX file.
 *
 * Written from scratch for mylab (MIT licence).  mylab's mex compiles this
 * file into every MEX file it builds (as C++ when any source is C++).  It
 * holds the pointer to mylab's function table and the code that runs
 * mexFunction and the mexAtExit function so that an error can unwind safely:
 *
 *   - In C, mylab_mex_entry calls setjmp before mexFunction, and
 *     mylab_mex_unwind longjmps back to it.  The Python side never sees the
 *     jump: by the time it happens, the Python callback that recorded the
 *     error has already returned.
 *   - In C++, mylab_mex_unwind throws a private exception type that the entry
 *     point catches, so destructors of objects on the stack run.  Any other
 *     C++ exception that escapes mexFunction becomes a mylab error too.
 *
 * The presence of the symbol mylab_mex_entry is how mylab tells its own MEX
 * files from ones built for MATLAB, which it cannot load.
 */
#include "mex.h"

#if defined(_WIN32)
#define MYLAB_EXPORT __declspec(dllexport)
#else
#define MYLAB_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#include <exception>
#define MYLAB_C extern "C"
#else
#include <setjmp.h>
#define MYLAB_C
#endif

/* (declared extern "C" in the headers, so these definitions keep C linkage in C++ too) */
mylab_mex_api *mylab_mex_table = 0;
void (*mylab_mex_atexit_fn)(void) = 0;

MYLAB_C MYLAB_EXPORT int mylab_mex_abi(void) { return MYLAB_MEX_ABI; }

#ifdef __cplusplus

struct mylab_mex_unwinding {};

/* The flag is cleared first: destructors that run while the exception travels may call mx/mex
 * functions themselves (mylab keeps the first error). */
MYLAB_C void mylab_mex_unwind(void)
{
    if (mylab_mex_table)
        mylab_mex_table->pending = 0;
    throw mylab_mex_unwinding();
}

typedef void (*mylab_body)(void *);

static int mylab_run(mylab_mex_api *api, mylab_body body, void *arg)
{
    mylab_mex_api *outer = mylab_mex_table;
    int status = 0;
    mylab_mex_table = api;
    try {
        body(arg);
    } catch (mylab_mex_unwinding &) {
        status = 1;
    } catch (std::exception &e) {
        api->mylab_error("mylab:mex:cppException", e.what());
        status = 1;
    } catch (...) {
        api->mylab_error("mylab:mex:cppException", "The MEX file threw a C++ exception.");
        status = 1;
    }
    mylab_mex_table = outer;
    return status;
}

#else

static jmp_buf *mylab_jump = 0;

MYLAB_C void mylab_mex_unwind(void)
{
    if (mylab_mex_table)
        mylab_mex_table->pending = 0;
    if (mylab_jump)
        longjmp(*mylab_jump, 1);
    abort(); /* an error with no entry point to return to: cannot happen through mylab */
}

typedef void (*mylab_body)(void *);

static int mylab_run(mylab_mex_api *api, mylab_body body, void *arg)
{
    jmp_buf here;
    jmp_buf *outer_jump = mylab_jump;          /* calls nest through mexCallMATLAB */
    mylab_mex_api *outer = mylab_mex_table;
    volatile int status = 0;
    mylab_mex_table = api;
    mylab_jump = &here;
    if (setjmp(here) == 0)
        body(arg);
    else
        status = 1;
    mylab_jump = outer_jump;
    mylab_mex_table = outer;
    return status;
}

#endif

typedef struct {
    int nlhs;
    mxArray **plhs;
    int nrhs;
    const mxArray **prhs;
} mylab_call;

static void mylab_call_body(void *arg)
{
    mylab_call *c = (mylab_call *)arg;
    mexFunction(c->nlhs, c->plhs, c->nrhs, c->prhs);
}

static void mylab_atexit_body(void *arg)
{
    void (*fn)(void) = mylab_mex_atexit_fn;
    (void)arg;
    mylab_mex_atexit_fn = 0;
    if (fn)
        fn();
}

/* Run mexFunction.  Returns 0, or 1 when it ended with an error (mylab holds the error). */
MYLAB_C MYLAB_EXPORT int mylab_mex_entry(mylab_mex_api *api, int nlhs, mxArray **plhs, int nrhs,
                                         const mxArray **prhs)
{
    mylab_call c;
    c.nlhs = nlhs;
    c.plhs = plhs;
    c.nrhs = nrhs;
    c.prhs = prhs;
    return mylab_run(api, mylab_call_body, &c);
}

/* Whether a function was registered with mexAtExit. */
MYLAB_C MYLAB_EXPORT int mylab_mex_has_atexit(void) { return mylab_mex_atexit_fn != 0; }

/* Run (once) the function registered with mexAtExit.  Returns 0, or 1 on an error. */
MYLAB_C MYLAB_EXPORT int mylab_mex_run_atexit(mylab_mex_api *api)
{
    return mylab_run(api, mylab_atexit_body, 0);
}
