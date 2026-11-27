/*
 * mex.h - the MEX interface for C and C++ files built with mylab's mex.
 *
 * Written from scratch for mylab (MIT licence).  Include it in the file that
 * defines mexFunction, the function mylab calls:
 *
 *     void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
 *
 * nrhs/prhs are the inputs (read them, do not change them); nlhs is the number
 * of outputs the caller asked for, and the function stores new arrays in
 * plhs[0] .. plhs[nlhs-1] (plhs[0] may be set even when nlhs is 0: it becomes
 * ans).
 *
 * Errors: mexErrMsgTxt and mexErrMsgIdAndTxt never return.  mylab records the
 * error, then control jumps straight back to the code that called mexFunction
 * (setjmp/longjmp in C, a C++ exception in C++), and the error is raised in
 * the calling program, where try/catch can handle it.  The same happens when
 * any other mx/mex function fails (for example mexCallMATLAB calling code that
 * raises an error).  Memory from mxMalloc/mxCalloc and arrays that were not
 * returned are released after every call, so the jump leaks nothing that
 * mylab manages.  See docs/MEX.md.
 */
#ifndef MYLAB_MEX_H
#define MYLAB_MEX_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/* The function registered with mexAtExit (defined in mylab_mex_entry.c). */
extern void (*mylab_mex_atexit_fn)(void);

#ifdef __cplusplus
}
#endif

/* Strict C89 builds (-ansi) hide these C99 names; the C library has them anyway. */
#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L) && !defined(_MSC_VER)
extern int vsnprintf(char *buf, size_t size, const char *fmt, va_list ap);
#endif
#if !defined(va_copy) && defined(__va_copy)
#define va_copy __va_copy
#endif
#ifndef va_copy
#define va_copy(dst, src) memcpy(&(dst), &(src), sizeof(va_list))
#endif

/* Format with vsnprintf into memory from mxMalloc (released after the call). */
MYLAB_INLINE char *mylab_mex_vformat(const char *fmt, va_list ap)
{
    char small[512];
    char *buf;
    int n;
    va_list again;
    va_copy(again, ap);
    n = vsnprintf(small, sizeof small, fmt, ap);
    if (n < 0) {
        va_end(again);
        n = 0;
        small[0] = '\0';
    }
    buf = (char *)mxMalloc((size_t)n + 1);
    if ((size_t)n < sizeof small)
        memcpy(buf, small, (size_t)n + 1);
    else
        vsnprintf(buf, (size_t)n + 1, fmt, again);
    va_end(again);
    return buf;
}

MYLAB_INLINE int mexPrintf(const char *fmt, ...)
{
    va_list ap;
    char *text;
    int n;
    va_start(ap, fmt);
    text = mylab_mex_vformat(fmt, ap);
    va_end(ap);
    n = (int)strlen(text);
    mylab_mex_table->mylab_print(text);
    mxFree(text);
    MYLAB_MEX_CHECK();
    return n;
}

MYLAB_NORETURN MYLAB_INLINE void mexErrMsgTxt(const char *message)
{
    mylab_mex_table->mylab_error("", message);
    mylab_mex_unwind();
}

MYLAB_NORETURN MYLAB_INLINE void mexErrMsgIdAndTxt(const char *identifier, const char *fmt, ...)
{
    va_list ap;
    char *text;
    va_start(ap, fmt);
    text = mylab_mex_vformat(fmt, ap);
    va_end(ap);
    mylab_mex_table->mylab_error(identifier, text);
    mylab_mex_unwind();
}

MYLAB_INLINE void mexWarnMsgTxt(const char *message)
{
    mylab_mex_table->mylab_warning("", message);
    MYLAB_MEX_CHECK();
}

MYLAB_INLINE void mexWarnMsgIdAndTxt(const char *identifier, const char *fmt, ...)
{
    va_list ap;
    char *text;
    va_start(ap, fmt);
    text = mylab_mex_vformat(fmt, ap);
    va_end(ap);
    mylab_mex_table->mylab_warning(identifier, text);
    mxFree(text);
    MYLAB_MEX_CHECK();
}

/* Register a function to run when the MEX file is cleared (clear name, clear mex, clear all)
 * or when mylab exits.  Registering again replaces the earlier function. */
MYLAB_INLINE int mexAtExit(void (*fn)(void))
{
    mylab_mex_atexit_fn = fn;
    return 0;
}

MYLAB_INLINE bool mexIsGlobal(const mxArray *pa) { (void)pa; return false; }

/* mxAssert / mxAssertS are active only in builds made with "mex -g". */
#ifdef MYLAB_MEX_DEBUG
#define mxAssert(expr, msg) \
    do { if (!(expr)) mexErrMsgIdAndTxt("mylab:mex:assertion", "Assertion failed: %s (%s), %s line %d.", \
                                        #expr, (msg), __FILE__, __LINE__); } while (0)
#define mxAssertS(expr, msg) \
    do { if (!(expr)) mexErrMsgIdAndTxt("mylab:mex:assertion", "Assertion failed: %s, %s line %d.", \
                                        (msg), __FILE__, __LINE__); } while (0)
#else
#define mxAssert(expr, msg) ((void)0)
#define mxAssertS(expr, msg) ((void)0)
#endif

#endif /* MYLAB_MEX_H */
