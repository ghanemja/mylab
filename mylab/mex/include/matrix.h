/*
 * matrix.h - array functions for MEX files built by mylab.
 *
 * Written from scratch for mylab (MIT licence).  It declares the same
 * function names and types that C code written for MATLAB's MEX interface
 * uses, so such code compiles unchanged, but nothing here links to a matrix
 * library: every call goes through a table of function pointers that mylab
 * fills in (from Python) when it loads the MEX file.  See docs/MEX.md.
 *
 * mxArray is an opaque handle.  The data pointers it hands out (mxGetPr,
 * mxGetData, mxGetDoubles, ...) point into memory owned by mylab, laid out
 * in column-major order, and stay valid until the array is destroyed.
 */
#ifndef MYLAB_MATRIX_H
#define MYLAB_MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
#define MYLAB_EXTERN_C extern "C"
#else
#define MYLAB_EXTERN_C extern
#endif

#if defined(_MSC_VER)
#define MYLAB_INLINE static __inline
#define MYLAB_NORETURN __declspec(noreturn)
#elif defined(__GNUC__) || defined(__clang__)
#define MYLAB_INLINE static __inline__
#define MYLAB_NORETURN __attribute__((noreturn))
#else
#define MYLAB_INLINE static inline
#define MYLAB_NORETURN
#endif

#define MYLAB_MEX 1   /* lets portable code tell it is being built by mylab */

/* ---- types ------------------------------------------------------------ */

typedef struct mylab_mxArray_tag mxArray;   /* never defined: handles only */

typedef size_t mwSize;          /* sizes and dimensions (always 64-bit on 64-bit systems) */
typedef size_t mwIndex;         /* indices, including sparse row and column indices */
typedef ptrdiff_t mwSignedIndex;

typedef uint16_t mxChar;        /* one UTF-16 code unit */
typedef bool mxLogical;         /* one byte, 0 or 1 */

typedef double mxDouble;
typedef float mxSingle;
typedef int8_t mxInt8;
typedef uint8_t mxUint8;
typedef int16_t mxInt16;
typedef uint16_t mxUint16;
typedef int32_t mxInt32;
typedef uint32_t mxUint32;
typedef int64_t mxInt64;
typedef uint64_t mxUint64;

typedef struct { mxDouble real, imag; } mxComplexDouble;
typedef struct { mxSingle real, imag; } mxComplexSingle;
typedef struct { mxInt8 real, imag; } mxComplexInt8;
typedef struct { mxUint8 real, imag; } mxComplexUint8;
typedef struct { mxInt16 real, imag; } mxComplexInt16;
typedef struct { mxUint16 real, imag; } mxComplexUint16;
typedef struct { mxInt32 real, imag; } mxComplexInt32;
typedef struct { mxUint32 real, imag; } mxComplexUint32;
typedef struct { mxInt64 real, imag; } mxComplexInt64;
typedef struct { mxUint64 real, imag; } mxComplexUint64;

typedef enum {
    mxUNKNOWN_CLASS = 0,
    mxCELL_CLASS,
    mxSTRUCT_CLASS,
    mxLOGICAL_CLASS,
    mxCHAR_CLASS,
    mxVOID_CLASS,
    mxDOUBLE_CLASS,
    mxSINGLE_CLASS,
    mxINT8_CLASS,
    mxUINT8_CLASS,
    mxINT16_CLASS,
    mxUINT16_CLASS,
    mxINT32_CLASS,
    mxUINT32_CLASS,
    mxINT64_CLASS,
    mxUINT64_CLASS,
    mxFUNCTION_CLASS,
    mxOPAQUE_CLASS,
    mxOBJECT_CLASS
} mxClassID;

#define mxINDEX_CLASS mxUINT64_CLASS

typedef enum { mxREAL = 0, mxCOMPLEX = 1 } mxComplexity;

/* Which complex API the code was written for.  Build with "mex -R2018a" (or define
 * MX_HAS_INTERLEAVED_COMPLEX to 1) for interleaved complex data; the default is the older
 * separate real/imaginary arrays.  Only mxGetData, mxSetData, mxGetImagData and mxSetImagData
 * behave differently; the typed accessors (mxGetComplexDoubles, ...) and mxGetPr/mxGetPi are
 * available in both. */
#ifndef MX_HAS_INTERLEAVED_COMPLEX
#define MX_HAS_INTERLEAVED_COMPLEX 0
#endif

#include "mylab_mex_api.h"

/* ---- data pointers ------------------------------------------------------ */

#if MX_HAS_INTERLEAVED_COMPLEX
MYLAB_INLINE void *mxGetData(const mxArray *pa)
{ void *r = mylab_mex_table->mylab_data(pa, 2); MYLAB_MEX_CHECK(); return r; }
MYLAB_INLINE void mxSetData(mxArray *pa, void *data)
{ mylab_mex_table->mylab_set_data(pa, data, 2); MYLAB_MEX_CHECK(); }
#else
MYLAB_INLINE void *mxGetData(const mxArray *pa)
{ void *r = mylab_mex_table->mylab_data(pa, 0); MYLAB_MEX_CHECK(); return r; }
MYLAB_INLINE void mxSetData(mxArray *pa, void *data)
{ mylab_mex_table->mylab_set_data(pa, data, 0); MYLAB_MEX_CHECK(); }
#endif
MYLAB_INLINE void *mxGetImagData(const mxArray *pa)
{ void *r = mylab_mex_table->mylab_data(pa, 1); MYLAB_MEX_CHECK(); return r; }
MYLAB_INLINE void mxSetImagData(mxArray *pa, void *data)
{ mylab_mex_table->mylab_set_data(pa, data, 1); MYLAB_MEX_CHECK(); }

/* Typed accessors: NULL unless the array has exactly that class and complexity. */
#define MYLAB_TYPED(ctype, suffix, cls, cplx) \
    MYLAB_INLINE ctype *mxGet##suffix(const mxArray *pa) \
    { ctype *r = (ctype *)mylab_mex_table->mylab_typed_data(pa, cls, cplx); MYLAB_MEX_CHECK(); return r; } \
    MYLAB_INLINE int mxSet##suffix(mxArray *pa, ctype *data) \
    { int r = mylab_mex_table->mylab_set_typed_data(pa, (void *)data, cls, cplx); MYLAB_MEX_CHECK(); return r; }

MYLAB_TYPED(mxDouble, Doubles, mxDOUBLE_CLASS, 0)
MYLAB_TYPED(mxSingle, Singles, mxSINGLE_CLASS, 0)
MYLAB_TYPED(mxInt8, Int8s, mxINT8_CLASS, 0)
MYLAB_TYPED(mxUint8, Uint8s, mxUINT8_CLASS, 0)
MYLAB_TYPED(mxInt16, Int16s, mxINT16_CLASS, 0)
MYLAB_TYPED(mxUint16, Uint16s, mxUINT16_CLASS, 0)
MYLAB_TYPED(mxInt32, Int32s, mxINT32_CLASS, 0)
MYLAB_TYPED(mxUint32, Uint32s, mxUINT32_CLASS, 0)
MYLAB_TYPED(mxInt64, Int64s, mxINT64_CLASS, 0)
MYLAB_TYPED(mxUint64, Uint64s, mxUINT64_CLASS, 0)
MYLAB_TYPED(mxComplexDouble, ComplexDoubles, mxDOUBLE_CLASS, 1)
MYLAB_TYPED(mxComplexSingle, ComplexSingles, mxSINGLE_CLASS, 1)
MYLAB_TYPED(mxComplexInt8, ComplexInt8s, mxINT8_CLASS, 1)
MYLAB_TYPED(mxComplexUint8, ComplexUint8s, mxUINT8_CLASS, 1)
MYLAB_TYPED(mxComplexInt16, ComplexInt16s, mxINT16_CLASS, 1)
MYLAB_TYPED(mxComplexUint16, ComplexUint16s, mxUINT16_CLASS, 1)
MYLAB_TYPED(mxComplexInt32, ComplexInt32s, mxINT32_CLASS, 1)
MYLAB_TYPED(mxComplexUint32, ComplexUint32s, mxUINT32_CLASS, 1)
MYLAB_TYPED(mxComplexInt64, ComplexInt64s, mxINT64_CLASS, 1)
MYLAB_TYPED(mxComplexUint64, ComplexUint64s, mxUINT64_CLASS, 1)
MYLAB_TYPED(mxLogical, Logicals, mxLOGICAL_CLASS, 0)
MYLAB_TYPED(mxChar, Chars, mxCHAR_CLASS, 0)
#undef MYLAB_TYPED

/* ---- class tests (answered from mxGetClassID, without another trip to mylab) ---- */

#define MYLAB_IS(name, cls) \
    MYLAB_INLINE bool mxIs##name(const mxArray *pa) { return mxGetClassID(pa) == cls; }
MYLAB_IS(Double, mxDOUBLE_CLASS)
MYLAB_IS(Single, mxSINGLE_CLASS)
MYLAB_IS(Int8, mxINT8_CLASS)
MYLAB_IS(Uint8, mxUINT8_CLASS)
MYLAB_IS(Int16, mxINT16_CLASS)
MYLAB_IS(Uint16, mxUINT16_CLASS)
MYLAB_IS(Int32, mxINT32_CLASS)
MYLAB_IS(Uint32, mxUINT32_CLASS)
MYLAB_IS(Int64, mxINT64_CLASS)
MYLAB_IS(Uint64, mxUINT64_CLASS)
MYLAB_IS(Char, mxCHAR_CLASS)
MYLAB_IS(Logical, mxLOGICAL_CLASS)
MYLAB_IS(Cell, mxCELL_CLASS)
MYLAB_IS(Struct, mxSTRUCT_CLASS)
MYLAB_IS(FunctionHandle, mxFUNCTION_CLASS)
#undef MYLAB_IS

MYLAB_INLINE bool mxIsNumeric(const mxArray *pa)
{ mxClassID c = mxGetClassID(pa); return c >= mxDOUBLE_CLASS && c <= mxUINT64_CLASS; }
MYLAB_INLINE bool mxIsLogicalScalar(const mxArray *pa)
{ return mxIsLogical(pa) && mxGetNumberOfElements(pa) == 1; }
MYLAB_INLINE bool mxIsObject(const mxArray *pa) { return mxGetClassID(pa) == mxOBJECT_CLASS; }
MYLAB_INLINE bool mxIsOpaque(const mxArray *pa) { return mxGetClassID(pa) == mxOPAQUE_CLASS; }
MYLAB_INLINE bool mxIsFromGlobalWS(const mxArray *pa) { (void)pa; return false; }

/* ---- special values (plain C, no call to mylab) ---- */

MYLAB_INLINE double mxGetInf(void) { return HUGE_VAL; }
#ifdef NAN
MYLAB_INLINE double mxGetNaN(void) { return (double)NAN; }
#else
MYLAB_INLINE double mxGetNaN(void) { return HUGE_VAL - HUGE_VAL; }
#endif
MYLAB_INLINE double mxGetEps(void) { return DBL_EPSILON; }
MYLAB_INLINE bool mxIsInf(double x) { return x == HUGE_VAL || x == -HUGE_VAL; }
MYLAB_INLINE bool mxIsNaN(double x) { return x != x; }
MYLAB_INLINE bool mxIsFinite(double x) { return !mxIsInf(x) && !mxIsNaN(x); }

/* Older spellings some code still uses. */
#define mxCreateScalarDouble mxCreateDoubleScalar

#endif /* MYLAB_MATRIX_H */
