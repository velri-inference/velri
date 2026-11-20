#include "simd/x86.h"

/* The same traversal is instantiated for each accumulator type and ISA.
 * Native BF16/INT8 dot products retain their own arithmetic and packed layouts. */
#define GEMV_F64 0
#include "gemv.inc"
#undef GEMV_F64
#define GEMV_F64 1
#include "gemv.inc"
#undef GEMV_F64
