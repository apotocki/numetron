// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

// The one place that sets which implementation of each algorithm the library uses. Every choice
// below is only a default: define the macro yourself (compiler flag, or #define before including
// any numetron header) to override it. No other header defines these.
// Tuning numbers are separate: the multiplication thresholds are in
// limb_arithmetic/toom/thresholds.hpp, the division ones in limb_arithmetic.hpp.

// ---- Assembly ---------------------------------------------------------------------------------
// NUMETRON_USE_ASM: off by default, and numetron is then a pure header-only library. Define it to
// use the src/arch assembly (x86-64: mul_basecase, add/sub_n, the Karatsuba kernels); the
// numetron static library built from src/arch must then be linked. The CMake target `numetron`
// (lib/CMakeLists.txt) exports this define to whatever links it on x86-64; the MSVC test and bench
// projects set it themselves.

// Which mul_basecase: picked at run time from CPUID (NUMETRON_PLATFORM_AUTODETECT) unless one is
// pinned with NUMETRON_PLATFORM_ALDERLAKE, NUMETRON_PLATFORM_CORE2 or NUMETRON_PLATFORM_K8.
#if !defined(NUMETRON_PLATFORM_AUTODETECT) && !defined(NUMETRON_PLATFORM_ALDERLAKE) \
    && !defined(NUMETRON_PLATFORM_CORE2) && !defined(NUMETRON_PLATFORM_K8)
#   define NUMETRON_PLATFORM_AUTODETECT
#endif

// ---- Karatsuba --------------------------------------------------------------------------------
// NUMETRON_KARATSUBA_IMPL, one of:
#define NUMETRON_KARATSUBA_IMPL_CXX    1 // umul_karatsuba_impl (umul_karatsuba.hpp)
#define NUMETRON_KARATSUBA_IMPL_FUSED  2 // umul_karatsuba_fused_impl: the middle-column passes of
                                         // the interpolation fused into one (asm kernel with
                                         // NUMETRON_USE_ASM)
#define NUMETRON_KARATSUBA_IMPL_ASM    3 // umul_karatsuba_asm_impl: the whole recursion in
                                         // assembly; x86-64 only (experiment)
#define NUMETRON_KARATSUBA_IMPL_ENGINE 4 // the Toom engine's 2 x 2 plan
// e.g. -DNUMETRON_KARATSUBA_IMPL=NUMETRON_KARATSUBA_IMPL_FUSED. The shorthand flags
// NUMETRON_KARATSUBA_FUSED and NUMETRON_KARATSUBA_ASM select the same (ASM wins if both are set).
// Default: ASM wherever the assembly is in use (NUMETRON_USE_ASM on x86-64), CXX otherwise.
#ifndef NUMETRON_KARATSUBA_IMPL
#   if defined(NUMETRON_KARATSUBA_ASM)
#       define NUMETRON_KARATSUBA_IMPL NUMETRON_KARATSUBA_IMPL_ASM
#   elif defined(NUMETRON_KARATSUBA_FUSED)
#       define NUMETRON_KARATSUBA_IMPL NUMETRON_KARATSUBA_IMPL_FUSED
#   elif defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
#       define NUMETRON_KARATSUBA_IMPL NUMETRON_KARATSUBA_IMPL_ASM // default with the assembly
#   else
#       define NUMETRON_KARATSUBA_IMPL NUMETRON_KARATSUBA_IMPL_CXX // default without it
#   endif
#endif

#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM \
    && !(defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64)))
#   error "NUMETRON_KARATSUBA_IMPL_ASM needs NUMETRON_USE_ASM on x86-64"
#endif

// ---- Toom-3 (balanced operands) ---------------------------------------------------------------
// NUMETRON_TOOM3_IMPL, one of:
#define NUMETRON_TOOM3_IMPL_CXX    1 // the hand-written umul_toom3_impl (umul_toom3.hpp)
#define NUMETRON_TOOM3_IMPL_ENGINE 2 // the same algorithm as the engine's toom3_balanced plan
// The shorthand flag NUMETRON_TOOM3_USE_ENGINE selects ENGINE.
#ifndef NUMETRON_TOOM3_IMPL
#   if defined(NUMETRON_TOOM3_USE_ENGINE)
#       define NUMETRON_TOOM3_IMPL NUMETRON_TOOM3_IMPL_ENGINE
#   else
#       define NUMETRON_TOOM3_IMPL NUMETRON_TOOM3_IMPL_CXX // default
#   endif
#endif

// ---- Division ---------------------------------------------------------------------------------
// Estimate the quotient digits in udiv() with a precomputed reciprocal (Möller-Granlund) instead
// of a hardware 2/1 division per digit. Measured on x86-64 that is 10-38% faster over the whole
// size range, most of it on small divisors. Define NUMETRON_ARITHMETIC_NO_INVINT_DIV to opt out.
#if !defined(NUMETRON_ARITHMETIC_NO_INVINT_DIV) && !defined(NUMETRON_ARITHMETIC_USE_INVINT_DIV)
#   define NUMETRON_ARITHMETIC_USE_INVINT_DIV
#endif

namespace numetron::config {

// The choices above as text, for benchmarks and diagnostics.
inline constexpr const char* karatsuba_impl_name =
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_CXX
    "cxx";
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_FUSED
    "fused";
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
    "asm";
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ENGINE
    "engine";
#else
#   error "unknown NUMETRON_KARATSUBA_IMPL"
#endif

inline constexpr const char* toom3_impl_name =
#if NUMETRON_TOOM3_IMPL == NUMETRON_TOOM3_IMPL_CXX
    "cxx";
#elif NUMETRON_TOOM3_IMPL == NUMETRON_TOOM3_IMPL_ENGINE
    "engine";
#else
#   error "unknown NUMETRON_TOOM3_IMPL"
#endif

inline constexpr const char* mul_basecase_name =
#if !defined(NUMETRON_USE_ASM) || !(defined(__x86_64__) || defined(_M_X64))
    "c++";
#elif defined(NUMETRON_PLATFORM_ALDERLAKE)
    "asm alderlake";
#elif defined(NUMETRON_PLATFORM_CORE2)
    "asm core2";
#elif defined(NUMETRON_PLATFORM_K8)
    "asm k8";
#else
    "asm, runtime-detected";
#endif

}
