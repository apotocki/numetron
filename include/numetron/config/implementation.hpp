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

// NUMETRON_ASM_LICENSE: which assembly NUMETRON_USE_ASM may bring in, one of:
#define NUMETRON_ASM_LICENSE_MIT      1 // numetron's own code only (MIT): the mul_basecase is
                                        // mul_basecase_adx on CPUs with BMI2 + ADX, the C++
                                        // basecase (NUMETRON_CXX_BASECASE) on older ones
#define NUMETRON_ASM_LICENSE_GMP_LGPL 2 // also the GMP-derived mul_basecase routines
                                        // (src/arch/x86_64/{alderlake,core2,k8}, LGPL): picked
                                        // per CPU; a binary using them is subject to the LGPL
// The shorthand flag NUMETRON_USE_GMP_LGPL selects GMP_LGPL (the CMake option NUMETRON_GMP_LGPL
// builds the LGPL sources into the `numetron` library and exports that flag). Default: MIT.
#ifndef NUMETRON_ASM_LICENSE
#   if defined(NUMETRON_USE_GMP_LGPL)
#       define NUMETRON_ASM_LICENSE NUMETRON_ASM_LICENSE_GMP_LGPL
#   else
#       define NUMETRON_ASM_LICENSE NUMETRON_ASM_LICENSE_MIT
#   endif
#endif

// Which mul_basecase: picked at run time from CPUID (NUMETRON_PLATFORM_AUTODETECT; among the
// licenses allowed above) unless one is pinned with NUMETRON_PLATFORM_ADX (our own, MIT; needs
// BMI2 + ADX), or with NUMETRON_PLATFORM_ALDERLAKE, NUMETRON_PLATFORM_CORE2 or
// NUMETRON_PLATFORM_K8 (the GMP-derived ones; need NUMETRON_ASM_LICENSE_GMP_LGPL).
#if !defined(NUMETRON_PLATFORM_AUTODETECT) && !defined(NUMETRON_PLATFORM_ALDERLAKE) \
    && !defined(NUMETRON_PLATFORM_CORE2) && !defined(NUMETRON_PLATFORM_K8) && !defined(NUMETRON_PLATFORM_ADX)
#   define NUMETRON_PLATFORM_AUTODETECT
#endif

#if defined(NUMETRON_USE_ASM) && NUMETRON_ASM_LICENSE != NUMETRON_ASM_LICENSE_GMP_LGPL \
    && (defined(NUMETRON_PLATFORM_ALDERLAKE) || defined(NUMETRON_PLATFORM_CORE2) || defined(NUMETRON_PLATFORM_K8))
#   error "NUMETRON_PLATFORM_ALDERLAKE / CORE2 / K8 are the GMP-derived (LGPL) mul_basecase: define NUMETRON_USE_GMP_LGPL"
#endif

// The C++ basecase, used without NUMETRON_USE_ASM, and with it on CPUs no assembly basecase is
// picked for (limb_arithmetic/umul_basecase_variants.hpp), NUMETRON_CXX_BASECASE, one of:
#define NUMETRON_CXX_BASECASE_UNROLLED 1 // umul_basecase_unrolled: portable C++, the reference
#define NUMETRON_CXX_BASECASE_ADX      2 // umul_basecase_adx: rows in GCC / Clang inline assembly
                                         // with mulx + adcx/adox; needs -march with ADX and BMI2
#define NUMETRON_CXX_BASECASE_BLOCKED  3 // umul_basecase_blocked: rows four limbs at a time with
                                         // x64 intrinsics (mul, adc); what MSVC compiles best
// Default: ADX where the compiler targets it (GCC / Clang, __ADX__ && __BMI2__), BLOCKED with MSVC
// on x64, UNROLLED otherwise. Products of fewer than four limbs always take UNROLLED.
#ifndef NUMETRON_CXX_BASECASE
#   if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__ADX__) && defined(__BMI2__)
#       define NUMETRON_CXX_BASECASE NUMETRON_CXX_BASECASE_ADX
#   elif defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#       define NUMETRON_CXX_BASECASE NUMETRON_CXX_BASECASE_BLOCKED
#   else
#       define NUMETRON_CXX_BASECASE NUMETRON_CXX_BASECASE_UNROLLED
#   endif
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

// ---- FFT (limb_arithmetic/umul_fft.hpp) -------------------------------------------------------
// NUMETRON_FFT_IMPL, the transform kernel, one of:
#define NUMETRON_FFT_IMPL_SCALAR 1 // portable C++, five 62-bit primes (fft/ntt.hpp)
#define NUMETRON_FFT_IMPL_AVX2   2 // AVX2 + FMA in double precision, six 49-bit primes
                                   // (fft/ntt_avx2.hpp); needs the compiler to target AVX2
// Default: AVX2 when the compiler targets it (MSVC /arch:AVX2, GCC/Clang -mavx2 -mfma or a
// -march that has them), SCALAR otherwise. No runtime CPU detection: the choice is the build's.
#if defined(__AVX2__) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__)))
#   define NUMETRON_FFT_AVX2_AVAILABLE
#endif
#ifndef NUMETRON_FFT_IMPL
#   ifdef NUMETRON_FFT_AVX2_AVAILABLE
#       define NUMETRON_FFT_IMPL NUMETRON_FFT_IMPL_AVX2
#   else
#       define NUMETRON_FFT_IMPL NUMETRON_FFT_IMPL_SCALAR
#   endif
#endif

#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2 && !defined(NUMETRON_FFT_AVX2_AVAILABLE)
#   error "NUMETRON_FFT_IMPL_AVX2 needs a compiler targeting AVX2 and FMA"
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

inline constexpr const char* fft_impl_name =
#if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
    "avx2";
#elif NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_SCALAR
    "scalar";
#else
#   error "unknown NUMETRON_FFT_IMPL"
#endif

inline constexpr const char* mul_basecase_name =
#if !defined(NUMETRON_USE_ASM) || !(defined(__x86_64__) || defined(_M_X64))
#   if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
    "c++ adx";
#   elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
    "c++ blocked";
#   else
    "c++";
#   endif
#elif defined(NUMETRON_PLATFORM_ALDERLAKE)
    "asm alderlake (GMP-derived, LGPL)";
#elif defined(NUMETRON_PLATFORM_CORE2)
    "asm core2 (GMP-derived, LGPL)";
#elif defined(NUMETRON_PLATFORM_K8)
    "asm k8 (GMP-derived, LGPL)";
#elif defined(NUMETRON_PLATFORM_ADX)
    "asm adx (numetron, MIT)";
#elif NUMETRON_ASM_LICENSE == NUMETRON_ASM_LICENSE_GMP_LGPL
    "asm, runtime-detected (GMP-derived, LGPL)";
#else
    "asm, runtime-detected (numetron adx, MIT; c++ without BMI2 + ADX)";
#endif

}
