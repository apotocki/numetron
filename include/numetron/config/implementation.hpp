// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

// The one place that sets which implementation of each algorithm the library uses. Every choice
// below is only a default: define the macro yourself (compiler flag, or #define before including
// any numetron header) to override it. No other header defines these.
// Tuning numbers are separate: the multiplication thresholds are in
// limb_arithmetic/toom/thresholds.hpp, the division ones in limb_arithmetic.hpp.

// ---- Front end and back end ---------------------------------------------------------------------
// An application sees one switch, NUMETRON_COMPILED. Off (the default), numetron is header-only
// and pure C++: every translation unit that uses a heavy algorithm compiles it. Defined, the
// headers are a front end to the numetron library (the back end), which must then be linked: the
// heavy algorithms are compiled there once (src/*.cpp) and the headers only declare their entry
// points. The CMake target `numetron` builds it and exports the define (option NUMETRON_COMPILED,
// default ON); the MSVC test and bench projects set it themselves. What the library compiles:
// - the multiplication chain above the basecase (Karatsuba, the Toom engine and plans, the FFT:
//   limb_arithmetic/umul_dispatch.hpp), src/umul_large.cpp: the entry point is
//   detail::umul_large in limb_arithmetic/umul.hpp, for uint64_t limbs (the only ones the chain
//   exists for).
// - the division (the basecase and Svoboda's: limb_arithmetic/udiv.hpp), src/udiv_large.cpp:
//   udiv() calls detail::udiv_large for uint64_t limbs, a pointer quotient iterator and
//   std::allocator (all the library's own calls); other limb types stay inline.
// - the tunable thresholds and, with the assembly, the choice of the basecase kernels for the CPU
//   (src/backend.cpp).
//
// NUMETRON_BACKEND_ASM: the library contains the x86-64 assembly, so the front end's inline code
// calls its kernels (mul/sqr_basecase, add/sub_n: limb_arithmetic/backend.hpp) instead of its own
// C++ ones. A fact about the library, not a choice: whatever builds the library says it to the
// code that uses it -- the CMake target `numetron` exports it along with NUMETRON_COMPILED when it
// has the assembly; the MSVC test and bench projects set it for x64. Without NUMETRON_COMPILED it
// means nothing. The application's units, the library's (which derive it from NUMETRON_USE_ASM)
// and NUMETRON_BACKEND_INTERNAL ones share that inline code, so all must see the same value; a
// mismatch costs speed, not correctness (the library exports the same symbols either way).
//
// Everything below configures the back end: it takes effect where the library
// is compiled (build it with the intended -march / /arch: the FFT kernel and the C++ basecase
// follow the compiler's target) and is not the application's to set. The library's own units
// are compiled with NUMETRON_BUILDING_LIBRARY. Code that runs the chain directly rather than
// through the front end (the threshold tuner, limb_arithmetic/mul_tuning.hpp; tests of the
// plans) is compiled with NUMETRON_BACKEND_INTERNAL and the library's configuration (the CMake
// target `numetron_internal` carries both).
#if defined(NUMETRON_BUILDING_LIBRARY) || defined(NUMETRON_BACKEND_INTERNAL)
#   if !defined(NUMETRON_COMPILED)
#       error "NUMETRON_BUILDING_LIBRARY / NUMETRON_BACKEND_INTERNAL: the back end is the compiled library, define NUMETRON_COMPILED"
#   endif
#   define NUMETRON_BACKEND_TU
#endif

// ---- Assembly (back end) ----------------------------------------------------------------------
// NUMETRON_USE_ASM, internal: the back end uses the src/arch assembly (x86-64: mul_basecase,
// sqr_basecase, add/sub_n, the Karatsuba kernels). On by default in the library's units on
// x86-64; build the library with NUMETRON_NO_ASM (CMake: -DNUMETRON_ASM=OFF) to do without it.
// The header-only build and the front end never use it (the front end: NUMETRON_BACKEND_ASM).
#if defined(NUMETRON_USE_ASM)
#   error "NUMETRON_USE_ASM is internal: the numetron library decides it (build the library with NUMETRON_NO_ASM / -DNUMETRON_ASM=OFF to do without the assembly)"
#endif
#if defined(NUMETRON_BACKEND_TU) && !defined(NUMETRON_NO_ASM) && (defined(__x86_64__) || defined(_M_X64))
#   define NUMETRON_USE_ASM
#endif
#if defined(NUMETRON_BACKEND_TU)
#   if defined(NUMETRON_BACKEND_ASM) && !defined(NUMETRON_USE_ASM)
#       error "NUMETRON_BACKEND_ASM, but the library is built without the assembly (NUMETRON_NO_ASM, or not x86-64)"
#   elif defined(NUMETRON_USE_ASM) && !defined(NUMETRON_BACKEND_ASM)
#       define NUMETRON_BACKEND_ASM
#   endif
#endif

// The front end calls the library's assembly kernels (see NUMETRON_BACKEND_ASM above).
#if defined(NUMETRON_COMPILED) && defined(NUMETRON_BACKEND_ASM) && (defined(__x86_64__) || defined(_M_X64))
#   define NUMETRON_ASM_KERNELS
#endif

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

// The largest n the asm squaring basecase (src/arch/x86_64/sqr_basecase_adx.*) has straight-line
// code for, 2..32 (default 32). It is set where the assembly is built -- the CMake option of the
// same name, or the MASM definition -- and must be the same in the library's C++, as it picks the
// default squaring thresholds (limb_arithmetic/toom/thresholds.hpp); the CMake target passes it to
// both (and to `numetron_internal`). A smaller one
// trades speed at 17..32 limbs for code size (n = 17..32 are ~174 KB of code).
#ifndef NUMETRON_SQR_STRAIGHT_MAX
#   define NUMETRON_SQR_STRAIGHT_MAX 32
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

// (Checked where it takes effect: the front end's units don't run Karatsuba.)
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM && (defined(NUMETRON_BACKEND_TU) || !defined(NUMETRON_COMPILED)) \
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

// The choices above as text, for benchmarks and diagnostics: where they are the ones in effect
// (the header-only build, the library's units, NUMETRON_BACKEND_INTERNAL), not in the front end.
#if !defined(NUMETRON_COMPILED) || defined(NUMETRON_BACKEND_TU)
namespace numetron::config {

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
#endif
