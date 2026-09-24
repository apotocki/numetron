// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <algorithm>

#include "numetron/config/implementation.hpp" // NUMETRON_KARATSUBA_IMPL

// Compile-time defaults for the runtime thresholds below; override by defining these before
// including numetron, or retune at runtime with set_*_threshold() / tune_mul_thresholds()
// (numetron/limb_arithmetic/mul_tuning.hpp).
// The defaults come from tune_mul_thresholds() on x86-64 (Alder Lake class). They depend on the
// configuration: on the Karatsuba implementation (the asm one is faster, which moves every
// crossover above it) and on the compiler, since the Toom evaluation and interpolation kernels
// are compiled C++ and the compilers make rather different code of them.
#if !defined(NUMETRON_USE_ASM)
// Header-only (no NUMETRON_USE_ASM): the C++ basecase (NUMETRON_CXX_BASECASE) and C++ Karatsuba;
// tuned 2026-09-24, the median of six runs each (three with each FFT kernel; the Toom stages are
// tuned with the FFT off, so they don't depend on it). Toom-3/4 jump between plateau values from
// run to run.
#   if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
// GCC / Clang with the ADX rows (~as fast as the asm basecase)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 36
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 212
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 330
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 471
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 761
#   elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
// MSVC with the blocked rows (~1.6x the asm basecase)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 24
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 57
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 194
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 564
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 636
// the reference basecase (~2x the asm basecase)
#   elif defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 14
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 70
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 231
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 418
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 675
#   else // GCC (and Clang, untuned)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 20
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 137
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 218
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 311
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 500
#   endif
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
// asm basecase + asm Karatsuba (the default with NUMETRON_USE_ASM), MSVC with the AVX2 shift
// kernels; tuned 2026-09-24.
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 29
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 115
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 444
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 717 // --tune said 2249 / 2249, but the node curves are
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 967 // those of GCC (docs/multiplication.md, Toom-6.5)
#   else // GCC (and Clang, untuned)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 29
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 115
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 444
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 675
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 967
#   endif
#else
// The C++ Karatsuba implementations over the asm basecase (tuned before the asm Karatsuba).
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 38
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 115
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 330
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 717
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 967
#   else // GCC (and Clang, untuned)
#       define NUMETRON_DEFAULT_KARATSUBA_THRESHOLD 38
#       define NUMETRON_DEFAULT_TOOM3_THRESHOLD 57
#       define NUMETRON_DEFAULT_TOOM4_THRESHOLD 500
#       define NUMETRON_DEFAULT_TOOM6H_THRESHOLD 717
#       define NUMETRON_DEFAULT_TOOM8H_THRESHOLD 967
#   endif
#endif

#ifndef NUMETRON_KARATSUBA_THRESHOLD
#   define NUMETRON_KARATSUBA_THRESHOLD NUMETRON_DEFAULT_KARATSUBA_THRESHOLD
#endif

#ifndef NUMETRON_TOOM3_THRESHOLD
#   define NUMETRON_TOOM3_THRESHOLD NUMETRON_DEFAULT_TOOM3_THRESHOLD
#endif

#ifndef NUMETRON_TOOM4_THRESHOLD
#   define NUMETRON_TOOM4_THRESHOLD NUMETRON_DEFAULT_TOOM4_THRESHOLD
#endif

#ifndef NUMETRON_TOOM6H_THRESHOLD
#   define NUMETRON_TOOM6H_THRESHOLD NUMETRON_DEFAULT_TOOM6H_THRESHOLD
#endif

#ifndef NUMETRON_TOOM8H_THRESHOLD
#   define NUMETRON_TOOM8H_THRESHOLD NUMETRON_DEFAULT_TOOM8H_THRESHOLD
#endif

// The FFT (umul_fft.hpp, 64-bit limbs only): tune_mul_thresholds() on the same machine against
// the Toom defaults above, 2026-09-24. It depends on the transform kernel (NUMETRON_FFT_IMPL) and,
// through the speed of everything below it, on whether the assembly is used.
#if !defined(NUMETRON_USE_ASM) && NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
// header-only, ADX basecase: the median of three runs each (AVX2 1388/1663/1388; scalar
// 10852/10852/6685)
#   if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 1388
#   else
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 10852
#   endif
#elif !defined(NUMETRON_USE_ASM) && NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
// header-only, blocked basecase (MSVC): AVX2 1231/858/1231; scalar 5247 three times
#   if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 1231
#   else
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 5247
#   endif
#elif !defined(NUMETRON_USE_ASM)
// header-only, reference basecase: the median of three runs each (AVX2: MSVC 394/418/444, GCC
// 418/599/636; scalar: MSVC 1663/2389/2389, GCC 2538/2696/2696)
#   if NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
#       if defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_FFT_THRESHOLD 418
#       else
#           define NUMETRON_DEFAULT_FFT_THRESHOLD 599
#       endif
#   else // the scalar kernel
#       if defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_FFT_THRESHOLD 2389
#       else
#           define NUMETRON_DEFAULT_FFT_THRESHOLD 2696
#       endif
#   endif
#elif NUMETRON_FFT_IMPL == NUMETRON_FFT_IMPL_AVX2
// with the asm (two runs each, the same result both times)
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 2696
#   else
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 2538
#   endif
#else // the scalar kernel
#   if defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 11530
#   else
#       define NUMETRON_DEFAULT_FFT_THRESHOLD 13828
#   endif
#endif

#ifndef NUMETRON_FFT_THRESHOLD
#   define NUMETRON_FFT_THRESHOLD NUMETRON_DEFAULT_FFT_THRESHOLD
#endif

// Which Karatsuba / Toom-3 implementation runs: NUMETRON_KARATSUBA_IMPL / NUMETRON_TOOM3_IMPL in
// numetron/config/implementation.hpp.

namespace numetron::limb_arithmetic {

// Smallest operand sizes (in limbs) the algorithms are valid for: Karatsuba needs both halves
// of the split to be non-empty (vn >= 2), Toom-3 needs a non-empty top chunk of v (vn >= 5),
// Toom-4 non-empty top quarters of both operands (un, vn >= 9 or so), Toom-6.5 / Toom-8.5
// non-empty top sixths / eighths; the floors carry some margin. Setters clamp to these, so no threshold value can make
// the dispatch pick an algorithm on operands it can't handle.
inline constexpr size_t min_karatsuba_threshold = 4;
inline constexpr size_t min_toom3_threshold = 12;
inline constexpr size_t min_toom4_threshold = 20;
inline constexpr size_t min_toom6h_threshold = 42;
inline constexpr size_t min_toom8h_threshold = 72;
inline constexpr size_t min_fft_threshold = 1; // the FFT takes any size

namespace detail {

// Atomic so they can be retuned while other threads multiply; relaxed loads compile to plain
// loads on the dispatch path. A change is picked up at the next dispatch decision, so an
// in-flight multiplication may mix old and new values between recursion levels -- which only
// affects its speed, never its result.
inline std::atomic<size_t> karatsuba_threshold_value{ (std::max)(size_t{ NUMETRON_KARATSUBA_THRESHOLD }, min_karatsuba_threshold) };
inline std::atomic<size_t> toom3_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM3_THRESHOLD }, min_toom3_threshold) };
inline std::atomic<size_t> toom4_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM4_THRESHOLD }, min_toom4_threshold) };
inline std::atomic<size_t> toom6h_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM6H_THRESHOLD }, min_toom6h_threshold) };
inline std::atomic<size_t> toom8h_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM8H_THRESHOLD }, min_toom8h_threshold) };
inline std::atomic<size_t> fft_threshold_value{ (std::max)(size_t{ NUMETRON_FFT_THRESHOLD }, min_fft_threshold) };

}

inline size_t karatsuba_threshold() noexcept
{
    return detail::karatsuba_threshold_value.load(std::memory_order_relaxed);
}

inline size_t toom3_threshold() noexcept
{
    return detail::toom3_threshold_value.load(std::memory_order_relaxed);
}

inline void set_karatsuba_threshold(size_t limbs) noexcept
{
    detail::karatsuba_threshold_value.store((std::max)(limbs, min_karatsuba_threshold), std::memory_order_relaxed);
}

inline void set_toom3_threshold(size_t limbs) noexcept
{
    detail::toom3_threshold_value.store((std::max)(limbs, min_toom3_threshold), std::memory_order_relaxed);
}

inline size_t toom4_threshold() noexcept
{
    return detail::toom4_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom4_threshold(size_t limbs) noexcept
{
    detail::toom4_threshold_value.store((std::max)(limbs, min_toom4_threshold), std::memory_order_relaxed);
}

inline size_t toom6h_threshold() noexcept
{
    return detail::toom6h_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom6h_threshold(size_t limbs) noexcept
{
    detail::toom6h_threshold_value.store((std::max)(limbs, min_toom6h_threshold), std::memory_order_relaxed);
}

inline size_t toom8h_threshold() noexcept
{
    return detail::toom8h_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom8h_threshold(size_t limbs) noexcept
{
    detail::toom8h_threshold_value.store((std::max)(limbs, min_toom8h_threshold), std::memory_order_relaxed);
}

inline size_t fft_threshold() noexcept
{
    return detail::fft_threshold_value.load(std::memory_order_relaxed);
}

inline void set_fft_threshold(size_t limbs) noexcept
{
    detail::fft_threshold_value.store((std::max)(limbs, min_fft_threshold), std::memory_order_relaxed);
}

}
