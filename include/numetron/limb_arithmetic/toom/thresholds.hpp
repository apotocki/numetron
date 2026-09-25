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

// Toom-3/2 (unbalanced, un ~ 1.5 vn; the threshold is on vn): tune_mul_thresholds() on
// 1.5n x n (--tune-toom32, against the balanced defaults), 2026-09-25. Below the Karatsuba
// threshold it competes with the basecase, and wins there in the header-only builds.
#if !defined(NUMETRON_DEFAULT_TOOM32_THRESHOLD)
#   if !defined(NUMETRON_USE_ASM)
#       if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
#           define NUMETRON_DEFAULT_TOOM32_THRESHOLD 34       // GCC 36 / 34 (two runs)
#       elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
#           define NUMETRON_DEFAULT_TOOM32_THRESHOLD 28       // MSVC
#       elif defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM32_THRESHOLD 23       // reference basecase, MSVC
#       else
#           define NUMETRON_DEFAULT_TOOM32_THRESHOLD 26       // reference basecase, GCC 30 / 26
#       endif
#   elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
#       define NUMETRON_DEFAULT_TOOM32_THRESHOLD 44           // GCC 44 / 44, MSVC 44 / 42
#   else
#       define NUMETRON_DEFAULT_TOOM32_THRESHOLD 32           // C++ Karatsuba over the asm basecase (tuned with _CXX): GCC 32, MSVC 32
#   endif
#endif

#ifndef NUMETRON_DEFAULT_TOOM32_THRESHOLD
#   define NUMETRON_DEFAULT_TOOM32_THRESHOLD NUMETRON_DEFAULT_TOOM3_THRESHOLD
#endif

// Toom-4/2 (unbalanced, 1.75 <= un/vn < 2; the threshold is on vn): tune_mul_thresholds() on
// 1.875n x n (--tune-unbalanced, against the balanced defaults), 2026-09-25.
#if !defined(NUMETRON_DEFAULT_TOOM42_THRESHOLD)
#   if !defined(NUMETRON_USE_ASM)
#       if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
#           define NUMETRON_DEFAULT_TOOM42_THRESHOLD 63       // GCC
#       elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
#           define NUMETRON_DEFAULT_TOOM42_THRESHOLD 38       // MSVC
#       elif defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM42_THRESHOLD 42       // reference basecase, MSVC
#       else
#           define NUMETRON_DEFAULT_TOOM42_THRESHOLD 66       // reference basecase, GCC
#       endif
#   elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
#       define NUMETRON_DEFAULT_TOOM42_THRESHOLD 70           // GCC 70, MSVC 70
// C++ Karatsuba over the asm basecase (tuned with _CXX)
#   elif defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_TOOM42_THRESHOLD 38           // (60: +0.7%)
#   else
#       define NUMETRON_DEFAULT_TOOM42_THRESHOLD 78           // a plateau: 36..78 within 0.3%
#   endif
#endif

// Slicing u into vn-limb pieces (un >= 2 vn) starts at min(Karatsuba threshold, this), so also
// with basecase pieces when Karatsuba starts later: GCC, asm basecase + C++ Karatsuba (Karatsuba
// from 38), numetron_bench_mul --unbalanced, 2026-09-25: from 24 the vn = 32 row goes from
// 0.90-0.97 of GMP to 1.09-1.13 (un/vn >= 3); from 16 the vn = 16 row drops (1.31 -> 1.05 at
// un/vn = 2): pieces that short cost more than the long rows.
#ifndef NUMETRON_DEFAULT_SLICING_THRESHOLD
#   define NUMETRON_DEFAULT_SLICING_THRESHOLD 24
#endif

// Toom-6.5 7 x 6 (unbalanced, 1 < un/vn < 1.4; the threshold is on vn): tune_mul_thresholds() on
// 1.3n x n (--tune-unbalanced), 2026-09-25. With the asm Karatsuba: GCC 371 (the one-level scan
// crosses 1 at ~350, 0.88-0.93 from 1000 up), MSVC 636 (crosses at ~500-636; 500 within 0.1%).
// The others: two runs each, the same value unless noted.
#if !defined(NUMETRON_DEFAULT_TOOM76_THRESHOLD)
#   if !defined(NUMETRON_USE_ASM)
#       if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 253      // GCC
#       elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 394      // MSVC
#       elif defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 245      // reference basecase, MSVC
#       else
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 154      // reference basecase, GCC 154 / 154 / 145 / 145
#       endif
#   elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
#       if defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 636
#       else
#           define NUMETRON_DEFAULT_TOOM76_THRESHOLD 371
#       endif
// C++ Karatsuba over the asm basecase (tuned with _CXX)
#   elif defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_TOOM76_THRESHOLD 531          // 500 / 531 (0.99 at 500)
#   else
#       define NUMETRON_DEFAULT_TOOM76_THRESHOLD 371
#   endif
#endif

// Toom-6/3 (unbalanced, 1.75 <= un/vn < 2 over toom42; the threshold is on vn):
// tune_mul_thresholds() on 1.875n x n (--tune-unbalanced, against toom42), 2026-09-25. With the
// asm Karatsuba: GCC 218 (crosses 1 at ~210, 0.90-0.95 from ~400 up), MSVC 394 (crosses at ~390,
// 0.91-0.97 above). The others: two runs each, the same value unless noted.
#if !defined(NUMETRON_DEFAULT_TOOM63_THRESHOLD)
#   if !defined(NUMETRON_USE_ASM)
#       if NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_ADX
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 231      // GCC (~1.0 from 115 up)
#       elif NUMETRON_CXX_BASECASE == NUMETRON_CXX_BASECASE_BLOCKED
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 260      // MSVC
#       elif defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 137      // reference basecase, MSVC 137 / 97; a jagged
                                                              // curve (1.2 at 371 and 500 in both runs)
#       else
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 137      // reference basecase, GCC 129 / 137
#       endif
#   elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
#       if defined(_MSC_VER) && !defined(__clang__)
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 394
#       else
#           define NUMETRON_DEFAULT_TOOM63_THRESHOLD 218
#       endif
// C++ Karatsuba over the asm basecase (tuned with _CXX)
#   elif defined(_MSC_VER) && !defined(__clang__)
#       define NUMETRON_DEFAULT_TOOM63_THRESHOLD 371
#   else
#       define NUMETRON_DEFAULT_TOOM63_THRESHOLD 350
#   endif
#endif

#ifndef NUMETRON_KARATSUBA_THRESHOLD
#   define NUMETRON_KARATSUBA_THRESHOLD NUMETRON_DEFAULT_KARATSUBA_THRESHOLD
#endif
#ifndef NUMETRON_TOOM63_THRESHOLD
#   define NUMETRON_TOOM63_THRESHOLD NUMETRON_DEFAULT_TOOM63_THRESHOLD
#endif
#ifndef NUMETRON_TOOM76_THRESHOLD
#   define NUMETRON_TOOM76_THRESHOLD NUMETRON_DEFAULT_TOOM76_THRESHOLD
#endif
#ifndef NUMETRON_SLICING_THRESHOLD
#   define NUMETRON_SLICING_THRESHOLD NUMETRON_DEFAULT_SLICING_THRESHOLD
#endif
#ifndef NUMETRON_TOOM32_THRESHOLD
#   define NUMETRON_TOOM32_THRESHOLD NUMETRON_DEFAULT_TOOM32_THRESHOLD
#endif
#ifndef NUMETRON_TOOM42_THRESHOLD
#   define NUMETRON_TOOM42_THRESHOLD NUMETRON_DEFAULT_TOOM42_THRESHOLD
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
inline constexpr size_t min_toom32_threshold = 8;
inline constexpr size_t min_toom42_threshold = 8;
inline constexpr size_t min_slicing_threshold = 1;
inline constexpr size_t min_toom76_threshold = 12;
inline constexpr size_t min_toom63_threshold = 12;
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
inline std::atomic<size_t> toom32_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM32_THRESHOLD }, min_toom32_threshold) };
inline std::atomic<size_t> toom42_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM42_THRESHOLD }, min_toom42_threshold) };
inline std::atomic<size_t> slicing_threshold_value{ (std::max)(size_t{ NUMETRON_SLICING_THRESHOLD }, min_slicing_threshold) };
inline std::atomic<size_t> toom76_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM76_THRESHOLD }, min_toom76_threshold) };
inline std::atomic<size_t> toom63_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM63_THRESHOLD }, min_toom63_threshold) };
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

inline size_t toom32_threshold() noexcept
{
    return detail::toom32_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom32_threshold(size_t limbs) noexcept
{
    detail::toom32_threshold_value.store((std::max)(limbs, min_toom32_threshold), std::memory_order_relaxed);
}

inline size_t toom63_threshold() noexcept
{
    return detail::toom63_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom63_threshold(size_t limbs) noexcept
{
    detail::toom63_threshold_value.store((std::max)(limbs, min_toom63_threshold), std::memory_order_relaxed);
}

inline size_t toom76_threshold() noexcept
{
    return detail::toom76_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom76_threshold(size_t limbs) noexcept
{
    detail::toom76_threshold_value.store((std::max)(limbs, min_toom76_threshold), std::memory_order_relaxed);
}

inline size_t slicing_threshold() noexcept
{
    return detail::slicing_threshold_value.load(std::memory_order_relaxed);
}

inline void set_slicing_threshold(size_t limbs) noexcept
{
    detail::slicing_threshold_value.store((std::max)(limbs, min_slicing_threshold), std::memory_order_relaxed);
}

inline size_t toom42_threshold() noexcept
{
    return detail::toom42_threshold_value.load(std::memory_order_relaxed);
}

inline void set_toom42_threshold(size_t limbs) noexcept
{
    detail::toom42_threshold_value.store((std::max)(limbs, min_toom42_threshold), std::memory_order_relaxed);
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
