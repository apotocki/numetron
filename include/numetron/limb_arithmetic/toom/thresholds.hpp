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
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
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
// The C++ Karatsuba implementations over the asm basecase (tuned before the asm Karatsuba). The
// pure C++ build (no NUMETRON_USE_ASM) uses these too; it is not tuned.
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

}
