// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <algorithm>

// Compile-time defaults for the runtime thresholds below; override by defining these before
// including numetron, or retune at runtime with set_*_threshold() / tune_mul_thresholds()
// (numetron/limb_arithmetic/mul_tuning.hpp).
#ifndef NUMETRON_KARATSUBA_THRESHOLD
#   define NUMETRON_KARATSUBA_THRESHOLD 40
#endif

#ifndef NUMETRON_TOOM3_THRESHOLD
#   define NUMETRON_TOOM3_THRESHOLD 260
#endif

#ifndef NUMETRON_TOOM4_THRESHOLD
#   define NUMETRON_TOOM4_THRESHOLD 300
#endif

#define NUMETRON_EXPLICIT_KARATSUBA

// Balanced Toom-3 runs the hand-written umul_toom3_impl() unless NUMETRON_TOOM3_USE_ENGINE is
// defined, in which case the same algorithm runs as the engine's toom3_balanced plan.
#ifndef NUMETRON_TOOM3_USE_ENGINE
#   define NUMETRON_EXPLICIT_TOOM3
#endif

namespace numetron::limb_arithmetic {

// Smallest operand sizes (in limbs) the algorithms are valid for: Karatsuba needs both halves
// of the split to be non-empty (vn >= 2), Toom-3 needs a non-empty top chunk of v (vn >= 5),
// Toom-4 non-empty top quarters of both operands (un, vn >= 9 or so); the floors carry some
// margin. Setters clamp to these, so no threshold value can make the dispatch pick an algorithm
// on operands it can't handle.
inline constexpr size_t min_karatsuba_threshold = 4;
inline constexpr size_t min_toom3_threshold = 12;
inline constexpr size_t min_toom4_threshold = 20;

namespace detail {

// Atomic so they can be retuned while other threads multiply; relaxed loads compile to plain
// loads on the dispatch path. A change is picked up at the next dispatch decision, so an
// in-flight multiplication may mix old and new values between recursion levels -- which only
// affects its speed, never its result.
inline std::atomic<size_t> karatsuba_threshold_value{ (std::max)(size_t{ NUMETRON_KARATSUBA_THRESHOLD }, min_karatsuba_threshold) };
inline std::atomic<size_t> toom3_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM3_THRESHOLD }, min_toom3_threshold) };
inline std::atomic<size_t> toom4_threshold_value{ (std::max)(size_t{ NUMETRON_TOOM4_THRESHOLD }, min_toom4_threshold) };

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

}
