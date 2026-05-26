// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <concepts>

namespace numetron::limb_arithmetic {

// Unsigned subtract u[0..n) - v[0..n) inplace, assuming u >= v. where n is ve - v.
template <std::unsigned_integral LimbT>
inline LimbT usub_inplace(LimbT* u, LimbT const* v, LimbT const* ve) noexcept
{
    LimbT borrow = 0;
    for (; v != ve; ++u, ++v)
        std::tie(borrow, *u) = arithmetic::usub1c(*u, *v, borrow);
    return borrow;
}

// Propagate carry/borrow c into a[0..n): a -= c. Returns borrow.
template <std::unsigned_integral LimbT>
inline LimbT usub_limb(LimbT* u, LimbT* ue, LimbT c) noexcept
{
    for (; c && u != ue; ++u) {
        std::tie(c, *u) = arithmetic::usub1(*u, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline LimbT usub_partial_unchecked(LimbT const*& ub, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    LimbT c = 0;
    for (; vb != ve; ++ub, ++vb, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1c(*ub, *vb, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline void usub_unchecked(LimbT const*& ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    assert(ue - ub >= ve - vb);
    LimbT c = usub_partial_unchecked(ub, vb, ve, rb);
    for (; ub != ue; ++ub, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
    }
}

template <std::unsigned_integral LimbT>
inline LimbT usub_partial_limb(LimbT const*& ub, LimbT const* ue, LimbT c, LimbT*& rb) noexcept
{
    for (; c && ub != ue; ++ub, ++rb) {
        std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT>
inline LimbT usub_unchecked(LimbT last_u, LimbT const* ub, LimbT const* ue, LimbT last_v, LimbT const* vb, LimbT const* ve, LimbT*& rb)
{
    LimbT c = usub_partial_unchecked(ub, vb, ve, rb);
    if (ub != ue) {
        std::tie(c, *rb++) = numetron::arithmetic::usub1c(*ub++, last_v, c);
        for (; ub != ue; ++ub, ++rb) {
            std::tie(c, *rb) = numetron::arithmetic::usub1(*ub, c);
        }
        std::tie(c, *rb) = numetron::arithmetic::usub1(last_u, c);
    } else {
        std::tie(c, *rb) = numetron::arithmetic::usub1c(last_u, last_v, c);
    }
    return c;
}

template <std::unsigned_integral LimbT, typename RIteratorT>
inline LimbT usub_unchecked(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, RIteratorT& rb)
{
    return usub_unchecked<LimbT>(uh, u.data(), u.data() + u.size(), vh, v.data(), v.data() + v.size(), rb);
}

}