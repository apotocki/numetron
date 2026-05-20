// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <concepts>

namespace numetron::limb_arithmetic {

// prereq: size(r) >= u.size()
// returns carry
template <std::unsigned_integral LimbT, typename InputIteratorT, typename ResultIteratorT>
inline LimbT uadd1(InputIteratorT u, size_t usz, LimbT v, ResultIteratorT r, LimbT c = 0) noexcept
{
    if (!usz) {
        auto [hc, s] = numetron::arithmetic::uadd1(v, c);
        *r = s;
        return hc;
    }
    auto [hc, s] = numetron::arithmetic::uadd1<LimbT>(*u, v, c);
    *r = s;
    while (hc) {
        if (!--usz) break;
        ++u;
        ++r;
        std::tie(hc, *r) = numetron::arithmetic::uadd1<LimbT>(*u, hc);
    }
    return hc;
}

// Unsigned add u[0..n) += v[0..n) inplace, where n is ve - v. Returns carry.
template <std::unsigned_integral LimbT>
inline LimbT uadd_inplace(LimbT* u, LimbT const* v, LimbT const* ve) noexcept
{
    unsigned char c = 0;
    for (; v != ve; ++u, ++v)
        *u = arithmetic::uadd1c(*u, *v, c);
    return (LimbT)c;
}

// Propagate carry/borrow c into a[0..n): a += c. Returns carry.
template <std::unsigned_integral LimbT>
inline LimbT uadd_limb(LimbT* u, LimbT* ue, LimbT c) noexcept
{
    for (; c && u != ue; ++u) {
        std::tie(c, *u) = arithmetic::uadd1(*u, c);
    }
    return c;
}

// u size must be >= v size
template <typename UIteratorT, typename VIteratorT, typename RIteratorT>
inline unsigned char uadd_partial_unchecked(UIteratorT& ub, VIteratorT vb, VIteratorT ve, RIteratorT& rb) noexcept
{
    unsigned char c = 0;
    for (; vb != ve; ++ub, ++vb, ++rb) {
        *rb = numetron::arithmetic::uadd1c(*ub, *vb, c);
    }
    return c;
}

// u size must be >= v size
template <std::unsigned_integral LimbT, typename UIteratorT, typename VIteratorT, typename RIteratorT>
inline unsigned char uadd_unchecked(LimbT uh, UIteratorT ub, UIteratorT ue, LimbT vh, VIteratorT vb, VIteratorT ve, RIteratorT& rb) noexcept
{
    unsigned char c = uadd_partial_unchecked(ub, vb, ve, rb);
    if (ub != ue) {
        *rb = numetron::arithmetic::uadd1c(*ub, vh, c);
        ++rb;
        while (++ub != ue) {
            if (!c) {
                for (;;) {
                    *rb = *ub;
                    ++ub; ++rb;
                    if (ub == ue) break;
                }
                *rb = uh;
                ++rb;
                return 0;
            }
            *rb = numetron::arithmetic::uincx(*ub, c);
            ++rb;
        }
        std::tie(c, *rb) = numetron::arithmetic::uadd1(uh, (LimbT)c);
    } else {
        *rb = numetron::arithmetic::uadd1c(uh, vh, c);
    }
    ++rb;
    return c;
}

template <std::unsigned_integral LimbT, typename RIteratorT>
inline unsigned char uadd_unchecked(LimbT uh, std::span<const LimbT> u, LimbT vh, std::span<const LimbT> v, RIteratorT& rb) noexcept
{
    return uadd_unchecked<LimbT>(uh, u.data(), u.data() + u.size(), vh, v.data(), v.data() + v.size(), rb);
}


}
