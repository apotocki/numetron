// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <concepts>

#include "numetron/arithmetic.hpp"

namespace numetron::limb_arithmetic {

// (c, [u]) <- [u] * v + cl; returns c
template <std::unsigned_integral LimbT, typename UIteratorT>
inline LimbT umul1_inplace(UIteratorT ub, UIteratorT ue, LimbT v, LimbT cl = 0) noexcept
{
    while (ub != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        *ub = arithmetic::uadd1ca(l, cl, h);
        cl = h;
        ++ub;
    }
    return cl;
}

// (rh, r[u.size()]) <- [u] * v; returns rh
template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1(UIteratorT ub, UIteratorT ue, LimbT v, RIteratorT&& rraw) noexcept
{
    std::conditional_t<std::is_reference_v<RIteratorT>, RIteratorT, std::decay_t<RIteratorT>> r = std::forward<RIteratorT>(rraw);
    if (ub == ue) {
        assert(ub != ue);
    }
    assert(ub != ue);
    // unroll first iteration
    auto [h, l] = arithmetic::umul1(*ub, v);
    *r = l; ++r;
    
    if (++ub == ue) return h;
    
    // unroll second iteration
    auto [h1, l1] = arithmetic::umul1(*ub, v);
    *r = arithmetic::uadd1ca(l1, h, h1);
    ++r;
    
    // remaining iterations
    while (++ub != ue) {
        auto [h, l] = arithmetic::umul1(*ub, v);
        *r = arithmetic::uadd1ca(l, h1, h);
        ++r;
        h1 = h;
    }
    return h1;
}

template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
inline LimbT umul1(LimbT uh, UIteratorT ulb, UIteratorT ule, LimbT v, RIteratorT& r) noexcept
{
    LimbT pcl = umul1<LimbT>(ulb, ule, v, r);
    auto [h, ph] = numetron::arithmetic::umul1(uh, v);
    auto [c, res] = numetron::arithmetic::uadd1(ph, pcl);
    *r = res; ++r;
    return h + c;
}

// (c, p[u.size()]) <- [u] * v + p[u.size()]; returns [c, p + u.size()]
template <std::unsigned_integral LimbT, typename UIteratorT, typename ResultIteratorT>
inline LimbT umul1_add(UIteratorT ub, UIteratorT ue, LimbT v, ResultIteratorT & r) noexcept
{
    assert(ub != ue);
#if 0
    auto [h, l] = numetron::arithmetic::umul1(*ub, v);
    auto [cr, rl] = numetron::arithmetic::uadd1(l, *r);
    *r = rl; ++r;
    LimbT rh = h + cr;
    while (++ub != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        auto [cr, rl] = numetron::arithmetic::uadd1(l, *r, rh);
        *r = rl; ++r;
        rh = h + cr;
        assert(rh >= h); // if l > 1 => h < 0xff...fe => no cl overflow
    }
    return rh;
#else
    unsigned char cl = 0;
    auto [h0, l0] = numetron::arithmetic::umul1(*ub, v);
    auto [cl2, current] = numetron::arithmetic::uadd1(l0, *r);
    *r = current; ++r;
    while (++ub != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub, v);
        *r = numetron::arithmetic::uadd1c2(l, *r, h0, cl, cl2);
        ++r;
        h0 = h;
    }
    return h0 + cl + cl2;
#endif
}

#if 0

// (uh, [ul]) * v + cl -> (rh, r[ul.size() + 1]); returns rh
//template <std::unsigned_integral LimbT, typename UIteratorT, typename RIteratorT>
//inline LimbT umul1(LimbT uh, UIteratorT ulb, UIteratorT ule, LimbT v, RIteratorT&& rraw) noexcept
//{
//    std::conditional_t<std::is_reference_v<RIteratorT>, RIteratorT, std::decay_t<RIteratorT>> r = std::forward<RIteratorT>(rraw);
//    LimbT pcl = umul1<LimbT>(ulb, ule, v, r);
//    auto [h, ph] = numetron::arithmetic::umul1(uh, v);
//    auto [c, res] = numetron::arithmetic::uadd1(ph, pcl);
//    *r = res; ++r;
//    return h + c;
//}

// (c, ph, p[ul.size()]) <- (uhh, uh, [ul]) * v + cl; returns (c, ph)
template <std::unsigned_integral LimbT>
inline std::tuple<LimbT, LimbT, LimbT> umul1(LimbT uhh, LimbT uh, std::span<const LimbT> ul, LimbT v, LimbT* p, LimbT cl = 0) noexcept
{
    LimbT pcl = umul1<LimbT>(ul, v, p, cl);
    auto [h, ph] = numetron::arithmetic::umul1<LimbT>(uh, v);
    ph += pcl;
    pcl = (ph < pcl) + h;
    auto [hh, phh] = numetron::arithmetic::umul1<LimbT>(uhh, v);
    phh += pcl;
    pcl = (phh < pcl) + hh;
    return { pcl, phh, ph };
}


// (c, p[size(ul) + 2]) <- (uhh, uh, [ul]) * v + p[size(ul)] + cl; returns c
template <std::unsigned_integral LimbT>
inline LimbT umul1_add(LimbT uhh, LimbT uh, std::span<const LimbT> ul, LimbT v, LimbT* p, LimbT cl = 0) noexcept
{
    LimbT pcl = umul1_sum<LimbT>(ul, v, p, cl);
    p += ul.size();
    auto [h, ph] = numetron::arithmetic::umul1<LimbT>(uh, v);
    auto [rc, rh] = numetron::arithmetic::uadd1(ph, *p, pcl);
    *p++ = rh;
    auto [hh, phh] = numetron::arithmetic::umul1<LimbT>(uhh, v);
    auto [rch, rhh] = numetron::arithmetic::uadd1(phh, *p, rc);
    *p++ = rh;
    return rch;
}
#endif

}
