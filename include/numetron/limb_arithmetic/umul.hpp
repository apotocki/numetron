// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "umul_basecase.hpp"
#include "toom/engine.hpp"
#include "toom/thresholds.hpp"

#ifdef NUMETRON_EXPLICIT_KARATSUBA
#   include "umul_karatsuba.hpp"
#endif

namespace numetron::limb_arithmetic {

inline bool is_karatsuba_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    // Condition is expressed in terms of vn so that Toom-k dispatch is uniform:
    // each algorithm checks vn >= its threshold and un < k*vn (u fits in k pieces).
    // For Karatsuba (k=2): vn >= threshold and un < 2*vn.
    return vn >= NUMETRON_KARATSUBA_THRESHOLD && 2 * vn > un;
}

inline bool is_toom3_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= NUMETRON_TOOM3_THRESHOLD && 3 * vn > un;
}

template <std::unsigned_integral LimbT, typename AllocatorT>
inline LimbT* umul_dispatch(
    const LimbT* u, size_t un,
    const LimbT* v, size_t vn,
    LimbT* rb,
    AllocatorT alloc)
{
    while (un > 0 && u[un - 1] == 0) --un;
    while (vn > 0 && v[vn - 1] == 0) --vn;
    if (un < vn) {
        std::swap(u, v);
        std::swap(un, vn);
    }

    if (is_toom3_applicable(un, vn)) {
        return toom_engine<3, 3>::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_karatsuba_applicable(un, vn)) {
#ifndef NUMETRON_EXPLICIT_KARATSUBA
        return toom_engine<2, 2>::umul(u, un, v, vn, rb, std::move(alloc));
#else
        return detail::umul_karatsuba_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
#endif
    }
    if (vn) {
        return umul_basecase<LimbT>(u, un, v, vn, rb);
    }
    return rb;
}

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
{
    //if (is_toom3_applicable(u.size(), v.size())) {
    //    return toom_engine<3, 3>::umul(u, v, std::move(alloc));
    //}

    if (is_karatsuba_applicable(u.size(), v.size())) {
#ifndef NUMETRON_EXPLICIT_KARATSUBA
        return toom_engine<2, 2>::umul(u, v, std::move(alloc));
#else
        return umul_karatsuba(u, v, std::move(alloc));
#endif
    }

    if (!v.empty()) {
        size_t rsz = u.size() + v.size();
        LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
        LimbT* re = umul_basecase(u.data(), u.size(), v.data(), v.size(), r);
        return { r, static_cast<size_t>(re - r), rsz };
    }
    return { nullptr, 0, 0 };
}

// base case mul with explicit high limbs uh and vh
// precondition: |u| >= |v|, u.size() > 0
// returns iterator to one past the last written limb (r + u.size() + v.size() + 2)
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul(LimbT uh, std::span<const LimbT> ul, LimbT vh, std::span<const LimbT> vl, AllocatorT alloc)
{
    assert(!ul.empty());

    const size_t n = ul.size();
    const size_t m = vl.size();

    if (!m) {
        size_t rsz = ul.size() + 2;
        LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
        LimbT* re = r;
        LimbT rh = umul1(uh, ul.data(), ul.data() + n, vh, re);
        if (rh) { *re = rh; ++re; }
        return { r, static_cast<size_t>(re - r), rsz };
    }

    size_t rsz = n + m;
    LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
    
    // Low product
    LimbT* re = umul_basecase(ul.data(), n, vl.data(), m, r);

    if (!uh && !vh) return { r, static_cast<size_t>(re - r), rsz };

    // Ensure top two limbs exist and start from zero
    LimbT* top = r; top += (n + m);
        
    // Clear intermediate limbs if any
    while (re != top) {
        *re++ = LimbT{0};
    }

    if (uh) {
        *re = LimbT{0};
        ++re;
        // Add uh * v at offset n
        auto rvhu = r + n;
        LimbT c = umul1_add(vl.data(), vl.data() + m, uh, rvhu);
        c = uadd1<LimbT>(rvhu, re - rvhu, c, rvhu);
        assert(c == 0);
    }
    if (vh) {
        *re = LimbT{0};
        ++re;
        // Add vh * u at offset m
        auto rvhv = r + m;
        LimbT c = umul1_add(ul.data(), ul.data() + n, vh, rvhv);
        c = uadd1<LimbT>(rvhv, re - rvhv, c, rvhv);
        assert(c == 0);
    
        // Add uh * vh at offset n + m
        if (uh) {
            auto [h, l] = numetron::arithmetic::umul1<LimbT>(uh, vh);
            r += n + m;
            auto [c, s] = numetron::arithmetic::uadd1<LimbT>(*r, l);
            *r = s; ++r;
            auto [c2, s2] = numetron::arithmetic::uadd1<LimbT>(*r, h, c);
            *r = s2;
            assert(c2 == 0);
        }
    }

    return { r, static_cast<size_t>(re - r), rsz };;
}

#if 0
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT uh, std::span<const LimbT> ul, LimbT vh, std::span<const LimbT> vl, ResultIteratorT r) noexcept
{
    assert(!ul.empty());

    const size_t n = ul.size();
    const size_t m = vl.size();

    if (!m) {
        LimbT rh = umul1(uh, ul.data(), ul.data() + ul.size(), vh, r);
        if (rh) { *r = rh; ++r; }

        //if (ul.empty()) {
        //    auto [h, ph] = numetron::arithmetic::umul1(uh, vh);
        //    *r = ph; ++r;
        //    *r = h; ++r;
        //} else {
        //    LimbT rh = umul1(uh, ul.data(), ul.data() + ul.size(), vh, r);
        //    if (rh) { *r = rh; ++r; }
        //}
        return r;
    } 

    //else if (!ul.size()) {
    //    auto [pcl, ph] = umul1(vh, vl.data(), vl.data() + vl.size(), uh, r);

    //    if (ph) *r++ = ph;
    //    if (pcl) {
    //        assert(!pcl); // because cl = 0 in umul1 call
    //    }
    //    assert(!pcl); // because cl = 0 in umul1 call
    //    return r;
    //}

    // Low product
    ResultIteratorT re = umul<LimbT>(ul.data(), ul.data() + n, vl.data(), vl.data() + m, r); // may return r if one is empty

    if (!uh && !vh) return re;

    // Ensure top two limbs exist and start from zero
    ResultIteratorT top = r; top += (n + m);
        
    // Clear intermediate limbs if any
    while (re != top) {
        *re++ = LimbT{0};
    }

    if (uh) {
        *re = LimbT{0};
        ++re;
        // Add uh * v at offset n
        auto rvhu = r + n;
        LimbT c = umul1_add(vl.data(), vl.data() + m, uh, rvhu);
        c = uadd1<LimbT>(rvhu, re - rvhu, c, rvhu);
        assert(c == 0);
    }
    if (vh) {
        *re = LimbT{0};
        ++re;
        // Add vh * u at offset m
        auto rvhv = r + m;
        LimbT c = umul1_add(ul.data(), ul.data() + n, vh, rvhv);
        c = uadd1<LimbT>(rvhv, re - rvhv, c, rvhv);
        assert(c == 0);
    
        // Add uh * vh at offset n + m
        if (uh) {
            auto [h, l] = numetron::arithmetic::umul1<LimbT>(uh, vh);
            r += n + m;
            auto [c, s] = numetron::arithmetic::uadd1<LimbT>(*r, l);
            *r = s; ++r;
            auto [c2, s2] = numetron::arithmetic::uadd1<LimbT>(*r, h, c);
            *r = s2;
            assert(c2 == 0);
        }
    }

    return re;
}

// base case mul: {u} * {v} -> {rb, re}
// returns re
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, ResultIteratorT r) noexcept
{
    assert(ub != ue && vb != ve);

    ResultIteratorT rb = r;

    // first line
    unsigned char cl = 0;
    // unrolled first iteration
    auto [previous, current] = numetron::arithmetic::umul1(*ub, *vb);
    *rb = current; ++rb;
    // unrolled second iteration
    auto ub_it = ub;
    if (++ub_it != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
        auto [nc, current] = numetron::arithmetic::uadd1(l, previous);
        *rb = current; ++rb;
        cl = nc;
        previous = h;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
            *rb = numetron::arithmetic::uadd1c(l, previous, cl);
            ++rb;
            previous = h;
        }
    }
    *rb = previous + cl;

    // next lines
    size_t dec_usz = ue - ub - 1;
    for (++vb; vb != ve; ++vb) {
        cl = 0;
        ub_it = ub;
        rb -= dec_usz;
        LimbT vb_val = *vb;
        // first iteration unrolled
        auto [previous, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
        auto [cl2, current] = numetron::arithmetic::uadd1(l, *rb);
        *rb = current; ++rb;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
            *rb = numetron::arithmetic::uadd1c2(l, *rb, previous, cl, cl2);
            ++rb;
            previous = h;
        } 
        *rb = previous + cl + cl2;
    }
    return rb + 1;
}
#endif

}
