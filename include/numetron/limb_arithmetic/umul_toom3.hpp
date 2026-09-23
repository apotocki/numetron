// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <memory>
#include <cstring>
#include <algorithm>
#include <concepts>
#include <limits>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/assert.hpp"

#include "toom/kernels.hpp"

namespace numetron::limb_arithmetic {

template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_dispatch(
    const LimbT* u, size_t un,
    const LimbT* v, size_t vn,
    LimbT* rb,
    AllocatorT alloc);

namespace detail::toom3 {

using namespace toom_kernels;

// Evaluates x = x0 + x1*W + x2*W^2 (x0, x1: n limbs, x2: h limbs, 1 <= h <= n) at 1, -1 and 2,
// each result n+1 limbs:
//   p1 = x0 + x1 + x2              (top limb <= 2)
//   pm1 = |x0 - x1 + x2|           (top limb <= 1), returns true when x0 - x1 + x2 < 0
//   p2 = x0 + 2*x1 + 4*x2 = 2*(p1 + x2) - x0   (top limb <= 6)
// g = x0 + x2 is formed once and shared by both +-1 points; it is staged in p2, which is only
// written for real after g is no longer needed.
template <std::unsigned_integral LimbT>
inline bool eval3(LimbT const* x0, LimbT const* x1, LimbT const* x2, size_t n, size_t h,
    LimbT* p1, LimbT* pm1, LimbT* p2) noexcept
{
    LimbT* g = p2;
    LimbT gc = add_n(g, x0, x2, h);
    if (h < n) gc = add_1(g + h, x0 + h, n - h, gc);

    p1[n] = gc + add_n(p1, g, x1, n);

    bool neg = false;
    if (!gc && cmp_n(g, x1, n) < 0) {
        sub_n(pm1, x1, g, n);
        pm1[n] = 0;
        neg = true;
    } else {
        pm1[n] = gc - sub_n(pm1, g, x1, n);
    }

    LimbT c = add_n(p2, p1, x2, h);
    if (h < n) c = add_1(p2 + h, p1 + h, n - h, c);
    p2[n] = p1[n] + c;
    [[maybe_unused]] LimbT out = lshift1(p2, p2, n + 1);
    NUMETRON_ASSERT(!out);
    p2[n] -= sub_n(p2, p2, x0, n);
    return neg;
}

// dst[0..un+vn) = a[0..an) * b[0..bn), zero-padding whatever umul_dispatch leaves above the
// product's significant limbs (it strips leading zero limbs of its operands).
template <std::unsigned_integral LimbT, typename AllocatorT>
inline void mul_into(LimbT* dst, LimbT const* a, size_t an, LimbT const* b, size_t bn, AllocatorT alloc)
{
    LimbT* e = umul_dispatch(a, an, b, bn, dst, alloc);
    std::memset(e, 0, static_cast<size_t>(dst + an + bn - e) * sizeof(LimbT));
}

// rb[off..) += x[0..len), clipped to the result end re; the clipped-off limbs of x are zero
// because the whole product fits in the result.
template <std::unsigned_integral LimbT>
inline void add_at(LimbT* rb, LimbT* re, size_t off, LimbT const* x, size_t len) noexcept
{
    LimbT* p = rb + off;
    const size_t k = (std::min)(len, static_cast<size_t>(re - p));
#ifndef NDEBUG // NUMETRON_ASSERT still evaluates its argument in release builds
    NUMETRON_ASSERT(std::all_of(x + k, x + len, [](LimbT l) { return l == 0; }));
#endif
    if (LimbT c = add_n(p, p, x, k)) {
        uadd_limb(p + k, re, c);
    }
}

}

namespace detail {

// Whether umul_toom3_impl() can take un x vn (un >= vn): it splits u into thirds of
// n = ceil(un/3) limbs and needs v to reach into its third part (vn > 2n). More unbalanced
// products that are still Toom-3 territory stay with the generic toom_engine<3, 3>.
inline bool toom3_split_fits(size_t un, size_t vn) noexcept
{
    return vn > 2 * ((un + 2) / 3);
}

// Toom-3 multiplication core.
//
// Preconditions: un >= vn, toom3_split_fits(un, vn) (which is what guarantees 0 < t <= s <= n
// below), rb[0 .. un+vn) is the (uninitialized) output.
//
// Split:  u = a0 + a1*W + a2*W^2,  v = b0 + b1*W + b2*W^2,  W = B^n,
//   n = ceil(un/3), s = un - 2n = |a2|, t = vn - 2n = |b2|,  0 < t <= s <= n.
// The product r(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 is evaluated at 0, 1, -1, 2, inf:
//   c0 = a0*b0 (-> rb[0..2n)), c4 = a2*b2 (-> rb[4n..)), r(1), |r(-1)|, r(2) (-> scratch),
// then interpolated with (all intermediate values provably non-negative):
//   w2 = (r(2) - r(-1)) / 3   = c1 + c2 + 3c3 + 5c4
//   wm = (r(1) - r(-1)) / 2   = c1 + c3
//   w1 = r(1) - c0            = c1 + c2 + c3 + c4
//   w2 = (w2 - w1) / 2        = c3 + 2c4
//   w1 = w1 - wm              = c2 + c4
//   w2 = w2 - 2c4             = c3
//   w1 = w1 - c4              = c2
//   wm = wm - w2              = c1
// and c1, c2, c3 are added into rb at n, 2n, 3n.
//
// Allocates 12n + 12 limbs of scratch via alloc (LIFO, released before returning).
// Returns rb + un + vn.
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_toom3_impl(std::span<const LimbT> u, std::span<const LimbT> v, LimbT* rb, AllocatorT alloc)
{
    using namespace toom3;

    const size_t un = u.size();
    const size_t vn = v.size();
    NUMETRON_ASSERT(un >= vn && toom3_split_fits(un, vn));

    const size_t n = (un + 2) / 3;
    const size_t s = un - 2 * n;
    const size_t t = vn - 2 * n;
    const size_t st = s + t;
    NUMETRON_ASSERT(0 < t && t <= s && s <= n);

    const size_t e = n + 1;          // limbs of an evaluated operand
    const size_t pn = 2 * e;         // buffer for the product of two of them
    const size_t L = 2 * n + 1;      // limbs the point values r(1), |r(-1)|, r(2) actually need

    LimbT const* a0 = u.data();
    LimbT const* a1 = a0 + n;
    LimbT const* a2 = a0 + 2 * n;
    LimbT const* b0 = v.data();
    LimbT const* b1 = b0 + n;
    LimbT const* b2 = b0 + 2 * n;
    LimbT* const re = rb + un + vn;
    LimbT* const c4 = rb + 4 * n;

    const size_t scratch_sz = 6 * e + 3 * pn;
    LimbT* scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
    NUMETRON_SCOPE_EXIT([&] {
        std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });

    LimbT* as1 = scratch;
    LimbT* asm1 = as1 + e;
    LimbT* as2 = asm1 + e;
    LimbT* bs1 = as2 + e;
    LimbT* bsm1 = bs1 + e;
    LimbT* bs2 = bsm1 + e;
    LimbT* w1 = bs2 + e;
    LimbT* wm = w1 + pn;
    LimbT* w2 = wm + pn;

    const bool vm1_neg = eval3(a0, a1, a2, n, s, as1, asm1, as2) != eval3(b0, b1, b2, n, t, bs1, bsm1, bs2);

    mul_into(w1, as1, e, bs1, e, alloc);
    mul_into(wm, asm1, e, bsm1, e, alloc);
    mul_into(w2, as2, e, bs2, e, alloc);
    mul_into(rb, a0, n, b0, n, alloc);   // c0 -> rb[0..2n)
    mul_into(c4, a2, s, b2, t, alloc);   // c4 -> rb[4n..re)
    // rb[2n..4n) is still uninitialized; c2 fills it below.

    if (vm1_neg) add_n(w2, w2, wm, L); else sub_n(w2, w2, wm, L);
    divexact_by3(w2, w2, L);

    if (vm1_neg) add_n(wm, w1, wm, L); else sub_n(wm, w1, wm, L);
    rshift1(wm, wm, L);

    w1[2 * n] -= sub_n(w1, w1, rb, 2 * n);

    sub_n(w2, w2, w1, L);
    rshift1(w2, w2, L);

    sub_n(w1, w1, wm, L);

    {
        // 2*c4 (st + 1 limbs) staged in the evaluation area, free since the products are done.
        LimbT* c4x2 = as1;
        c4x2[st] = lshift1(c4x2, c4, st);
        LimbT b = sub_n(w2, w2, c4x2, st + 1);
        if (st + 1 < L) sub_1(w2 + st + 1, w2 + st + 1, L - st - 1, b);
    }

    {
        LimbT b = sub_n(w1, w1, c4, st);
        if (st < L) sub_1(w1 + st, w1 + st, L - st, b);
    }

    sub_n(wm, wm, w2, L);

    // Compose: c2 first (its low 2n limbs are exactly the untouched gap, its top limb lands on
    // c4), then c1 and c3 on top.
    std::memcpy(rb + 2 * n, w1, 2 * n * sizeof(LimbT));
    add_1(c4, c4, st, w1[2 * n]);
    add_at(rb, re, n, wm, L);
    add_at(rb, re, 3 * n, w2, L);

    return re;
}

} // namespace detail

// Toom-3 unsigned multiplication.
// Preconditions: un >= vn, detail::toom3_split_fits(un, vn).
// Allocates the result buffer via alloc; all scratch of the recursion comes from scratch_alloc,
// which must serve allocations in LIFO order (see umul() for the one place it is chosen).
// Returns {ptr, size, capacity}.
template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul_toom3(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
{
    const size_t alloc_sz = u.size() + v.size();
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_toom3_impl(u, v, rb, scratch_alloc);
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
