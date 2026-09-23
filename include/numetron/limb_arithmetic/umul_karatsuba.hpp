// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <memory>
#include <cstring>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/stack_allocator.hpp"
#include "numetron/detail/assert.hpp"

#include "uadd.hpp"
#include "usub.hpp"
//#include "umul_basecase.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

template <std::unsigned_integral LimbT>
int uabs_diff(LimbT const* u, size_t un, LimbT const* v, size_t vn, LimbT* r, LimbT* re)
{
    LimbT const* ue = u + un - 1;
    LimbT const* ve = v + vn - 1;
    if (un > vn) {
        do {
            if (*ue) [[unlikely]] {
                usub_unchecked(u, ue + 1, v, ve + 1, r);
                while (r != re) *r++ = 0;
                return 1;
            }
            --ue; --un;
        } while (un > vn);
        // Now un == vn. Compare the top limbs to determine the sign.
    } else if (un < vn) {
        do {
            if (*ve) [[unlikely]] {
                usub_unchecked(v, ve + 1, u, ue + 1, r);
                while (r != re) *r++ = 0;
                return -1;
            }
            --ve; --vn;
        } while (un < vn);
        // Now un == vn. Compare the top limbs to determine the sign.
    }
    // un == vn. Compare the top limbs to determine the sign.
    for (;;) {
        if (*ue > *ve) {
            usub_unchecked(u, ue + 1, v, ve + 1, r);
            while (r != re) *r++ = 0;
            return 1;
        } else if (*ue < *ve) {
            usub_unchecked(v, ve + 1, u, ue + 1, r);
            while (r != re) *r++ = 0;
            return -1;
        }
        if (ue == u) break;
        --ue; --ve;
    }
    while (r != re) *r++ = 0;
    return 0;
}

// Forward declaration.
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_karatsuba_impl(
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    AllocatorT alloc);

// Dispatch: use basecase below threshold, Karatsuba above.
// Strips leading zeros, ensures un >= vn, then routes to the appropriate algorithm.
//template <std::unsigned_integral LimbT, typename AllocatorT>
//inline LimbT* umul_dispatch(
//    const LimbT* u, size_t un,
//    const LimbT* v, size_t vn,
//    LimbT* rb,
//    AllocatorT alloc)
//{
//    while (un > 0 && u[un - 1] == 0) --un;
//    while (vn > 0 && v[vn - 1] == 0) --vn;
//    if (un < vn) {
//        std::swap(u, v);
//        std::swap(un, vn);
//    }
//
//    if (is_karatsuba_applicable(un, vn)) {
//        //return umul_karatsuba_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
//    }
//    if (vn) {
//        return umul_basecase<LimbT>(u, un, v, vn, rb);
//    }
//    return rb;
//}

// Karatsuba multiplication core (Toom-2), structured after GMP's mpn_toom22_mul.
//
// Preconditions:
//   un >= vn >= 2, un < 2*vn  (nearly square; the dispatch additionally requires
//   vn >= karatsuba_threshold(), but that's a speed choice, not a correctness one)
//   rb[0 .. un+vn) is the output buffer (uninitialized)
//
// Split by un:  u = a0 + a1*B^n,  v = b0 + b1*B^n  with
//   s = floor(un/2) = |a1|,  n = un - s = |a0| = |b0|,  t = vn - n = |b1|,  0 <= t <= s <= n.
//
// Writing v0 = a0*b0 = L0 + H0*B^n, vinf = a1*b1 = Li + Hi*B^n, vm1 = |a0-a1|*|b0-b1|:
//   u*v = L0 + (L0 + H0 + Li)*B^n + (H0 + Li + Hi)*B^2n + Hi*B^3n  -/+  vm1*B^n
// so the shared sum X = H0 + Li is computed once and reused for both middle columns.
//
// Allocates at most 2n limbs of scratch (only for vm1, only when it's non-zero) via alloc
// and frees it before returning. Returns rb + un + vn (the end of the written result).
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_karatsuba_impl(std::span<const LimbT> u, std::span<const LimbT> v,
    LimbT* rb,
    AllocatorT alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    // Checked against the fixed floor rather than karatsuba_threshold(): the threshold is a
    // runtime tunable and may change while this multiplication is in flight.
    assert(un >= vn && vn >= 2 && 2 * vn > un);

    const size_t s = un / 2;
    const size_t n = un - s;

    LimbT const* a0 = u.data();
    LimbT const* a1 = a0 + n;
    LimbT const* b0 = v.data();
    LimbT const* b1 = b0 + n;

    LimbT* const re = rb + un + vn;

    LimbT* scratch = nullptr;
    size_t scratch_sz = 0;
    NUMETRON_SCOPE_EXIT([&] {
        if (scratch) std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });

    if (vn == n) [[unlikely]] {
        // t == 0 (un odd, vn == ceil(un/2)): v has no high half, so the three-product
        // identity doesn't apply (vinf would be shorter than the n limbs it's combined over).
        // u*v = a0*v + a1*v*B^n -- two nearly square products instead.
        LimbT* e = umul_dispatch(a0, n, b0, n, rb, alloc);
        std::memset(e, 0, (re - e) * sizeof(LimbT));

        scratch_sz = s + n;
        scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
        LimbT* pe = umul_dispatch(a1, s, b0, n, scratch, alloc);
        if (LimbT c = uadd_inplace(rb + n, scratch, pe))
            uadd_limb(rb + n + (pe - scratch), re, c);
        return re;
    }

    const size_t t = vn - n; // 0 < t <= s
    const size_t h = s + t - n; // |Hi|, 0 <= h <= n

    // |a0 - a1| and |b0 - b1| go into rb[0..2n), which stays free until v0 is computed last.
    LimbT* const asm1 = rb;
    LimbT* const bsm1 = rb + n;
    const int sign = uabs_diff(a0, n, a1, s, asm1, asm1 + n) * uabs_diff(b0, n, b1, t, bsm1, bsm1 + n);

    LimbT* vm1 = nullptr;
    if (sign) {
        scratch_sz = 2 * n;
        scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);
        vm1 = scratch;
        LimbT* e = umul_dispatch(asm1, n, bsm1, n, vm1, alloc);
        std::memset(e, 0, (vm1 + 2 * n - e) * sizeof(LimbT));
    }

    // vinf -> rb[2n .. re)
    {
        LimbT* e = umul_dispatch(a1, s, b1, t, rb + 2 * n, alloc);
        std::memset(e, 0, (re - e) * sizeof(LimbT));
    }
    // v0 -> rb[0 .. 2n), overwriting asm1/bsm1
    {
        LimbT* e = umul_dispatch(a0, n, b0, n, rb, alloc);
        std::memset(e, 0, (rb + 2 * n - e) * sizeof(LimbT));
    }

    // Carries out of the n-limb column windows, applied at the end: cy2 lands at limb 2n,
    // cy at limb 3n. cy2 is in [0, 2], cy in [-1, 2] (the final product fits, so every
    // intermediate over/underflow is resolved by these two adjustments).
    int cy, cy2;

    // rb[2n..3n) = X = H0 + Li
    {
        LimbT const* src = rb + n;
        LimbT* dst = rb + 2 * n;
        cy = uadd_partial_unchecked(src, rb + 2 * n, rb + 3 * n, dst);
    }
    // rb[n..2n) = X + L0
    {
        LimbT const* src = rb + 2 * n;
        LimbT* dst = rb + n;
        cy2 = cy + uadd_partial_unchecked(src, rb, rb + n, dst);
    }
    // rb[2n..3n) = X + Hi
    if (h) {
        LimbT c = uadd_inplace(rb + 2 * n, rb + 3 * n, rb + 3 * n + h);
        if (c && h < n) c = uadd_limb(rb + 2 * n + h, rb + 3 * n, c);
        cy += static_cast<int>(c);
    }
    // rb[n..3n) -/+= vm1
    if (sign > 0) {
        cy -= static_cast<int>(usub_inplace(rb + n, vm1, vm1 + 2 * n));
    } else if (sign < 0) {
        cy += static_cast<int>(uadd_inplace(rb + n, vm1, vm1 + 2 * n));
    }

    uadd_limb(rb + 2 * n, re, static_cast<LimbT>(cy2));
    if (cy > 0) {
        uadd_limb(rb + 3 * n, re, static_cast<LimbT>(cy));
    } else if (cy < 0) {
        usub_limb(rb + 3 * n, re, LimbT{ 1 });
    }

    return re;
}

} // namespace detail

// Karatsuba unsigned multiplication (Toom-2).
// Preconditions: un >= vn >= 2, un < 2*vn
// Allocates the result buffer via alloc; all scratch of the recursion comes from scratch_alloc,
// which must serve allocations in LIFO order (see umul() for the one place it is chosen).
// Returns {ptr, size, capacity}.
template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul_karatsuba(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    assert(un > 0 && vn > 0 && un >= vn);

    const size_t alloc_sz = un + vn;
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_karatsuba_impl(u, v, rb, scratch_alloc);
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
