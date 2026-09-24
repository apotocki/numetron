// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <memory>
#include <cstring>
#include <utility>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/stack_allocator.hpp"
#include "numetron/detail/assert.hpp"

#include "platform.hpp"
#include "uadd.hpp"
#include "usub.hpp"
#include "umul_karatsuba.hpp" // detail::uabs_diff

// Alternative Karatsuba: umul_karatsuba_impl with the three middle-column add passes of the
// interpolation fused into one (the assembly numetron_karatsuba_interp where available).
// Selected with NUMETRON_KARATSUBA_IMPL (numetron/config/implementation.hpp).

namespace numetron::limb_arithmetic {

template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_dispatch(
    const LimbT* u, size_t un,
    const LimbT* v, size_t vn,
    LimbT* rb,
    AllocatorT alloc);

namespace detail {

// The fused middle-column pass. On entry rp[0..2n) = v0 = L0 + H0*B^n and
// rp[2n..3n+h) = vinf = Li + Hi*B^n (h <= n). With X = H0 + Li, one pass over i in [0, n) writes
//   rp[n..2n)  = X + L0        (low n limbs)
//   rp[2n..3n) = X + Hi        (low n limbs; Hi zero-extended to n limbs)
// and returns {cX + cL, cX + cH}: the carries into limbs 2n and 3n, each in [0, 2] (cX, cL, cH
// are the carries out of X, X + L0, X + Hi). Nothing above limb 3n is touched.
template <std::unsigned_integral LimbT>
inline std::pair<unsigned, unsigned> karatsuba_interp(LimbT* rp, size_t n, size_t h) noexcept
{
    assert(n >= 1 && h <= n);
#if defined(NUMETRON_USE_ASM) && (defined(__x86_64__) || defined(_M_X64))
    if constexpr (sizeof(LimbT) == 8) {
        const uint64_t c = numetron_karatsuba_interp(reinterpret_cast<uint64_t*>(rp), n, h);
        return { static_cast<unsigned>(c & 0xffffffffu), static_cast<unsigned>(c >> 32) };
    } else
#endif
    {
        unsigned char cx = 0, cl = 0, ch = 0;
        LimbT* const p = rp + n;
        for (size_t i = 0; i < n; ++i) {
            const LimbT x = arithmetic::uadd1c(p[i], p[n + i], cx);
            const LimbT y = arithmetic::uadd1c(x, i < h ? p[2 * n + i] : LimbT{ 0 }, ch);
            p[i] = arithmetic::uadd1c(x, rp[i], cl);
            p[n + i] = y;
        }
        return { static_cast<unsigned>(cx) + cl, static_cast<unsigned>(cx) + ch };
    }
}

// Karatsuba multiplication core; same contract, split and scratch use as umul_karatsuba_impl
// (see there for the derivation). Only the interpolation differs: the middle columns
//   rb[n..2n) = L0 + H0 + Li,   rb[2n..3n) = H0 + Li + Hi
// come from one karatsuba_interp pass instead of three add passes; vm1 is then subtracted or
// added over rb[n..3n) as before.
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_karatsuba_fused_impl(std::span<const LimbT> u, std::span<const LimbT> v,
    LimbT* rb,
    AllocatorT alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

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
        // t == 0: u*v = a0*v + a1*v*B^n, as in umul_karatsuba_impl.
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

    // cy2 lands at limb 2n (in [0, 2]), cy at limb 3n (in [-1, 2] after vm1).
    auto [c2n, c3n] = karatsuba_interp(rb, n, h);
    const int cy2 = static_cast<int>(c2n);
    int cy = static_cast<int>(c3n);

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

// umul_karatsuba() running umul_karatsuba_fused_impl. Same contract.
template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul_karatsuba_fused(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    assert(un > 0 && vn > 0 && un >= vn);

    const size_t alloc_sz = un + vn;
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_karatsuba_fused_impl(u, v, rb, scratch_alloc);
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
