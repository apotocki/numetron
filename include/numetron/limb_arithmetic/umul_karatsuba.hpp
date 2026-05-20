// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <memory>
#include <cstring>

#include "numetron/detail/scope_exit.hpp"
#include "numetron/detail/stack_allocator.hpp"

#include "uadd.hpp"
#include "usub.hpp"
#include "umul_basecase.hpp"
#include "thresholds.hpp"

namespace numetron::limb_arithmetic {

inline bool is_karatsuba_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    // Condition is expressed in terms of vn so that Toom-k dispatch is uniform:
    // each algorithm checks vn >= its threshold and un < k*vn (u fits in k pieces).
    // For Karatsuba (k=2): vn >= threshold and un < 2*vn.
    return vn >= NUMETRON_KARATSUBA_THRESHOLD && 2 * vn > un;
}

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

    if (is_karatsuba_applicable(un, vn)) {
        return umul_karatsuba_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
    }
    if (vn) {
        return umul_basecase<LimbT>(u, un, v, vn, rb);
    }
    return rb;
}

// Karatsuba multiplication core (Toom-2).
//
// Preconditions:
//   un >= vn, vn >= NUMETRON_KARATSUBA_THRESHOLD, un < 2*vn  (nearly square)
//   rb[0 .. un+vn) is the output buffer (uninitialized)
//
// Allocates its own scratch via alloc and frees it before returning.
// Returns rb + un + vn (the end of the written result).
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_karatsuba_impl(std::span<const LimbT> u, std::span<const LimbT> v,
    LimbT* rb,
    AllocatorT alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    // Split by vn (consistent with future Toom-k where split size = ceil(vn/k)):
    //   n2   = ceil(vn/2)
    //   u_lo = u[0..n2),  u_hi = u[n2..un)  -- u_hi may be longer than u_lo
    //   v_lo = v[0..n2),  v_hi = v[n2..vn)  -- v_hi_n >= 1 always (vn >= threshold >= 2)
    const size_t n2 = (vn + 1) / 2;

    assert(un >= vn && vn >= NUMETRON_KARATSUBA_THRESHOLD && 2 * vn > un);

    LimbT const* u_lo = u.data();
    const size_t u_lo_n = n2;
    LimbT const* u_hi = u.data() + n2;
    const size_t u_hi_n = un - n2;    // may exceed n2 for mildly rectangular inputs

    LimbT const* v_lo = v.data();
    const size_t v_lo_n = n2;
    LimbT const* v_hi = v.data() + n2;
    const size_t v_hi_n = vn - n2;    // >= 1

    // d_u = |u_lo - u_hi| needs max(n2, u_hi_n) limbs because u_hi may be longer.
    const size_t d_buf_n = (std::max)(n2, u_hi_n);

    LimbT* re = rb + un + vn;

    // -----------------------------------------------------------------------
    // Allocate scratch for this level:
    //   [0              .. d_buf_n)          d_u    = |u_lo - u_hi|  (d_buf_n limbs)
    //   [d_buf_n        .. d_buf_n+n2)       d_v    = |v_lo - v_hi|  (n2 limbs)
    //   [d_buf_n+n2     .. d_buf_n+3*n2)     c0_tmp = u_lo*v_lo copy (2*n2 limbs)
    //   [d_buf_n+3*n2   .. 2*d_buf_n+4*n2)  c2_tmp = d_u * d_v      (d_buf_n+n2 limbs)
    // Recursive sub-calls allocate their own scratch independently.
    // -----------------------------------------------------------------------
    const size_t scratch_sz = 2 * d_buf_n + 4 * n2;
    LimbT* scratch = std::allocator_traits<AllocatorT>::allocate(alloc, scratch_sz);

    NUMETRON_SCOPE_EXIT([&alloc, scratch, scratch_sz] {
        std::allocator_traits<AllocatorT>::deallocate(alloc, scratch, scratch_sz);
    });

    LimbT* d_u    = scratch;
    LimbT* d_v    = scratch + d_buf_n;
    LimbT* c0_tmp = scratch + d_buf_n + n2;
    LimbT* c2_tmp = scratch + d_buf_n + 3 * n2;

    // -----------------------------------------------------------------------
    // d_u = |u_lo - u_hi|.  Either half may be longer; zero-extend to d_buf_n.
    // -----------------------------------------------------------------------
    int sign_u = uabs_diff(u_lo, u_lo_n, u_hi, u_hi_n, d_u, d_u + d_buf_n);

    // -----------------------------------------------------------------------
    // d_v = |v_lo - v_hi|.  Both halves are n2 / (vn-n2) limbs; zero-extend to n2.
    // -----------------------------------------------------------------------
    int sign_v = uabs_diff(v_lo, v_lo_n, v_hi, v_hi_n, d_v, d_v + n2);

    // -----------------------------------------------------------------------
    // c0 = u_lo * v_lo  ->  written directly to rb[0 .. 2*n2)
    // c1 = u_hi * v_hi  ->  written to rb[2*n2 .. un+vn)
    // -----------------------------------------------------------------------
    {
        LimbT* cxe = umul_dispatch(u_lo, u_lo_n, v_lo, v_lo_n, rb, alloc);
        LimbT* c0_end = rb + 2 * n2;
        std::memset(cxe, 0, (c0_end - cxe) * sizeof(LimbT));
        cxe = umul_dispatch(u_hi, u_hi_n, v_hi, v_hi_n, c0_end, alloc);
        std::memset(cxe, 0, (re - cxe) * sizeof(LimbT));
    }

    // -----------------------------------------------------------------------
    // c2_abs = d_u * d_v  (non-negative product of absolute differences)
    // -----------------------------------------------------------------------
    int sign = sign_u * sign_v;
    size_t c2_sz = 0;
    if (sign) {
        LimbT* c2e = umul_dispatch(d_u, d_buf_n, d_v, n2, c2_tmp, alloc);
        while (c2e != c2_tmp && !*(c2e - 1)) { --c2e; }
        c2_sz = c2e - c2_tmp;
    }

    // -----------------------------------------------------------------------
    // Accumulate middle term into rb[n2..un+vn).
    //
    // Karatsuba identity:  u*v = c0 + (c0 + c1 - sign*c2_abs)*B^n2 + c1*B^(2*n2)
    //
    // So we need:  rb[n2..un+vn) += c0 + c1 - sign*c2_abs
    //   sign > 0  => mid -= c2_abs
    //   sign < 0  => mid += c2_abs
    //   sign == 0 => c2_abs term vanishes
    //
    // c0 lives in rb[0..2*n2) and mid = rb+n2, so they overlap at rb[n2..2*n2).
    // We copy c0 into c0_tmp before modifying mid to avoid the overlap.
    // c1 lives in rb[2*n2..un+vn) = mid[n2..mid_len), so it must be added
    // before c0 is written into that region.
    // -----------------------------------------------------------------------
    {
        const size_t mid_len = (un + vn) - n2;
        LimbT* mid = rb + n2;

        std::memcpy(c0_tmp, rb, 2 * n2 * sizeof(LimbT));

        // mid += c1  (must come first: c1 lives at mid[n2..], which mid += c0 would corrupt)
        {
            LimbT* c1 = mid + n2;
            size_t c1_len = u_hi_n + v_hi_n;
            assert(c1_len <= mid_len);
            if (LimbT c = uadd_inplace(mid, c1, c1 + c1_len))
                uadd_limb(mid + c1_len, re, c);
        }

        // mid += c0
        {
            LimbT c = uadd_inplace(mid, c0_tmp, c0_tmp + (std::min)(2 * n2, mid_len));
            if (c && 2 * n2 < mid_len)
                uadd_limb(mid + 2 * n2, re, c);
        }

        // mid -= or += c2_abs
        if (sign > 0 && c2_sz) {
            LimbT b = usub_inplace(mid, c2_tmp, c2_tmp + (std::min)(c2_sz, mid_len));
            if (b && c2_sz < mid_len)
                usub_limb(mid + c2_sz, re, b);
        } else if (sign < 0 && c2_sz) {
            LimbT c = uadd_inplace(mid, c2_tmp, c2_tmp + (std::min)(c2_sz, mid_len));
            if (c && c2_sz < mid_len)
                uadd_limb(mid + c2_sz, re, c);
        }
    }

    return re;
}

} // namespace detail

// Karatsuba unsigned multiplication (Toom-2).
// Preconditions: un >= vn, vn >= NUMETRON_KARATSUBA_THRESHOLD, un < 2*vn
// Allocates result buffer via alloc; scratch is allocated internally per recursion level.
// Returns {ptr, size, capacity}.
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul_karatsuba(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
{
    const size_t un = u.size();
    const size_t vn = v.size();

    assert(un > 0 && vn > 0 && un >= vn);

    const size_t alloc_sz = un + vn;
    LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
    try {
        LimbT* re = detail::umul_karatsuba_impl(u, v, rb, numetron::detail::stack_allocator<LimbT>{});
        //LimbT* re = detail::umul_karatsuba_impl(u, v, rb, std::allocator<LimbT>{});
        while (re != rb && *(re - 1) == 0) --re;
        return { rb, static_cast<size_t>(re - rb), alloc_sz };
    }
    catch (...) {
        std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
        throw;
    }
}

}
