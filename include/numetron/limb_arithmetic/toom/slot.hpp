// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <concepts>
#include <cstring>
#include <utility>
#include <algorithm>

#include "numetron/detail/assert.hpp"

//#include "numetron/limb_arithmetic/umul_karatsuba.hpp"   // detail::uabs_diff, detail::umul_dispatch
#include "numetron/limb_arithmetic/udivby1.hpp"
#include "numetron/limb_arithmetic/toom/kernels.hpp"
#include "numetron/detail/stack_allocator.hpp"

namespace numetron::limb_arithmetic {

template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_dispatch(
    const LimbT* u, size_t un,
    const LimbT* v, size_t vn,
    LimbT* rb,
    AllocatorT alloc);

}

namespace numetron::limb_arithmetic::toom_runtime_detail {

template <std::unsigned_integral LimbT>
struct toom_slot
{
    LimbT* ptr = nullptr;
    size_t len = 0;
    size_t cap = 0;
    int sign = 0;
};

template <std::unsigned_integral LimbT>
inline void slot_trim(toom_slot<LimbT>& s)
{
    while (s.len && s.ptr[s.len - 1] == 0) --s.len;
    if (!s.len) s.sign = 0;
}

template <std::unsigned_integral LimbT>
inline void slot_clear(toom_slot<LimbT>& s)
{
    s.len = 0;
    s.sign = 0;
}

template <std::unsigned_integral LimbT>
inline void slot_copy(toom_slot<LimbT>& dst, toom_slot<LimbT> const& src)
{
    NUMETRON_ASSERT(dst.cap >= src.len);
    if (src.len) std::memcpy(dst.ptr, src.ptr, src.len * sizeof(LimbT));
    dst.len = src.len;
    dst.sign = src.sign;
}

template <std::unsigned_integral LimbT>
void slot_add_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> a, toom_slot<LimbT> b)
{
    slot_trim(a);
    if (!a.sign) { slot_copy(dst, b); return; }
    slot_trim(b);
    if (!b.sign) { slot_copy(dst, a); return; }

    if (a.sign == b.sign) {
        NUMETRON_ASSERT(dst.cap >= a.len);
        NUMETRON_ASSERT(dst.cap >= b.len);
        dst.sign = a.sign;
        size_t isz = (std::min)(a.len, b.len);
        LimbT const* aptr = a.ptr;
        LimbT* dstptr = dst.ptr;
        LimbT carry = uadd_partial_unchecked(aptr, b.ptr, b.ptr + isz, dstptr);
        if (isz < a.len) {
            std::memcpy(dstptr, aptr, (a.len - isz) * sizeof(LimbT));
            dst.len = a.len;
        } else if (isz < b.len) {
            std::memcpy(dstptr, b.ptr + isz, (b.len - isz) * sizeof(LimbT));
            dst.len = b.len;
        } else {
            dst.len = isz;
        }
        if (carry) {
            carry = uadd_limb(dst.ptr + isz, dst.ptr + dst.len, carry);
            if (carry) {
                NUMETRON_ASSERT(dst.cap >= dst.len + 1);
                if (dst.cap >= dst.len + 1) {
                    dst.ptr[dst.len] = carry;
                    ++dst.len;
                }
            }
        }
        return;
    }

    NUMETRON_ASSERT(a.len && b.len);
    LimbT const* plast_a = a.ptr + a.len - 1;
    LimbT const* plast_b = b.ptr + b.len - 1;

    bool do_swap = false;
    if (a.len == b.len) {
        while (*plast_a == *plast_b) {
            if (plast_a == a.ptr) {
                dst.len = 0;
                dst.sign = 0;
                return;
            }
            --plast_a; --plast_b; --a.len;
        }
        b.len = a.len;
        do_swap = *plast_a < *plast_b;
    }
    LimbT* dstptr = dst.ptr;
    if ((a.len < b.len) ^ do_swap) {
        std::swap(a, b);
    }
    LimbT const* aptr = a.ptr;
    LimbT const* aptr_e = a.ptr + a.len;
    NUMETRON_ASSERT(dst.cap >= a.len);
    LimbT borrow = usub_partial_unchecked<LimbT>(aptr, b.ptr, b.ptr + b.len, dstptr);
    if (borrow) {
        borrow = usub_partial_limb(aptr, aptr_e, borrow, dstptr);
        NUMETRON_ASSERT(!borrow);
    }
    dstptr = std::copy(aptr, aptr_e, dstptr);
    dst.sign = a.sign;
    dst.len = static_cast<size_t>(dstptr - dst.ptr);
}


template <std::unsigned_integral LimbT>
void slot_inplace_add(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a)
{
    // This path is intentionally window-oriented for rb accumulation:
    // carry/borrow that cannot be fully propagated inside dst window may be ignored.
    // It matches Karatsuba-style middle-term accumulation semantics on bounded rb spans.
    if (!a.sign) { return; }
    
    NUMETRON_ASSERT(dst.cap >= a.len);

    if (dst.sign == a.sign) {
        size_t isz = (std::min)(dst.len, a.len);
        auto carry = uadd_inplace(dst.ptr, a.ptr, a.ptr + isz);

        if (isz < a.len) {
            std::memcpy(dst.ptr + isz, a.ptr + isz, (a.len - isz) * sizeof(LimbT));
            dst.len = a.len;
        }
        if (carry) {
            carry = uadd_limb(dst.ptr + isz, dst.ptr + dst.len, carry);
            if (carry) {
                // we can store the carry if there is a storage for it, but we don't require it
                if (dst.cap >= dst.len + 1) {
                    dst.ptr[dst.len] = static_cast<LimbT>(carry);
                    ++dst.len;
                }
            }
        }
        return;
    }
    NUMETRON_ASSERT(dst.len > a.len);
    auto borrow = usub_inplace(dst.ptr, a.ptr, a.ptr + a.len);
    if (borrow) {
        borrow = usub_limb(dst.ptr + a.len, dst.ptr + dst.len, borrow);
        // just ignore the borrow
        (void)borrow;
    }
}

template <std::unsigned_integral LimbT, typename ScratchAllocatorT>
inline void slot_mul_dispatch(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b, ScratchAllocatorT scratch_alloc)
{
    if (!a.sign || !b.sign) {
        std::memset(dst.ptr, 0, dst.len * sizeof(LimbT));
        dst.sign = 0;
        return;
    }
    NUMETRON_ASSERT(dst.cap >= a.len + b.len);
    LimbT* re = umul_dispatch(a.ptr, a.len, b.ptr, b.len, dst.ptr, scratch_alloc);
    size_t result_len = static_cast<size_t>(re - dst.ptr);
    if (dst.len > result_len) {
        std::memset(dst.ptr + result_len, 0, (dst.len - result_len) * sizeof(LimbT));
    } else {
        dst.len = result_len;
    }
    dst.sign = a.sign * b.sign * !!result_len;
}

////////////

template <std::unsigned_integral LimbT>
inline void slot_mul_small(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, LimbT k)
{
    if (!a.sign || k == 0) {
        slot_clear(dst);
        return;
    }
    if (k == 1) {
        slot_copy(dst, a);
        return;
    }
    NUMETRON_ASSERT(dst.cap >= a.len + 1);
    LimbT* out = dst.ptr;
    LimbT hi = umul1<LimbT>(a.ptr, a.ptr + a.len, k, out);
    dst.len = a.len;
    if (hi) {
        dst.ptr[dst.len++] = hi;
    }
    dst.sign = a.sign;
    slot_trim(dst);
}

template <std::unsigned_integral LimbT>
inline void slot_divexact_small(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, LimbT d)
{
    if (!a.sign) {
        slot_clear(dst);
        return;
    }
    NUMETRON_ASSERT(dst.cap >= a.len);
    LimbT rem = udivby1<LimbT>(std::span<const LimbT>{a.ptr, a.len}, d, std::span<LimbT>{dst.ptr, a.len});
    NUMETRON_ASSERT(rem == 0);
    dst.len = a.len;
    dst.sign = a.sign;
    slot_trim(dst);
}

template <std::unsigned_integral LimbT>
inline void slot_add_shifted_to_result(LimbT* rb, size_t rsz, toom_slot<LimbT> const& c, size_t shift)
{
    if (!c.sign) return;
    NUMETRON_ASSERT(c.sign > 0);
    if (shift >= rsz) return;

    const size_t avail = rsz - shift;
    const size_t len = (std::min)(avail, c.len);
    if (!len) return;

    if (LimbT carry = uadd_inplace(rb + shift, c.ptr, c.ptr + len))
        uadd_limb(rb + shift + len, rb + rsz, carry);
}

//////////// Fixed-width ops (toom_op::uadd and following, see core.hpp)

// Length a fixed op reads from a source: its cap, clipped to the destination width. The clipped
// part must be zero -- the plan guarantees the value fits.
template <std::unsigned_integral LimbT>
inline size_t slot_fx_src_len(toom_slot<LimbT> const& s, size_t width) noexcept
{
    const size_t n = (std::min)(s.cap, width);
#ifndef NDEBUG // NUMETRON_ASSERT still evaluates its argument in release builds
    NUMETRON_ASSERT(std::all_of(s.ptr + n, s.ptr + s.cap, [](LimbT l) { return l == 0; }));
#endif
    return n;
}

// Zero-fills dst above the `written` limbs an op produced and marks all dst.cap limbs valid.
// The tail is almost always a limb or two (a carry limb that didn't materialize, the zero top
// of a product), so short tails are stored directly instead of calling into the CRT's memset.
template <std::unsigned_integral LimbT>
inline void slot_fx_finish(toom_slot<LimbT>& dst, size_t written, int sign = 1) noexcept
{
    NUMETRON_ASSERT(written <= dst.cap);
    LimbT* p = dst.ptr + written;
    switch (dst.cap - written) {
    case 0: break;
    case 3: p[2] = 0; [[fallthrough]];
    case 2: p[1] = 0; [[fallthrough]];
    case 1: p[0] = 0; break;
    default: std::memset(p, 0, (dst.cap - written) * sizeof(LimbT));
    }
    dst.len = dst.cap;
    dst.sign = sign;
}

// Exact-width variants: dst and the sources are tmp slots of the same width (checked when the
// plan is compiled, see same_tmp_width()), all fully written, so the kernel is called directly.

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_add_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    [[maybe_unused]] LimbT c = add_n(dst.ptr, a.ptr, b.ptr, dst.cap);
    NUMETRON_ASSERT(!c);
    dst.len = dst.cap;
    dst.sign = 1;
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_sub_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    [[maybe_unused]] LimbT br = sub_n(dst.ptr, a.ptr, b.ptr, dst.cap);
    NUMETRON_ASSERT(!br);
    dst.len = dst.cap;
    dst.sign = 1;
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_sub_signed_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b) noexcept
{
    if (b.sign < 0) {
        slot_fx_add_exact(dst, a, b);
    } else if (b.sign > 0) {
        slot_fx_sub_exact(dst, a, b);
    } else {
        if (dst.ptr != a.ptr) std::memmove(dst.ptr, a.ptr, dst.cap * sizeof(LimbT));
        dst.len = dst.cap;
        dst.sign = 1;
    }
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shl1_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    [[maybe_unused]] LimbT out = lshift1(dst.ptr, s.ptr, dst.cap);
    NUMETRON_ASSERT(!out);
    dst.len = dst.cap;
    dst.sign = 1;
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shr1_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    [[maybe_unused]] LimbT out = rshift1(dst.ptr, s.ptr, dst.cap);
    NUMETRON_ASSERT(!out);
    dst.len = dst.cap;
    dst.sign = 1;
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_divexact3_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    divexact_by3(dst.ptr, s.ptr, dst.cap);
    dst.len = dst.cap;
    dst.sign = 1;
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_add_into_exact(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    [[maybe_unused]] LimbT c = add_n(dst.ptr, dst.ptr, s.ptr, dst.cap);
    NUMETRON_ASSERT(!c);
    dst.len = dst.cap;
}

// r[0..xn) = x[0..xn) - y[0..yn), yn <= xn, no borrow out.
template <std::unsigned_integral LimbT>
inline void slot_fx_sub_raw(LimbT* r, LimbT const* x, size_t xn, LimbT const* y, size_t yn) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    NUMETRON_ASSERT(yn <= xn);
    LimbT br = sub_n(r, x, y, yn);
    if (xn > yn) br = sub_1(r + yn, x + yn, xn - yn, br);
    NUMETRON_ASSERT(!br);
}

// dst = a + b over dst.cap, general widths: the body of slot_fx_add(), also inlined into
// slot_fx_addsub_abs().
template <std::unsigned_integral LimbT>
inline void slot_fx_add_body(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t w = dst.cap;
    LimbT const* ap = a.ptr;
    LimbT const* bp = b.ptr;
    size_t an = slot_fx_src_len(a, w);
    size_t bn = slot_fx_src_len(b, w);
    if (an < bn) {
        std::swap(ap, bp);
        std::swap(an, bn);
    }
    LimbT c = add_n(dst.ptr, ap, bp, bn);
    if (an > bn) c = add_1(dst.ptr + bn, ap + bn, an - bn, c);
    // The carry limb is stored whether or not it's zero, so how much of dst is left for
    // slot_fx_finish() to clear depends only on the widths, not on the data.
    size_t written = an;
    if (an < w) {
        dst.ptr[an] = c;
        written = an + 1;
    } else {
        NUMETRON_ASSERT(!c);
    }
    slot_fx_finish(dst, written);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_add(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    slot_fx_add_body(dst, a, b);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_sub(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    const size_t w = dst.cap;
    const size_t an = slot_fx_src_len(a, w);
    size_t bn = slot_fx_src_len(b, w);
    while (bn > an && !b.ptr[bn - 1]) --bn; // a zero top of b wider than a is fine
    slot_fx_sub_raw(dst.ptr, a.ptr, an, b.ptr, bn);
    slot_fx_finish(dst, an);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_sub_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    if (b.sign < 0) {
        slot_fx_add(dst, a, b);
    } else if (b.sign > 0) {
        slot_fx_sub(dst, a, b);
    } else {
        slot_fx_add(dst, a, toom_slot<LimbT>{ b.ptr, 0, 0, 0 });
    }
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_abs_sub(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t w = dst.cap;
    const size_t an = slot_fx_src_len(a, w);
    const size_t bn = slot_fx_src_len(b, w);
    // Compare as zero-extended numbers without trimming either operand, so the width written
    // below depends only on which one is larger, not on how many zero limbs they happen to have.
    int c = 0;
    {
        size_t ai = an, bi = bn;
        while (ai > bi) { if (a.ptr[ai - 1]) { c = 1; break; } --ai; }
        while (!c && bi > ai) { if (b.ptr[bi - 1]) { c = -1; break; } --bi; }
        if (!c) c = cmp_n(a.ptr, b.ptr, ai);
    }
    if (c >= 0) {
        // b's limbs above an (if any) are zero here: a >= b
        slot_fx_sub_raw(dst.ptr, a.ptr, an, b.ptr, (std::min)(an, bn));
        slot_fx_finish(dst, an, c);
    } else {
        slot_fx_sub_raw(dst.ptr, b.ptr, bn, a.ptr, (std::min)(an, bn));
        slot_fx_finish(dst, bn, -1);
    }
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shl1(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    size_t written = sn;
    if (sn) {
        LimbT out = lshift1(dst.ptr, s.ptr, sn);
        // stored unconditionally when there's room, see slot_fx_add()
        if (sn < dst.cap) {
            dst.ptr[sn] = out;
            written = sn + 1;
        } else {
            NUMETRON_ASSERT(!out);
        }
    }
    slot_fx_finish(dst, written);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shr1(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    if (sn) {
        [[maybe_unused]] LimbT out = rshift1(dst.ptr, s.ptr, sn);
        NUMETRON_ASSERT(!out);
    }
    slot_fx_finish(dst, sn);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_divexact3(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    divexact_by3(dst.ptr, s.ptr, sn);
    slot_fx_finish(dst, sn);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_copy_low(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s)
{
    const size_t n = (std::min)(s.cap, dst.cap);
    if (n && dst.ptr != s.ptr) std::memmove(dst.ptr, s.ptr, n * sizeof(LimbT));
    slot_fx_finish(dst, n);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_add_into(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t w = dst.cap;
    const size_t sn = slot_fx_src_len(s, w);
    LimbT c = add_n(dst.ptr, dst.ptr, s.ptr, sn);
    if (c && sn < w) c = add_1(dst.ptr + sn, dst.ptr + sn, w - sn, c);
    NUMETRON_ASSERT(!c);
    dst.len = dst.cap;
}

template <std::unsigned_integral LimbT, typename ScratchAllocatorT>
NUMETRON_NOINLINE void slot_fx_mul(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b, ScratchAllocatorT scratch_alloc)
{
    NUMETRON_ASSERT(dst.cap >= a.cap + b.cap);
    if (!a.cap || !b.cap) {
        slot_fx_finish(dst, 0, 0);
        return;
    }
    // umul_dispatch() strips leading zero limbs of its operands, so how much it leaves unwritten
    // at the top varies with the data -- usually 0 to 2 limbs (the evaluated operands' small top
    // limbs). Those two are cleared up front and the product simply overwrites them if it
    // reaches that far, leaving only the rare longer tail to handle afterwards.
    const size_t pre = (std::min)(dst.cap, size_t{ 2 });
    for (size_t i = dst.cap - pre; i < dst.cap; ++i) dst.ptr[i] = 0;
    LimbT* e = umul_dispatch(a.ptr, a.cap, b.ptr, b.cap, dst.ptr, scratch_alloc);
    const size_t written = static_cast<size_t>(e - dst.ptr);
    if (written + pre < dst.cap) [[unlikely]] {
        std::memset(e, 0, (dst.cap - pre - written) * sizeof(LimbT));
    }
    dst.len = dst.cap;
    dst.sign = a.sign * b.sign;
}

// toom_op::eval_pm1: E = e0 + e1, dst = E + o, dst2 = |E - o| (sign in dst2.sign).
// E is formed in dst2's storage, so it needs no slot of its own and is computed once for both
// points; dst is written from it before dst2 is turned into the difference in place.
template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_eval_pm1(toom_slot<LimbT>& dst, toom_slot<LimbT>& dst2,
    toom_slot<LimbT> const& e0, toom_slot<LimbT> const& e1, toom_slot<LimbT> const& o) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    NUMETRON_ASSERT(dst.ptr != dst2.ptr);
    const size_t w2 = dst2.cap;
    LimbT* g = dst2.ptr;

    // E = e0 + e1, all w2 limbs of dst2 written
    {
        LimbT const* ap = e0.ptr;
        LimbT const* bp = e1.ptr;
        size_t an = slot_fx_src_len(e0, w2);
        size_t bn = slot_fx_src_len(e1, w2);
        if (an < bn) {
            std::swap(ap, bp);
            std::swap(an, bn);
        }
        LimbT c = add_n(g, ap, bp, bn);
        if (an > bn) c = add_1(g + bn, ap + bn, an - bn, c);
        if (an < w2) {
            g[an] = c;
            for (size_t i = an + 1; i < w2; ++i) g[i] = 0;
        } else {
            NUMETRON_ASSERT(!c);
        }
    }

    const size_t on = slot_fx_src_len(o, w2);

    // sign of E - o: E's limbs above o's width decide first, then the common part
    int sgn = 0;
    for (size_t i = w2; i > on; --i) {
        if (g[i - 1]) { sgn = 1; break; }
    }
    if (!sgn) sgn = cmp_n(g, o.ptr, on);

    // dst = E + o
    {
        const size_t w = dst.cap;
        NUMETRON_ASSERT(w >= w2);
        LimbT c = add_n(dst.ptr, g, o.ptr, on);
        if (w2 > on) c = add_1(dst.ptr + on, g + on, w2 - on, c);
        size_t written = w2;
        if (w2 < w) {
            dst.ptr[w2] = c;
            written = w2 + 1;
        } else {
            NUMETRON_ASSERT(!c);
        }
        slot_fx_finish(dst, written);
    }

    // dst2 = |E - o|, in place over E
    if (sgn >= 0) {
        slot_fx_sub_raw(g, g, w2, o.ptr, on);
        dst2.len = dst2.cap;
        dst2.sign = sgn;
    } else {
        // E < o: E's limbs above on are zero, so o - E fits in on limbs; the rest of dst2 stays zero
        [[maybe_unused]] LimbT br = sub_n(g, o.ptr, g, on);
        NUMETRON_ASSERT(!br);
        dst2.len = dst2.cap;
        dst2.sign = -1;
    }
}

// toom_op::addsub_abs: dst = a + b, dst2 = |a - b| (sign in dst2.sign). dst must not alias a or
// b; dst2 may be a (the difference is formed in place after the sum was taken).
template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_addsub_abs(toom_slot<LimbT>& dst, toom_slot<LimbT>& dst2,
    toom_slot<LimbT> const& a, toom_slot<LimbT> const& b) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    NUMETRON_ASSERT(dst.ptr != a.ptr && dst.ptr != b.ptr);
    const size_t w2 = dst2.cap;
    const size_t an = slot_fx_src_len(a, w2);
    const size_t bn = slot_fx_src_len(b, w2);

    int sgn = 0;
    {
        size_t ai = an, bi = bn;
        while (ai > bi) { if (a.ptr[ai - 1]) { sgn = 1; break; } --ai; }
        while (!sgn && bi > ai) { if (b.ptr[bi - 1]) { sgn = -1; break; } --bi; }
        if (!sgn) sgn = cmp_n(a.ptr, b.ptr, ai);
    }

    slot_fx_add_body(dst, a, b);

    if (sgn >= 0) {
        slot_fx_sub_raw(dst2.ptr, a.ptr, an, b.ptr, (std::min)(an, bn));
        slot_fx_finish(dst2, an, sgn);
    } else {
        slot_fx_sub_raw(dst2.ptr, b.ptr, bn, a.ptr, (std::min)(an, bn));
        slot_fx_finish(dst2, bn, -1);
    }
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shl(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s, unsigned k)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    size_t written = sn;
    if (sn) {
        LimbT out = lshift(dst.ptr, s.ptr, sn, k);
        if (sn < dst.cap) {
            dst.ptr[sn] = out;
            written = sn + 1;
        } else {
            NUMETRON_ASSERT(!out);
        }
    }
    slot_fx_finish(dst, written);
}

template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_shr(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s, unsigned k)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    if (sn) {
        [[maybe_unused]] LimbT out = rshift(dst.ptr, s.ptr, sn, k);
        NUMETRON_ASSERT(!out);
    }
    slot_fx_finish(dst, sn);
}

// dst = a + (b << k), general widths. dst may be a or b (the kernels stream).
template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_addlsh(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b, unsigned k)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    const size_t w = dst.cap;
    const size_t an = slot_fx_src_len(a, w);
    const size_t bn = slot_fx_src_len(b, w);
    const size_t m = (std::min)(an, bn);
    LimbT hi = 0;
    unsigned char c = static_cast<unsigned char>(addlsh_n(dst.ptr, a.ptr, b.ptr, m, k, hi));
    size_t i = m;
    for (; i < an; ++i) { // b exhausted: its shifted-out bits, then nothing
        dst.ptr[i] = arithmetic::uadd1c(a.ptr[i], hi, c);
        hi = 0;
    }
    for (; i < bn; ++i) { // a exhausted
        const LimbT bi = b.ptr[i];
        dst.ptr[i] = arithmetic::uadd1c((bi << k) | hi, LimbT{ 0 }, c);
        hi = bi >> rk;
    }
    const LimbT top = hi + c;
    size_t written = i;
    if (i < w) {
        dst.ptr[i] = top;
        written = i + 1;
    } else {
        NUMETRON_ASSERT(!top);
    }
    slot_fx_finish(dst, written);
}

// dst = a - (b << k) >= 0, general widths. dst may be a or b (the kernels stream).
template <std::unsigned_integral LimbT>
NUMETRON_NOINLINE void slot_fx_sublsh(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b, unsigned k)
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const unsigned rk = std::numeric_limits<LimbT>::digits - k;
    const size_t w = dst.cap;
    const size_t an = slot_fx_src_len(a, w);
    const size_t bn = slot_fx_src_len(b, w);
    const size_t m = (std::min)(an, bn);
    LimbT hi = 0;
    unsigned char br = static_cast<unsigned char>(sublsh_n(dst.ptr, a.ptr, b.ptr, m, k, hi));
    size_t i = m;
    for (; i < an; ++i) { // b exhausted: its shifted-out bits, then only the borrow
        dst.ptr[i] = sbb1(a.ptr[i], hi, br);
        hi = 0;
    }
    for (; i < bn; ++i) { // a exhausted: b's remaining shifted limbs must cancel out
        const LimbT bi = b.ptr[i];
        dst.ptr[i] = sbb1(LimbT{ 0 }, (bi << k) | hi, br);
        hi = bi >> rk;
    }
    NUMETRON_ASSERT(!hi && !br);
    slot_fx_finish(dst, i);
}

template <std::unsigned_integral LimbT, unsigned D>
NUMETRON_NOINLINE void slot_fx_divexact_by(toom_slot<LimbT>& dst, toom_slot<LimbT> const& s) noexcept
{
    using namespace numetron::limb_arithmetic::detail::toom_kernels;
    const size_t sn = slot_fx_src_len(s, dst.cap);
    divexact_by<LimbT, D>(dst.ptr, s.ptr, sn);
    slot_fx_finish(dst, sn);
}

// Desc with term J's slot_sign resolved: negated once more when the slot's value is negative.
consteval detail::toom_kernels::lincomb_desc lincomb_resolve_sign(detail::toom_kernels::lincomb_desc d, unsigned j, bool negative)
{
    d.slot_sign[j] = false;
    if (negative) d.neg[j] = !d.neg[j];
    return d;
}

// Picks the kernel instantiation for the runtime signs of the slot_sign terms (bit j of
// neg_mask: term j's slot is negative), one branch per such term.
template <std::unsigned_integral LimbT, detail::toom_kernels::lincomb_desc Desc, unsigned J>
NUMETRON_FORCEINLINE void slot_fx_lincomb_signed(LimbT* r, size_t w, detail::toom_kernels::lincomb_src<LimbT> const* src, unsigned neg_mask) noexcept
{
    if constexpr (J == Desc.count) {
        numetron::limb_arithmetic::detail::toom_kernels::lincomb<LimbT, Desc>(r, w, src);
    } else if constexpr (!Desc.slot_sign[J]) {
        slot_fx_lincomb_signed<LimbT, Desc, J + 1>(r, w, src, neg_mask);
    } else {
        if ((neg_mask >> J) & 1u) slot_fx_lincomb_signed<LimbT, lincomb_resolve_sign(Desc, J, true), J + 1>(r, w, src, neg_mask);
        else slot_fx_lincomb_signed<LimbT, lincomb_resolve_sign(Desc, J, false), J + 1>(r, w, src, neg_mask);
    }
}

// toom_op::lincomb. Sources are read over their caps clipped to dst.cap (zero-extended), dst gets
// all dst.cap limbs. dst may be one of the sources (the same memory) but not overlap one otherwise.
template <std::unsigned_integral LimbT, detail::toom_kernels::lincomb_desc Desc>
NUMETRON_NOINLINE void slot_fx_lincomb(toom_slot<LimbT>& dst,
    toom_slot<LimbT> const& s0, toom_slot<LimbT> const& s1, toom_slot<LimbT> const& s2, toom_slot<LimbT> const& s3) noexcept
{
    using detail::toom_kernels::lincomb_src;
    static_assert(!Desc.slot_sign[0], "lincomb: the first term must be added");
    toom_slot<LimbT> const* const s[4] = { &s0, &s1, &s2, &s3 };
    const size_t w = dst.cap;
    lincomb_src<LimbT> src[4] = {};
    unsigned neg_mask = 0;
    for (unsigned j = 0; j < Desc.count; ++j) {
        const size_t n = slot_fx_src_len(*s[j], w);
        NUMETRON_ASSERT(s[j]->ptr == dst.ptr || s[j]->ptr + n <= dst.ptr || dst.ptr + w <= s[j]->ptr);
        src[j] = lincomb_src<LimbT>{ s[j]->ptr, n };
        if (Desc.slot_sign[j] && s[j]->sign < 0) neg_mask |= 1u << j;
    }
    slot_fx_lincomb_signed<LimbT, Desc, 0>(dst.ptr, w, src, neg_mask);
    dst.len = w;
    dst.sign = 1;
}

// toom_op::lincomb_dual: dst from s0..s3 and dst_b from sb0..sb3, same shape (no slot-signed
// terms), both over dst.cap limbs (dst_b.cap must match).
template <std::unsigned_integral LimbT, detail::toom_kernels::lincomb_desc Desc>
NUMETRON_NOINLINE void slot_fx_lincomb_dual(toom_slot<LimbT>& dst, toom_slot<LimbT>& dst_b,
    toom_slot<LimbT> const& s0, toom_slot<LimbT> const& s1, toom_slot<LimbT> const& s2, toom_slot<LimbT> const& s3,
    toom_slot<LimbT> const& sb0, toom_slot<LimbT> const& sb1, toom_slot<LimbT> const& sb2, toom_slot<LimbT> const& sb3) noexcept
{
    using detail::toom_kernels::lincomb_src;
    static_assert(!Desc.slot_sign[0] && !Desc.slot_sign[1] && !Desc.slot_sign[2] && !Desc.slot_sign[3],
        "lincomb_dual: no slot-signed terms");
    const size_t w = dst.cap;
    NUMETRON_ASSERT(dst_b.cap == w);
    NUMETRON_ASSERT(dst.ptr + w <= dst_b.ptr || dst_b.ptr + w <= dst.ptr);
    toom_slot<LimbT> const* const sa[4] = { &s0, &s1, &s2, &s3 };
    toom_slot<LimbT> const* const sb[4] = { &sb0, &sb1, &sb2, &sb3 };
    lincomb_src<LimbT> srca[4] = {};
    lincomb_src<LimbT> srcb[4] = {};
    for (unsigned j = 0; j < Desc.count; ++j) {
        const size_t na = slot_fx_src_len(*sa[j], w);
        const size_t nb = slot_fx_src_len(*sb[j], w);
        // each result may be its own sources (same memory), but not touch the other stream's
        NUMETRON_ASSERT(sa[j]->ptr == dst.ptr || sa[j]->ptr + na <= dst.ptr || dst.ptr + w <= sa[j]->ptr);
        NUMETRON_ASSERT(sb[j]->ptr == dst_b.ptr || sb[j]->ptr + nb <= dst_b.ptr || dst_b.ptr + w <= sb[j]->ptr);
        NUMETRON_ASSERT(sa[j]->ptr + na <= dst_b.ptr || dst_b.ptr + w <= sa[j]->ptr);
        NUMETRON_ASSERT(sb[j]->ptr + nb <= dst.ptr || dst.ptr + w <= sb[j]->ptr);
        srca[j] = lincomb_src<LimbT>{ sa[j]->ptr, na };
        srcb[j] = lincomb_src<LimbT>{ sb[j]->ptr, nb };
    }
    numetron::limb_arithmetic::detail::toom_kernels::lincomb_dual<LimbT, Desc>(dst.ptr, srca, dst_b.ptr, srcb, w);
    dst.len = w;
    dst.sign = 1;
    dst_b.len = w;
    dst_b.sign = 1;
}

#if 0

template <std::unsigned_integral LimbT>
inline int slot_cmp_mag(toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    if (a.len < b.len) return -1;
    if (a.len > b.len) return 1;
    for (size_t i = a.len; i-- > 0;) {
        if (a.ptr[i] < b.ptr[i]) return -1;
        if (a.ptr[i] > b.ptr[i]) return 1;
    }
    return 0;
}

template <std::unsigned_integral LimbT>
inline void slot_sub_mag_ge(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    NUMETRON_ASSERT(slot_cmp_mag(a, b) >= 0);
    NUMETRON_ASSERT(dst.cap >= a.len);
    if (&dst != &a && a.len) {
        std::memcpy(dst.ptr, a.ptr, a.len * sizeof(LimbT));
    }
    dst.len = a.len;
    dst.sign = 1;
    if (b.len) {
        LimbT br = usub_inplace(dst.ptr, b.ptr, b.ptr + b.len);
        if (br)
            usub_limb(dst.ptr + b.len, dst.ptr + dst.len, br);
    }
    slot_trim(dst);
    if (!dst.len) dst.sign = 0;
}

template <std::unsigned_integral LimbT>
inline void slot_add_mag(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    const size_t m = (std::max)(a.len, b.len);
    //NUMETRON_ASSERT(dst.cap >= m + 1);
    if (a.len) {
        std::memcpy(dst.ptr, a.ptr, a.len * sizeof(LimbT));
    }
    for (size_t i = a.len; i < m + 1; ++i) dst.ptr[i] = 0;

    LimbT carry = 0;
    if (b.len) {
        carry = uadd_inplace(dst.ptr, b.ptr, b.ptr + b.len);
        if (carry) {
            if (dst.cap >= m + 1) {
                carry = uadd_limb(dst.ptr + b.len, dst.ptr + m + 1, carry);
            } else {
                carry = uadd_limb(dst.ptr + b.len, dst.ptr + m, carry);
            }
        }
    }
    dst.len = m + (carry ? 1 : 0);
    if (!carry) {
        while (dst.len && dst.ptr[dst.len - 1] == 0) --dst.len;
    }
    dst.sign = dst.len ? 1 : 0;
}

template <std::unsigned_integral LimbT>
inline void slot_set_positive(toom_slot<LimbT>& dst, std::span<const LimbT> src)
{
    NUMETRON_ASSERT(dst.cap >= src.size());
    if (!src.empty()) std::memcpy(dst.ptr, src.data(), src.size() * sizeof(LimbT));
    dst.len = src.size();
    dst.sign = dst.len ? 1 : 0;
    slot_trim(dst);
    if (!dst.len) dst.sign = 0;
}

//template <std::unsigned_integral LimbT>
//inline void slot_add_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
//{
//    if (!a.sign) { slot_copy(dst, b); return; }
//    if (!b.sign) { slot_copy(dst, a); return; }
//
//    if (a.sign == b.sign) {
//        slot_add_mag(dst, a, b);
//        dst.sign = a.sign;
//        return;
//    }
//
//    int c = slot_cmp_mag(a, b);
//    if (c == 0) {
//        slot_clear(dst);
//    } else if (c > 0) {
//        slot_sub_mag_ge(dst, a, b);
//        dst.sign = a.sign;
//    } else {
//        slot_sub_mag_ge(dst, b, a);
//        dst.sign = b.sign;
//    }
//}









#endif
}
