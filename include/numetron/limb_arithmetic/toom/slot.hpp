// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <concepts>

#include "numetron/detail/assert.hpp"

//#include "numetron/limb_arithmetic/umul_karatsuba.hpp"   // detail::uabs_diff, detail::umul_dispatch
#include "numetron/limb_arithmetic/udivby1.hpp"
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
        size_t isz = (std::min)(dst.len, a.len);
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
        slot_clear(dst);
        return;
    }
    NUMETRON_ASSERT(dst.cap >= a.len + b.len);
    LimbT* re = umul_dispatch(a.ptr, a.len, b.ptr, b.len, dst.ptr, scratch_alloc);
    dst.len = static_cast<size_t>(re - dst.ptr);
    dst.sign = a.sign * b.sign;
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

#endif
}
