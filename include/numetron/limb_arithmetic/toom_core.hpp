// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <memory>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <array>
#include <utility>
#include <concepts>

#include "numetron/detail/assert.hpp"

#include "umul_karatsuba.hpp"   // detail::uabs_diff, detail::umul_dispatch
#include "udivby1.hpp"
#include "numetron/detail/stack_allocator.hpp"

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



template <size_t Parts, std::unsigned_integral LimbT>
inline std::span<const LimbT> resolve_input_part(std::span<const LimbT> src, size_t chunk, unsigned short part_id)
{
    const size_t idx = static_cast<size_t>(part_id);
    const size_t start = idx * chunk;
    if (start >= src.size()) {
        return {};
    }

    if (idx + 1 < Parts) {
        const size_t n = (std::min)(chunk, src.size() - start);
        return { src.data() + start, n };
    }

    return { src.data() + start, src.size() - start };
}

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

template <std::unsigned_integral LimbT, typename ScratchAllocatorT>
inline void slot_mul_dispatch(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b, ScratchAllocatorT scratch_alloc)
{
    if (!a.sign || !b.sign) {
        slot_clear(dst);
        return;
    }
    NUMETRON_ASSERT(dst.cap >= a.len + b.len);
    LimbT* re = detail::umul_dispatch(a.ptr, a.len, b.ptr, b.len, dst.ptr, scratch_alloc);
    dst.len = static_cast<size_t>(re - dst.ptr);
    dst.sign = a.sign * b.sign;
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

enum class toom_op : unsigned char
{
    // dst <- 0
    clear,
    // dst <- src0
    copy,
    // dst <- src0 + src1 (signed slots)
    add,
    // dst += src0 (signed slots)
    inplace_add,
    // dst <- src0 - src1 (signed slots)
    sub,
    // dst -= src0 (signed slots)
    inplace_sub,
    // dst <- src0 * imm
    mul_small,
    // dst <- src0 / imm, exact division is required (remainder must be 0)
    divexact_small,
    // dst <- src0 * src1
    mul_block,
    // rb += src0 << (imm * chunk) limbs
    compose_shifted,
    // print dst for debugging (src0, src1 are ignored)
    print
};

enum class toom_mem_kind : unsigned char
{
    u = 0,
    v = 1,
    rb = 2,
    tmp = 3,
};

struct toom_ref
{
    // Packed ref: [kind:2 bits | expr/id:14 bits]
    unsigned short bits = 0;
};

[[nodiscard]] constexpr toom_ref make_ref(toom_mem_kind kind, unsigned short expr_id) noexcept
{
    return toom_ref{ static_cast<unsigned short>((static_cast<unsigned short>(kind) << 14) | (expr_id & 0x3FFF)) };
}

[[nodiscard]] constexpr toom_mem_kind ref_kind(toom_ref ref) noexcept
{
    return static_cast<toom_mem_kind>((ref.bits >> 14) & 0x3u);
}

[[nodiscard]] constexpr unsigned short ref_expr_id(toom_ref ref) noexcept
{
    return static_cast<unsigned short>(ref.bits & 0x3FFFu);
}

struct toom_instr
{
    toom_op op;
    // Destination ref (expected tmp for all ops except compose_shifted where dst is ignored)
    toom_ref dst;
    // Primary source ref
    toom_ref src0;
    // Secondary source ref (used by binary ops)
    toom_ref src1;
    // Immediate argument (small multiplier/divisor or compose shift index)
    unsigned short imm;
    unsigned short imm2 = 0;
    unsigned short imm3 = 0;
};

enum class toom_size_var : unsigned short
{
    un,
    vn,
    chunk,
    u_hi,
    v_hi,
    d_buf_n,
};

enum class toom_size_expr_op : unsigned char
{
    constant,
    variable,
    add,
    sub,
    mul_const,
    max2,
};

struct toom_size_expr
{
    toom_size_expr_op op;
    uint64_t a;
    uint64_t b;
};

consteval uint_least64_t make_const(uint_least64_t value)
{
    CONSTEVAL_STATIC_ASSERT((value >> 62) == 0, "Constant value out of range");
    return value | 0x8000000000000000;
}

consteval uint_least64_t make_variable(toom_size_var value)
{
    return static_cast<uint_least64_t>(static_cast<unsigned short>(value)) | 0xC000000000000000;
}

struct toom_slot_layout
{
    toom_mem_kind kind;
    //signed char sign;
    unsigned short var;
    unsigned short off_expr;
    unsigned short cap_expr;
};

template <size_t>
inline constexpr bool always_false_v = false;

template <size_t N, size_t M>
struct toom_stage_traits
{
    static_assert(always_false_v<N + M>, "Missing toom_stage_traits specialization for this (N,M)");
};

} // namespace numetron::limb_arithmetic::toom_runtime_detail
