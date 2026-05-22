// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <memory>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <vector>
#include <array>
#include <utility>

#include "toom_plan.hpp"
#include "umul_karatsuba.hpp"   // detail::uabs_diff, detail::umul_dispatch
#include "udivby1.hpp"
#include "numetron/detail/stack_allocator.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

template <std::unsigned_integral LimbT>
struct toom_slot
{
    LimbT* ptr = nullptr;
    size_t len = 0;
    size_t cap = 0;
    int sign = 0;
};

template <std::unsigned_integral LimbT>
struct toom_stage_context
{
    LimbT* slab = nullptr;
    size_t slab_len = 0;
    bool slab_owned = false;
    std::array<toom_slot<LimbT>, 64> slots{};
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
    assert(dst.cap >= src.len);
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
    assert(slot_cmp_mag(a, b) >= 0);
    assert(dst.cap >= a.len);
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
    assert(dst.cap >= m + 1);
    if (a.len) {
        std::memcpy(dst.ptr, a.ptr, a.len * sizeof(LimbT));
    }
    for (size_t i = a.len; i < m + 1; ++i) dst.ptr[i] = 0;

    LimbT carry = 0;
    if (b.len) {
        carry = uadd_inplace(dst.ptr, b.ptr, b.ptr + b.len);
        if (carry)
            carry = uadd_limb(dst.ptr + b.len, dst.ptr + m + 1, carry);
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
    assert(dst.cap >= src.size());
    if (!src.empty()) std::memcpy(dst.ptr, src.data(), src.size() * sizeof(LimbT));
    dst.len = src.size();
    dst.sign = dst.len ? 1 : 0;
    slot_trim(dst);
    if (!dst.len) dst.sign = 0;
}

template <std::unsigned_integral LimbT>
inline void slot_add_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    if (!a.sign) { slot_copy(dst, b); return; }
    if (!b.sign) { slot_copy(dst, a); return; }

    if (a.sign == b.sign) {
        slot_add_mag(dst, a, b);
        dst.sign = a.sign;
        return;
    }

    int c = slot_cmp_mag(a, b);
    if (c == 0) {
        slot_clear(dst);
    } else if (c > 0) {
        slot_sub_mag_ge(dst, a, b);
        dst.sign = a.sign;
    } else {
        slot_sub_mag_ge(dst, b, a);
        dst.sign = b.sign;
    }
}

template <std::unsigned_integral LimbT>
inline void slot_sub_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    toom_slot<LimbT> nb = b;
    nb.sign = -nb.sign;
    slot_add_signed(dst, a, nb);
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
    assert(dst.cap >= a.len + 1);
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
    assert(dst.cap >= a.len);
    LimbT rem = udivby1<LimbT>(std::span<const LimbT>{a.ptr, a.len}, d, std::span<LimbT>{dst.ptr, a.len});
    assert(rem == 0);
    dst.len = a.len;
    dst.sign = a.sign;
    slot_trim(dst);
}

template <std::unsigned_integral LimbT>
inline void slot_mul_dispatch(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a, toom_slot<LimbT> const& b)
{
    if (!a.sign || !b.sign) {
        slot_clear(dst);
        return;
    }
    assert(dst.cap >= a.len + b.len);
    LimbT* re = detail::umul_dispatch(a.ptr, a.len, b.ptr, b.len, dst.ptr, numetron::detail::stack_allocator<LimbT>{});
    dst.len = static_cast<size_t>(re - dst.ptr);
    dst.sign = a.sign * b.sign;
    slot_trim(dst);
}

template <std::unsigned_integral LimbT>
inline void slot_add_shifted_to_result(LimbT* rb, size_t rsz, toom_slot<LimbT> const& c, size_t shift)
{
    if (!c.sign) return;
    assert(c.sign > 0);
    if (shift >= rsz) return;

    const size_t avail = rsz - shift;
    const size_t len = (std::min)(avail, c.len);
    if (!len) return;

    if (LimbT carry = uadd_inplace(rb + shift, c.ptr, c.ptr + len))
        uadd_limb(rb + shift + len, rb + rsz, carry);
}


enum class toom_op : unsigned char
{
    allocate,
    deallocate,
    slot_layout,
    load_part_u,
    load_part_v,
    clear,
    copy,
    add,
    sub,
    mul_small,
    divexact_small,
    mul_block,
    compose_shifted,
};

struct toom_instr
{
    toom_op op;
    unsigned short dst;
    unsigned short src0;
    unsigned short src1;
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
    mul_const,
    max2,
};

struct toom_size_expr
{
    toom_size_expr_op op;
    unsigned short a;
    unsigned short b;
};

struct toom_size_eval_context
{
    size_t un = 0;
    size_t vn = 0;
    size_t chunk = 0;
    size_t u_hi = 0;
    size_t v_hi = 0;
    size_t d_buf_n = 0;
};

template <size_t ExprCount>
inline size_t eval_size_expr(std::array<toom_size_expr, ExprCount> const& exprs, unsigned short id, toom_size_eval_context const& c)
{
    if (id >= ExprCount) {
        return static_cast<size_t>(id) * c.chunk;
    }

    auto const& e = exprs[id];
    switch (e.op) {
    case toom_size_expr_op::constant:
        return static_cast<size_t>(e.a);
    case toom_size_expr_op::variable:
        switch (static_cast<toom_size_var>(e.a)) {
        case toom_size_var::un: return c.un;
        case toom_size_var::vn: return c.vn;
        case toom_size_var::chunk: return c.chunk;
        case toom_size_var::u_hi: return c.u_hi;
        case toom_size_var::v_hi: return c.v_hi;
        case toom_size_var::d_buf_n: return c.d_buf_n;
        default: return 0;
        }
    case toom_size_expr_op::add:
        return eval_size_expr(exprs, e.a, c) + eval_size_expr(exprs, e.b, c);
    case toom_size_expr_op::mul_const:
        return eval_size_expr(exprs, e.a, c) * static_cast<size_t>(e.b);
    case toom_size_expr_op::max2:
        return (std::max)(eval_size_expr(exprs, e.a, c), eval_size_expr(exprs, e.b, c));
    default:
        return 0;
    }
}

namespace toom3_slots {

inline constexpr unsigned short A0 = 0;
inline constexpr unsigned short A1 = 1;
inline constexpr unsigned short A2 = 2;
inline constexpr unsigned short B0 = 3;
inline constexpr unsigned short B1 = 4;
inline constexpr unsigned short B2 = 5;

inline constexpr unsigned short EA0 = 6;
inline constexpr unsigned short EA1 = 7;
inline constexpr unsigned short EAM1 = 8;
inline constexpr unsigned short EA2 = 9;
inline constexpr unsigned short EAINF = 10;

inline constexpr unsigned short EB0 = 11;
inline constexpr unsigned short EB1 = 12;
inline constexpr unsigned short EBM1 = 13;
inline constexpr unsigned short EB2 = 14;
inline constexpr unsigned short EBINF = 15;

inline constexpr unsigned short W0 = 16;
inline constexpr unsigned short W1 = 17;
inline constexpr unsigned short WM1 = 18;
inline constexpr unsigned short W2 = 19;
inline constexpr unsigned short WINF = 20;

inline constexpr unsigned short R0 = 21;
inline constexpr unsigned short R1 = 22;
inline constexpr unsigned short R2 = 23;
inline constexpr unsigned short R3 = 24;
inline constexpr unsigned short R4 = 25;

inline constexpr unsigned short T0 = 26;
inline constexpr unsigned short T1 = 27;
inline constexpr unsigned short T2 = 28;
inline constexpr unsigned short T3 = 29;

inline constexpr unsigned short SLOT_COUNT = 30;

} // namespace toom3_slots

namespace toom2_slots {

inline constexpr unsigned short A0 = 0;
inline constexpr unsigned short A1 = 1;
inline constexpr unsigned short B0 = 2;
inline constexpr unsigned short B1 = 3;

inline constexpr unsigned short W0 = 4;
inline constexpr unsigned short W1 = 5;
inline constexpr unsigned short W2 = 6;

inline constexpr unsigned short R0 = 7;
inline constexpr unsigned short R1 = 8;
inline constexpr unsigned short R2 = 9;

inline constexpr unsigned short T0 = 10;
inline constexpr unsigned short T1 = 11;

inline constexpr unsigned short SLOT_COUNT = 12;

} // namespace toom2_slots

inline constexpr std::array<toom_size_expr, 2> toom2_size_exprs{
    toom_size_expr{ toom_size_expr_op::variable, static_cast<unsigned short>(toom_size_var::chunk), 0 },
    toom_size_expr{ toom_size_expr_op::mul_const, 0, 12 },
};

inline constexpr std::array<toom_size_expr, 2> toom3_size_exprs{
    toom_size_expr{ toom_size_expr_op::variable, static_cast<unsigned short>(toom_size_var::chunk), 0 },
    toom_size_expr{ toom_size_expr_op::mul_const, 0, 64 },
};

inline constexpr std::array toom2_plan{
    toom_instr{ toom_op::allocate, 0, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::A0, 0, 0, 0, 1, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::A1, 1, 0, 0, 1, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::B0, 2, 0, 0, 1, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::B1, 3, 0, 0, 1, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::W0, 4, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::W1, 6, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::W2, 8, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::R0, 4, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::R1, 10, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::R2, 6, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::T0, 10, 0, 0, 2, 0 },
    toom_instr{ toom_op::slot_layout, toom2_slots::T1, 10, 0, 0, 2, 0 },
    toom_instr{ toom_op::load_part_u, toom2_slots::A0, 0, 0, 0, 0, 0 },
    toom_instr{ toom_op::load_part_u, toom2_slots::A1, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::load_part_v, toom2_slots::B0, 0, 0, 0, 0, 0 },
    toom_instr{ toom_op::load_part_v, toom2_slots::B1, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::mul_block, toom2_slots::W0, toom2_slots::A0, toom2_slots::B0, 0 },
    toom_instr{ toom_op::mul_block, toom2_slots::W1, toom2_slots::A1, toom2_slots::B1, 0 },
    toom_instr{ toom_op::sub, toom2_slots::T0, toom2_slots::A0, toom2_slots::A1, 0 },
    toom_instr{ toom_op::sub, toom2_slots::T1, toom2_slots::B0, toom2_slots::B1, 0 },
    toom_instr{ toom_op::mul_block, toom2_slots::W2, toom2_slots::T0, toom2_slots::T1, 0 },

    toom_instr{ toom_op::copy, toom2_slots::R0, toom2_slots::W0, 0, 0 },
    toom_instr{ toom_op::copy, toom2_slots::R2, toom2_slots::W1, 0, 0 },
    toom_instr{ toom_op::add, toom2_slots::T0, toom2_slots::W0, toom2_slots::W1, 0 },
    toom_instr{ toom_op::sub, toom2_slots::R1, toom2_slots::T0, toom2_slots::W2, 0 },

    toom_instr{ toom_op::compose_shifted, 0, toom2_slots::R0, 0, 0 },
    toom_instr{ toom_op::compose_shifted, 0, toom2_slots::R1, 0, 1 },
    toom_instr{ toom_op::compose_shifted, 0, toom2_slots::R2, 0, 2 },
    toom_instr{ toom_op::deallocate, 0, 0, 0, 0, 0, 0 },
};

inline constexpr std::array toom3_plan{
    toom_instr{ toom_op::allocate, 0, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::load_part_u, toom3_slots::A0, 0, 0, 0, 0, 0 },
    toom_instr{ toom_op::load_part_u, toom3_slots::A1, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::load_part_u, toom3_slots::A2, 0, 0, 2, 0, 0 },
    toom_instr{ toom_op::load_part_v, toom3_slots::B0, 0, 0, 0, 0, 0 },
    toom_instr{ toom_op::load_part_v, toom3_slots::B1, 0, 0, 1, 0, 0 },
    toom_instr{ toom_op::load_part_v, toom3_slots::B2, 0, 0, 2, 0, 0 },
    // Evaluate A
    toom_instr{ toom_op::copy, toom3_slots::EA0,   toom3_slots::A0,    0, 0 },
    toom_instr{ toom_op::copy, toom3_slots::EAINF, toom3_slots::A2,    0, 0 },
    toom_instr{ toom_op::add,  toom3_slots::T0,    toom3_slots::A0,    toom3_slots::A1, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EA1,   toom3_slots::T0,    toom3_slots::A2, 0 },
    toom_instr{ toom_op::sub,  toom3_slots::T0,    toom3_slots::A0,    toom3_slots::A1, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EAM1,  toom3_slots::T0,    toom3_slots::A2, 0 },
    toom_instr{ toom_op::mul_small, toom3_slots::T0, toom3_slots::A1, 0, 2 },
    toom_instr{ toom_op::mul_small, toom3_slots::T1, toom3_slots::A2, 0, 4 },
    toom_instr{ toom_op::add,  toom3_slots::T0,    toom3_slots::A0,    toom3_slots::T0, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EA2,   toom3_slots::T0,    toom3_slots::T1, 0 },

    // Evaluate B
    toom_instr{ toom_op::copy, toom3_slots::EB0,   toom3_slots::B0,    0, 0 },
    toom_instr{ toom_op::copy, toom3_slots::EBINF, toom3_slots::B2,    0, 0 },
    toom_instr{ toom_op::add,  toom3_slots::T0,    toom3_slots::B0,    toom3_slots::B1, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EB1,   toom3_slots::T0,    toom3_slots::B2, 0 },
    toom_instr{ toom_op::sub,  toom3_slots::T0,    toom3_slots::B0,    toom3_slots::B1, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EBM1,  toom3_slots::T0,    toom3_slots::B2, 0 },
    toom_instr{ toom_op::mul_small, toom3_slots::T0, toom3_slots::B1, 0, 2 },
    toom_instr{ toom_op::mul_small, toom3_slots::T1, toom3_slots::B2, 0, 4 },
    toom_instr{ toom_op::add,  toom3_slots::T0,    toom3_slots::B0,    toom3_slots::T0, 0 },
    toom_instr{ toom_op::add,  toom3_slots::EB2,   toom3_slots::T0,    toom3_slots::T1, 0 },

    // Pointwise multiplications
    toom_instr{ toom_op::mul_block, toom3_slots::W0,   toom3_slots::EA0,   toom3_slots::EB0, 0 },
    toom_instr{ toom_op::mul_block, toom3_slots::W1,   toom3_slots::EA1,   toom3_slots::EB1, 0 },
    toom_instr{ toom_op::mul_block, toom3_slots::WM1,  toom3_slots::EAM1,  toom3_slots::EBM1, 0 },
    toom_instr{ toom_op::mul_block, toom3_slots::W2,   toom3_slots::EA2,   toom3_slots::EB2, 0 },
    toom_instr{ toom_op::mul_block, toom3_slots::WINF, toom3_slots::EAINF, toom3_slots::EBINF, 0 },

    // Interpolate
    toom_instr{ toom_op::copy, toom3_slots::R0, toom3_slots::W0,   0, 0 },
    toom_instr{ toom_op::copy, toom3_slots::R4, toom3_slots::WINF, 0, 0 },
    toom_instr{ toom_op::add,  toom3_slots::T0, toom3_slots::W1,   toom3_slots::WM1, 0 },
    toom_instr{ toom_op::divexact_small, toom3_slots::T1, toom3_slots::T0, 0, 2 },
    toom_instr{ toom_op::sub,  toom3_slots::T0, toom3_slots::W1,   toom3_slots::WM1, 0 },
    toom_instr{ toom_op::divexact_small, toom3_slots::T2, toom3_slots::T0, 0, 2 },
    toom_instr{ toom_op::sub,  toom3_slots::T0, toom3_slots::T1,   toom3_slots::R0, 0 },
    toom_instr{ toom_op::sub,  toom3_slots::R2, toom3_slots::T0,   toom3_slots::R4, 0 },
    toom_instr{ toom_op::mul_small, toom3_slots::T0, toom3_slots::R4, 0, 16 },
    toom_instr{ toom_op::sub,  toom3_slots::T3, toom3_slots::W2,   toom3_slots::R0, 0 },
    toom_instr{ toom_op::sub,  toom3_slots::T3, toom3_slots::T3,   toom3_slots::T0, 0 },
    toom_instr{ toom_op::mul_small, toom3_slots::T0, toom3_slots::R2, 0, 4 },
    toom_instr{ toom_op::sub,  toom3_slots::T3, toom3_slots::T3,   toom3_slots::T0, 0 },
    toom_instr{ toom_op::divexact_small, toom3_slots::T3, toom3_slots::T3, 0, 2 },
    toom_instr{ toom_op::sub,  toom3_slots::T0, toom3_slots::T3,   toom3_slots::T2, 0 },
    toom_instr{ toom_op::divexact_small, toom3_slots::R3, toom3_slots::T0, 0, 3 },
    toom_instr{ toom_op::sub,  toom3_slots::R1, toom3_slots::T2,   toom3_slots::R3, 0 },

    // Compose
    toom_instr{ toom_op::compose_shifted, 0, toom3_slots::R0, 0, 0 },
    toom_instr{ toom_op::compose_shifted, 0, toom3_slots::R1, 0, 1 },
    toom_instr{ toom_op::compose_shifted, 0, toom3_slots::R2, 0, 2 },
    toom_instr{ toom_op::compose_shifted, 0, toom3_slots::R3, 0, 3 },
    toom_instr{ toom_op::compose_shifted, 0, toom3_slots::R4, 0, 4 },
    toom_instr{ toom_op::deallocate, 0, 0, 0, 0, 0, 0 },
};




template <std::unsigned_integral LimbT>
struct stage_memory_state
{
    LimbT* slab = nullptr;
    size_t slab_len = 0;
    bool slab_owned = false;
    std::array<size_t, 64> slot_off{};
    std::array<size_t, 64> slot_cap{};
    std::array<toom_slot<LimbT>, 64> slots{};
};

template <size_t>
inline constexpr bool always_false_v = false;

template <size_t PlanSize>
consteval size_t required_slot_count(std::array<toom_instr, PlanSize> const& plan)
{
    unsigned short max_idx = 0;
    for (auto const& op : plan) {
        max_idx = (std::max)(max_idx, op.dst);
        max_idx = (std::max)(max_idx, op.src0);
        max_idx = (std::max)(max_idx, op.src1);
    }
    return static_cast<size_t>(max_idx) + 1;
}

template <size_t N, size_t M>
struct toom_stage_traits
{
    static_assert(always_false_v<N + M>, "Missing toom_stage_traits specialization for this (N,M)");
};

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan = toom2_plan;
    static constexpr auto const& size_exprs = toom2_size_exprs;
    static constexpr size_t slot_count = required_slot_count(plan);
};

template <>
struct toom_stage_traits<3, 3>
{
    static constexpr auto const& plan = toom3_plan;
    static constexpr auto const& size_exprs = toom3_size_exprs;
    static constexpr size_t slot_count = required_slot_count(plan);
};

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t I, typename ScratchAllocatorT>
inline void run_toom_op(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT& scratch_alloc)
{
    constexpr toom_instr op = toom_stage_traits<N, M>::plan[I];
    auto& dst = mem.slots[op.dst];
    auto const& s0 = mem.slots[op.src0];
    auto const& s1 = mem.slots[op.src1];

    if constexpr (op.op == toom_op::allocate) {
        const size_t alloc_sz = eval_size_expr(toom_stage_traits<N, M>::size_exprs, op.imm, size_ctx);
        if (mem.slab_owned) {
            std::allocator_traits<ScratchAllocatorT>::deallocate(scratch_alloc, mem.slab, mem.slab_len);
        }
        mem.slab = std::allocator_traits<ScratchAllocatorT>::allocate(scratch_alloc, alloc_sz);
        mem.slab_len = alloc_sz;
        mem.slab_owned = true;
    } else if constexpr (op.op == toom_op::slot_layout) {
        const size_t off = eval_size_expr(toom_stage_traits<N, M>::size_exprs, op.src0, size_ctx);
        const size_t cap = eval_size_expr(toom_stage_traits<N, M>::size_exprs, op.imm, size_ctx);
        assert(op.dst < mem.slot_off.size());
        mem.slot_off[op.dst] = off;
        mem.slot_cap[op.dst] = cap;
        if (mem.slab_owned) {
            assert(off + cap <= mem.slab_len);
        }
        mem.slots[op.dst].ptr = mem.slab + off;
        mem.slots[op.dst].cap = cap;
        mem.slots[op.dst].len = 0;
        mem.slots[op.dst].sign = 0;
    } else if constexpr (op.op == toom_op::deallocate) {
        if (mem.slab_owned) {
            std::allocator_traits<ScratchAllocatorT>::deallocate(scratch_alloc, mem.slab, mem.slab_len);
            mem.slab = nullptr;
            mem.slab_len = 0;
            mem.slab_owned = false;
        }
    } else if constexpr (op.op == toom_op::load_part_u) {
        const size_t idx = op.imm;
        const size_t start = idx * chunk;
        std::span<const LimbT> part;
        if (start < u.size()) {
            if (idx + 1 < N) {
                const size_t n = (std::min)(chunk, u.size() - start);
                part = { u.data() + start, n };
            } else {
                part = { u.data() + start, u.size() - start };
            }
        }
        slot_set_positive(dst, part);
    } else if constexpr (op.op == toom_op::load_part_v) {
        const size_t idx = op.imm;
        const size_t start = idx * chunk;
        std::span<const LimbT> part;
        if (start < v.size()) {
            if (idx + 1 < M) {
                const size_t n = (std::min)(chunk, v.size() - start);
                part = { v.data() + start, n };
            } else {
                part = { v.data() + start, v.size() - start };
            }
        }
        slot_set_positive(dst, part);
    } else if constexpr (op.op == toom_op::clear) {
        slot_clear(dst);
    } else if constexpr (op.op == toom_op::copy) {
        slot_copy(dst, s0);
    } else if constexpr (op.op == toom_op::add) {
        slot_add_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::sub) {
        slot_sub_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::mul_small) {
        slot_mul_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::divexact_small) {
        slot_divexact_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::mul_block) {
        slot_mul_dispatch(dst, s0, s1);
    } else if constexpr (op.op == toom_op::compose_shifted) {
        slot_add_shifted_to_result(rb, rsz, s0, static_cast<size_t>(op.imm) * chunk);
    } else {
        static_assert(op.op == toom_op::compose_shifted, "Unsupported toom op");
    }
}

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t... Is, typename ScratchAllocatorT>
inline void run_toom_stage(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    std::index_sequence<Is...>,
    ScratchAllocatorT& scratch_alloc)
{
    (run_toom_op<LimbT, N, M, Is>(mem, size_ctx, u, v, rb, rsz, chunk, scratch_alloc), ...);
}

template <std::unsigned_integral LimbT, size_t N, size_t M, typename ScratchAllocatorT>
inline void run_toom_stage(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT& scratch_alloc)
{
    run_toom_stage<LimbT, N, M>(mem, size_ctx, u, v, rb, rsz, chunk,
        std::make_index_sequence<toom_stage_traits<N, M>::plan.size()>{}, scratch_alloc);
}

template <std::unsigned_integral LimbT, size_t N, size_t M, typename ScratchAllocatorT>
inline void run_toom_stage(
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT scratch_alloc)
{
    const toom_size_eval_context size_ctx{
        u.size(),
        v.size(),
        chunk,
        u.size() - chunk,
        v.size() - chunk,
        (std::max)(chunk, u.size() - chunk),
    };

    stage_memory_state<LimbT> mem;
    NUMETRON_SCOPE_EXIT([&] {
        if (mem.slab_owned) {
            std::allocator_traits<ScratchAllocatorT>::deallocate(scratch_alloc, mem.slab, mem.slab_len);
            mem.slab = nullptr;
            mem.slab_len = 0;
            mem.slab_owned = false;
        }
    });
    run_toom_stage<LimbT, N, M>(mem, size_ctx, u, v, rb, rsz, chunk, scratch_alloc);
}

} // namespace toom_runtime_detail

template <size_t N, size_t M>
struct toom_engine
{
    using plan = toom_plan<N, M>;

    template <std::unsigned_integral LimbT, typename AllocatorT>
    requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
    static std::tuple<LimbT*, size_t, size_t>
    umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
    {
        const size_t un = u.size();
        const size_t vn = v.size();

        assert(un > 0 && vn > 0 && un >= vn);

        static_assert(detail::has_toom_plan<N, M>(),
            "numetron: toom_engine requested unsupported (N, M). Add row to toom_plan_table and executor path.");

        const size_t alloc_sz = un + vn;
        LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
        try {
            LimbT* re = rb;

            const size_t chunk = (vn + plan::split_rounding_bias) / plan::split_den;
            assert(chunk > 0);

            std::memset(rb, 0, alloc_sz * sizeof(LimbT));
            toom_runtime_detail::run_toom_stage<LimbT, N, M>(u, v, rb, alloc_sz, chunk, alloc);
            re = rb + alloc_sz;

            while (re != rb && *(re - 1) == 0) --re;
            return { rb, static_cast<size_t>(re - rb), alloc_sz };
        }
        catch (...) {
            std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
            throw;
        }
    }
};

} // namespace numetron::limb_arithmetic
