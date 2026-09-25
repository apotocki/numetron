// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-3/2 (toom32) splits u into three pieces and v into two of c limbs each, with
// c = max(ceil(un/3), ceil(vn/2)) as in GMP's mpn_toom32_mul, so that both top pieces are <= c.
// The engine derives c from one operand only, so there are two traits for the same plan:
// toom32_split_by_u() tells which of the two gives this c.
inline bool toom32_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 2) / 3 >= (vn + 1) / 2;
}

// Whether toom32 can take un x vn (un >= vn): both top pieces non-empty, s = un - 2c > 0 and
// t = vn - c > 0.
inline bool toom32_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 2) / 3, (vn + 1) / 2);
    return un > 2 * c && vn > c;
}

}

namespace toom_runtime_detail {

// Toom-3/2: u = a0 + a1 W + a2 W^2, v = b0 + b1 W, W = B^c, s = |a2| = un - 2c, t = |b1| = vn - c,
// 0 < s, t <= c. The product r(x) = c0 + c1 x + c2 x^2 + c3 x^3 is evaluated at 0, 1, -1, inf:
//   A(+-1) = (a0 + a2) +- a1       eval_pm1
//   B(+-1) = b0 +- b1              addsub_abs
// so the four point values are c0 = a0 b0 and c3 = a2 b1 (both straight into rb), r(1) and
// |r(-1)| -- four products of about vn/2 limbs, against Karatsuba's three of about 3vn/4.
// Interpolation, one lincomb pass each (results non-negative, the shift exact):
//   c1 = (r(1) - r(-1) - 2 c3) / 2
//   c2 = r(1) - c0 - c1 - c3
//
// Scratch (8c + 8 limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EB1 EBM1   4 x e     A(1), |A(-1)|, B(1), |B(-1)|
//   W1 WM1              2 x p     r(1), |r(-1)|, then c2, c1
//   W1HI  aliases W1[c, 2c+2)     the part of c2 above rb's c-limb gap, added onto c3
// Result views: C0 = rb[0, 2c), R2 = rb[2c, 3c), C3 = rb[3c, end), R1 = rb[c, end). rb is
// written in full by the plan (C0, R2, C3 cover it), so it doesn't need zeroing first.
consteval auto make_toom32()
{
    expr_builder b;

    auto c   = b.v(toom_size_var::chunk);
    auto s   = b.v(toom_size_var::u_hi);
    auto t   = b.v(toom_size_var::v_hi);
    auto rn  = b.add(b.v(toom_size_var::un), b.v(toom_size_var::vn));

    auto e   = b.add(c, b.c(1));
    auto p   = b.add(b.mul(c, 2), b.c(2));
    auto st  = b.add(s, t);
    auto two_c = b.mul(c, 2);

    auto off_eam1 = e;
    auto off_eb1  = b.mul(e, 2);
    auto off_ebm1 = b.mul(e, 3);
    auto off_w1   = b.mul(e, 4);
    auto off_wm1  = b.add(off_w1, p);
    auto slab     = b.add(off_wm1, p);

    slot_builder sb;
    auto EA1   = sb.tmp(b.c(0),   e);
    auto EAM1  = sb.tmp(off_eam1, e);
    auto EB1   = sb.tmp(off_eb1,  e);
    auto EBM1  = sb.tmp(off_ebm1, e);
    auto W1    = sb.tmp(off_w1,   p);
    auto WM1   = sb.tmp(off_wm1,  p);
    auto W1HI  = sb.tmp(b.add(off_w1, c), b.add(c, b.c(2)));

    auto C0    = sb.rb(b.c(0),      two_c);
    auto R2    = sb.rb(two_c,       c);
    auto C3    = sb.rb(b.mul(c, 3), st);
    auto R1    = sb.rb(c,           b.sub(rn, c));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 3> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2) };
    constexpr std::array<toom_ref, 2> v = { slot_builder::v(0), slot_builder::v(1) };

    std::array plan = {
        // Evaluate at +-1.
        toom_instr{ .op = toom_op::eval_pm1,   .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = u[1] },
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB1, .src0 = v[0], .src1 = v[1], .dst2 = EBM1 },

        // Pointwise products; c0 and c3 straight into the result.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },   // sign of r(-1) kept in WM1
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C3,   u[2], v[1] },

        // Interpolation.
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C3, 1) }, 1),              // c1
        lincomb(W1,  { lc_add(W1), lc_sub(C0), lc_sub(WM1), lc_sub(C3) }),               // c2

        // Compose: c2's low c limbs fill the gap rb[2c, 3c), the rest goes onto c3, then c1.
        toom_instr{ toom_op::copy_low,    R2,   W1         },
        toom_instr{ toom_op::uadd_into,   C3,   W1HI       },
        toom_instr{ toom_op::uadd_into,   R1,   WM1        },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom32 = make_toom32();

// c = ceil(un/3) (the larger one when un >= 1.5 vn or so)
struct toom32_u_traits
{
    static constexpr auto const& plan              = toom32.plan;
    static constexpr auto const& size_exprs        = toom32.exprs;
    static constexpr auto const& slot_layout       = toom32.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom32.slab_expr;
    static constexpr size_t N = 3;
    static constexpr size_t M = 2;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/2) (below that)
struct toom32_v_traits
{
    static constexpr auto const& plan              = toom32.plan;
    static constexpr auto const& size_exprs        = toom32.exprs;
    static constexpr auto const& slot_layout       = toom32.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom32.slab_expr;
    static constexpr size_t N = 3;
    static constexpr size_t M = 2;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
