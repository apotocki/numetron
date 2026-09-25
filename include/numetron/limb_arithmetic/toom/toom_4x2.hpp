// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-4/2 (toom42) splits u into four pieces and v into two of c limbs each, with
// c = max(ceil(un/4), ceil(vn/2)) as in GMP's mpn_toom42_mul, so that both top pieces are <= c.
// As with toom32 (toom_3x2.hpp), the engine derives c from one operand, so there are two traits
// for the one plan and toom42_split_by_u() picks between them.
inline bool toom42_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 3) / 4 >= (vn + 1) / 2;
}

// Whether toom42 can take un x vn (un >= vn): both top pieces non-empty, s = un - 3c > 0 and
// t = vn - c > 0.
inline bool toom42_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 3) / 4, (vn + 1) / 2);
    return un > 3 * c && vn > c;
}

}

namespace toom_runtime_detail {

// Toom-4/2: u = a0 + a1 W + a2 W^2 + a3 W^3, v = b0 + b1 W, W = B^c, s = |a3| = un - 3c,
// t = |b1| = vn - c, 0 < s, t <= c. The product r(x) = c0 + ... + c4 x^4 is evaluated at
// 0, 1, -1, 2, inf:
//   A(+-1) = (a0 + a2) +- (a1 + a3)      eval_pm1 over a ready odd sum
//   A(2)   = a0 + 2 a1 + 4 a2 + 8 a3     one lincomb
//   B(+-1) = b0 +- b1                    addsub_abs
//   B(2)   = b0 + 2 b1                   one lincomb
// so the five point values are c0 = a0 b0 and c4 = a3 b1 (both straight into rb), r(1), |r(-1)|
// and r(2) -- five products of about vn/2 limbs. A degree-4 product at the same points as the
// balanced Toom-3 (toom_3x3.hpp), so the interpolation is the same four lincomb passes and one
// subtraction, and so is the result layout.
//
// Scratch (12c + 12 limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EB1 EBM1 EB2   6 x e     A(1), |A(-1)|, A(2) and the same for B
//   W1 WM1 W2                   3 x p     r(1), |r(-1)|, r(2), then the interpolated c2, c1, c3
//   T     aliases W1[0, e)                a1 + a3, before W1 is written
//   W1HI  aliases W1[2c, 2c+2)            the part of c2 above rb's 2c-limb gap
// Result views: C0 = rb[0, 2c), R2 = rb[2c, 4c), C4 = rb[4c, end), R1 = rb[c, end),
// R3 = rb[3c, end). rb is written in full by the plan (C0, R2, C4 cover it).
consteval auto make_toom42()
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
    auto off_ea2  = b.mul(e, 2);
    auto off_eb1  = b.mul(e, 3);
    auto off_ebm1 = b.mul(e, 4);
    auto off_eb2  = b.mul(e, 5);
    auto off_w1   = b.mul(e, 6);
    auto off_wm1  = b.add(off_w1, p);
    auto off_w2   = b.add(off_wm1, p);
    auto slab     = b.add(off_w2, p);

    slot_builder sb;
    auto EA1   = sb.tmp(b.c(0),   e);
    auto EAM1  = sb.tmp(off_eam1, e);
    auto EA2   = sb.tmp(off_ea2,  e);
    auto EB1   = sb.tmp(off_eb1,  e);
    auto EBM1  = sb.tmp(off_ebm1, e);
    auto EB2   = sb.tmp(off_eb2,  e);
    auto W1    = sb.tmp(off_w1,   p);
    auto WM1   = sb.tmp(off_wm1,  p);
    auto W2    = sb.tmp(off_w2,   p);
    auto T     = sb.tmp(off_w1,   e);
    auto W1HI  = sb.tmp(b.add(off_w1, two_c), b.c(2));

    auto C0    = sb.rb(b.c(0),          two_c);
    auto R2    = sb.rb(two_c,           two_c);
    auto C4    = sb.rb(b.mul(c, 4),     st);
    auto R1    = sb.rb(c,               b.sub(rn, c));
    auto R3    = sb.rb(b.mul(c, 3),     b.sub(rn, b.mul(c, 3)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 4> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3) };
    constexpr std::array<toom_ref, 2> v = { slot_builder::v(0), slot_builder::v(1) };

    std::array plan = {
        // Evaluate A at +-1 (over the odd sum a1 + a3) and at 2.
        toom_instr{ toom_op::uadd, T, u[1], u[3] },
        toom_instr{ .op = toom_op::eval_pm1, .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = T },
        lincomb(EA2, { lc_add(u[0]), lc_add(u[1], 1), lc_add(u[2], 2), lc_add(u[3], 3) }),

        // Evaluate B at +-1 and at 2.
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB1, .src0 = v[0], .src1 = v[1], .dst2 = EBM1 },
        lincomb(EB2, { lc_add(v[0]), lc_add(v[1], 1) }),

        // Pointwise products; c0 and c4 straight into the result.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },   // sign of r(-1) kept in WM1
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C4,   u[3], v[1] },

        // Interpolation, as in the balanced Toom-3.
        lincomb(W2,  { lc_add(W2), lc_sub_signed(WM1) }, 0, 3),                            // (r(2) - r(-1)) / 3 = c1 + c2 + 3c3 + 5c4
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),                               // (r(1) - r(-1)) / 2 = c1 + c3
        lincomb(W2,  { lc_add(W2), lc_sub(W1), lc_add(C0), lc_sub(C4, 2) }, 1),            // c3
        lincomb(W1,  { lc_add(W1), lc_sub(C0), lc_sub(WM1), lc_sub(C4) }),                 // c2
        toom_instr{ toom_op::usub,        WM1,  WM1,  W2   },                              // c1

        // Compose: c2 fills the gap rb[2c, 4c) (its top on c4), then c1 and c3 are added.
        toom_instr{ toom_op::copy_low,    R2,   W1         },
        toom_instr{ toom_op::uadd_into,   C4,   W1HI       },
        toom_instr{ toom_op::uadd_into,   R1,   WM1        },
        toom_instr{ toom_op::uadd_into,   R3,   W2         },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom42 = make_toom42();

// c = ceil(un/4)
struct toom42_u_traits
{
    static constexpr auto const& plan              = toom42.plan;
    static constexpr auto const& size_exprs        = toom42.exprs;
    static constexpr auto const& slot_layout       = toom42.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom42.slab_expr;
    static constexpr size_t N = 4;
    static constexpr size_t M = 2;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/2)
struct toom42_v_traits
{
    static constexpr auto const& plan              = toom42.plan;
    static constexpr auto const& size_exprs        = toom42.exprs;
    static constexpr auto const& slot_layout       = toom42.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom42.slab_expr;
    static constexpr size_t N = 4;
    static constexpr size_t M = 2;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
