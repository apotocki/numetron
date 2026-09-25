// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-6/3 (toom63) splits u into six pieces and v into three of c limbs each,
// c = max(ceil(un/6), ceil(vn/3)) as in GMP's mpn_toom63_mul, so that both top pieces are <= c.
// Two traits for the one plan, as for toom32 (toom_3x2.hpp); toom63_split_by_u() picks.
inline bool toom63_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 5) / 6 >= (vn + 2) / 3;
}

// Whether toom63 can take un x vn (un >= vn): both top pieces non-empty, s = un - 5c > 0 and
// t = vn - 2c > 0.
inline bool toom63_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 5) / 6, (vn + 2) / 3);
    return un > 5 * c && vn > 2 * c;
}

}

namespace toom_runtime_detail {

// Toom-6/3: u = a0 + a1 W + ... + a5 W^5, v = b0 + b1 W + b2 W^2, W = B^c, s = |a5| = un - 5c,
// t = |b2| = vn - 2c, 0 < s, t <= c. The product r(x) = c0 + ... + c7 x^7 is evaluated at
// 0, +-1, +-2, +-4, inf (GMP's points): A(+-x) and B(+-x) from their even and odd parts,
// c0 = a0 b0 and c7 = a5 b2 straight into rb, and r(+-1), r(+-2), r(+-4) (the minus ones as
// magnitude and sign) -- eight products of about vn/3 limbs.
//
// Interpolation. Every pair splits into its halves with c0 and c7 taken out, so that with
// P(y) = c2 + c4 y + c6 y^2 and Q(y) = c1 + c3 y + c5 y^2:
//   Q(1)  = (r(1) - r(-1) - 2 c7) / 2              P(1)  = r(1) - Q(1) - c0 - c7
//   Q(4)  = (r(2) - r(-2) - 2^8 c7) / 4            P(4)  = (r(2) - 2 Q(4) - c0 - 2^7 c7) / 4
//   Q(16) = (r(4) - r(-4) - 2^15 c7) / 8           P(16) = (r(4) - 4 Q(16) - c0 - 2^14 c7) / 16
// and both halves are the same problem, a quadratic known at 1, 4, 16 -- solved for both at once,
// one lincomb_dual per step (all results non-negative; 180 = 4 * 45, 45 = 3 * 15 two divisions
// by divisors of B - 1):
//   z2 = (P(16) - 5 P(4) + 4 P(1)) / 180     (c6, c5)
//   z1 = (P(4) - P(1) - 15 z2) / 3            (c4, c3)
//   z0 = P(1) - z1 - z2                       (c2, c1)
//
// Scratch (13e + 6p limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EAM2 EA4 EAM4   6 x e    A at +-1, +-2, +-4
//   EB1 EBM1 EB2 EBM2 EB4 EBM4   6 x e    B at +-1, +-2, +-4
//   T                            e        the odd part while evaluating
//   W1 WM1 W2 WM2 W4 WM4         6 x p    the products, then P / Q values, ending as
//                                         c2 c1 c4 c3 c6 c5
//   W1HI W2HI  alias W1, W2 [2c, 2c + 2)  c2's, c4's limbs above 2c
//   W4HI       aliases W4[c, 2c + 2)      c6's limbs above rb's c-limb gap at 6c, added onto c7
// Result views: C0 = rb[0, 2c), R2 = rb[2c, 4c), R4 = rb[4c, 6c), R6 = rb[6c, 7c),
// C7 = rb[7c, end); they cover rb. R1, R3, R5, R4X, R6X run from c, 3c, 5c, 4c, 6c to the end.
consteval auto make_toom63()
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

    auto off_w = b.mul(e, 13);
    auto slab  = b.add(off_w, b.mul(p, 6));

    // (No helper lambdas here: GCC doesn't take consteval calls made inside them.)
    slot_builder sb;
    auto EA1  = sb.tmp(b.c(0),       e);
    auto EAM1 = sb.tmp(e,            e);
    auto EA2  = sb.tmp(b.mul(e, 2),  e);
    auto EAM2 = sb.tmp(b.mul(e, 3),  e);
    auto EA4  = sb.tmp(b.mul(e, 4),  e);
    auto EAM4 = sb.tmp(b.mul(e, 5),  e);
    auto EB1  = sb.tmp(b.mul(e, 6),  e);
    auto EBM1 = sb.tmp(b.mul(e, 7),  e);
    auto EB2  = sb.tmp(b.mul(e, 8),  e);
    auto EBM2 = sb.tmp(b.mul(e, 9),  e);
    auto EB4  = sb.tmp(b.mul(e, 10), e);
    auto EBM4 = sb.tmp(b.mul(e, 11), e);
    auto T    = sb.tmp(b.mul(e, 12), e);

    auto off_w1  = off_w;
    auto off_wm1 = b.add(off_w, p);
    auto off_w2  = b.add(off_w, b.mul(p, 2));
    auto off_wm2 = b.add(off_w, b.mul(p, 3));
    auto off_w4  = b.add(off_w, b.mul(p, 4));
    auto off_wm4 = b.add(off_w, b.mul(p, 5));
    auto W1   = sb.tmp(off_w1,  p);
    auto WM1  = sb.tmp(off_wm1, p);
    auto W2   = sb.tmp(off_w2,  p);
    auto WM2  = sb.tmp(off_wm2, p);
    auto W4   = sb.tmp(off_w4,  p);
    auto WM4  = sb.tmp(off_wm4, p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));
    auto W2HI = sb.tmp(b.add(off_w2, two_c), b.c(2));
    auto W4HI = sb.tmp(b.add(off_w4, c),     b.add(c, b.c(2)));

    auto C0   = sb.rb(b.c(0),       two_c);
    auto R2   = sb.rb(two_c,        two_c);
    auto R4   = sb.rb(b.mul(c, 4),  two_c);
    auto R6   = sb.rb(b.mul(c, 6),  c);
    auto C7   = sb.rb(b.mul(c, 7),  st);
    auto R1   = sb.rb(c,            b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),  b.sub(rn, b.mul(c, 3)));
    auto R5   = sb.rb(b.mul(c, 5),  b.sub(rn, b.mul(c, 5)));
    auto R4X  = sb.rb(b.mul(c, 4),  b.sub(rn, b.mul(c, 4)));
    auto R6X  = sb.rb(b.mul(c, 6),  b.sub(rn, b.mul(c, 6)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 6> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3), slot_builder::u(4), slot_builder::u(5) };
    constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };

    std::array plan = {
        // Evaluate A: even part into the minus slot, odd part into T, then both points.
        lincomb(EAM1, { lc_add(u[0]), lc_add(u[2]), lc_add(u[4]) }),
        lincomb(T,    { lc_add(u[1]), lc_add(u[3]), lc_add(u[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA1, .src0 = EAM1, .src1 = T, .dst2 = EAM1 },     // A(+-1)
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2), lc_add(u[4], 4) }),
        lincomb(T,    { lc_add(u[1], 1), lc_add(u[3], 3), lc_add(u[5], 5) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 },     // A(+-2)
        lincomb(EAM4, { lc_add(u[0]), lc_add(u[2], 4), lc_add(u[4], 8) }),
        lincomb(T,    { lc_add(u[1], 2), lc_add(u[3], 6), lc_add(u[5], 10) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA4, .src0 = EAM4, .src1 = T, .dst2 = EAM4 },     // A(+-4)

        // Evaluate B (three pieces): B(+-x) = (b0 + b2 x^2) +- b1 x.
        toom_instr{ .op = toom_op::eval_pm1, .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = v[1] },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2) }),
        lincomb(T,    { lc_add(v[1], 1) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = T, .dst2 = EBM2 },
        lincomb(EBM4, { lc_add(v[0]), lc_add(v[2], 4) }),
        lincomb(T,    { lc_add(v[1], 2) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB4, .src0 = EBM4, .src1 = T, .dst2 = EBM4 },

        // Pointwise products (signs of the minus points kept in the WM slots); c0 and c7 into rb.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2 },
        toom_instr{ toom_op::umul_fixed,  W4,   EA4,  EB4  },
        toom_instr{ toom_op::umul_fixed,  WM4,  EAM4, EBM4 },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C7,   u[5], v[2] },

        // Odd / even halves, c0 and c7 taken out: W* become P(1), P(4), P(16), WM* Q(1), Q(4), Q(16).
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C7, 1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C7) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2), lc_sub(C7, 8) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C7, 7) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4), lc_sub(C7, 15) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0), lc_sub(C7, 14) }, 4),

        // Both halves at once: the quadratic known at 1, 4, 16.
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W2, 2), lc_sub(W2), lc_add(W1, 2) },
                     WM4, { lc_add(WM4), lc_sub(WM2, 2), lc_sub(WM2), lc_add(WM1, 2) }, 2, 45),     // z2: c6, c5
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W1), lc_sub(W4, 4), lc_add(W4) },
                     WM2, { lc_add(WM2), lc_sub(WM1), lc_sub(WM4, 4), lc_add(WM4) }, 0, 3),         // z1: c4, c3
        lincomb_dual(W1,  { lc_add(W1), lc_sub(W2), lc_sub(W4) },
                     WM1, { lc_add(WM1), lc_sub(WM2), lc_sub(WM4) }),                               // z0: c2, c1

        // Compose: the even coefficients fill rb's pieces (c6's limbs above its c-limb gap onto
        // c7), then the odd ones are added.
        toom_instr{ toom_op::copy_low,  R2,   W1   },
        toom_instr{ toom_op::copy_low,  R4,   W2   },
        toom_instr{ toom_op::copy_low,  R6,   W4   },
        toom_instr{ toom_op::uadd_into, C7,   W4HI },
        toom_instr{ toom_op::uadd_into, R4X,  W1HI },
        toom_instr{ toom_op::uadd_into, R6X,  W2HI },
        toom_instr{ toom_op::uadd_into, R1,   WM1  },
        toom_instr{ toom_op::uadd_into, R3,   WM2  },
        toom_instr{ toom_op::uadd_into, R5,   WM4  },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom63 = make_toom63();

// c = ceil(un/6)
struct toom63_u_traits
{
    static constexpr auto const& plan              = toom63.plan;
    static constexpr auto const& size_exprs        = toom63.exprs;
    static constexpr auto const& slot_layout       = toom63.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom63.slab_expr;
    static constexpr size_t N = 6;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/3)
struct toom63_v_traits
{
    static constexpr auto const& plan              = toom63.plan;
    static constexpr auto const& size_exprs        = toom63.exprs;
    static constexpr auto const& slot_layout       = toom63.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom63.slab_expr;
    static constexpr size_t N = 6;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
