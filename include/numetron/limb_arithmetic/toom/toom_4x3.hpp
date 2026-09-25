// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-4/3 (toom43, GMP's mpn_toom43_mul) splits u into four pieces and v into three of c limbs
// each, c = max(ceil(un/4), ceil(vn/3)), so that both top pieces are <= c. Two traits for the one
// plan, as for toom32 (toom_3x2.hpp); toom43_split_by_u() picks.
inline bool toom43_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 3) / 4 >= (vn + 2) / 3;
}

// Whether toom43 can take un x vn (un >= vn): both top pieces non-empty, s = un - 3c > 0 and
// t = vn - 2c > 0 (about 1 < un/vn < 2).
inline bool toom43_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 3) / 4, (vn + 2) / 3);
    return un > 3 * c && vn > 2 * c;
}

}

namespace toom_runtime_detail {

// Toom-4/3: u = a0 + a1 W + a2 W^2 + a3 W^3, v = b0 + b1 W + b2 W^2, W = B^c, s = |a3| = un - 3c,
// t = |b2| = vn - 2c, 0 < s, t <= c. The product r(x) = c0 + ... + c5 x^5 is evaluated at 0, +-1,
// +-2, inf (GMP's points): c0 = a0 b0 and c5 = a3 b2 straight into rb, and r(+-1), r(+-2) (the
// minus ones as magnitude and sign) -- six products of about vn/3 limbs.
//
// Interpolation. Each pair splits into its halves with c0 and c5 taken out:
//   Q1 = (r(1) - r(-1) - 2 c5) / 2        = c1 + c3      E1 = r(1) - Q1 - c0 - c5              = c2 + c4
//   Q2 = (r(2) - r(-2) - 2^6 c5) / 4      = c1 + 4 c3    E2 = (r(2) - 2 Q2 - c0 - 2^5 c5) / 4  = c2 + 4 c4
// then, both halves at once (one lincomb_dual each):
//   c4 = (E2 - E1) / 3,  c3 = (Q2 - Q1) / 3;   c2 = E1 - c4,  c1 = Q1 - c3
//
// Scratch (9e + 4p limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EAM2, EB1 EBM1 EB2 EBM2   8 x e   A and B at +-1, +-2
//   T                                      e       the odd part while evaluating
//   W1 WM1 W2 WM2                          4 x p   r(1), |r(-1)|, r(2), |r(-2)|, ending as
//                                                  c2, c1, c4, c3
//   W1HI  aliases W1[2c, 2c + 2)     c2's limbs above 2c
//   W2HI  aliases W2[c, 2c + 2)      c4's limbs above rb's c-limb gap at 4c, added onto c5
// Result views: C0 = rb[0, 2c), R2 = rb[2c, 4c), R4 = rb[4c, 5c), C5 = rb[5c, end); they cover
// rb. R1, R3, R4X run from c, 3c, 4c to the end.
consteval auto make_toom43()
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

    auto off_w = b.mul(e, 9);
    auto slab  = b.add(off_w, b.mul(p, 4));

    // (No helper lambdas here: GCC doesn't take consteval calls made inside them.)
    slot_builder sb;
    auto EA1  = sb.tmp(b.c(0),      e);
    auto EAM1 = sb.tmp(e,           e);
    auto EA2  = sb.tmp(b.mul(e, 2), e);
    auto EAM2 = sb.tmp(b.mul(e, 3), e);
    auto EB1  = sb.tmp(b.mul(e, 4), e);
    auto EBM1 = sb.tmp(b.mul(e, 5), e);
    auto EB2  = sb.tmp(b.mul(e, 6), e);
    auto EBM2 = sb.tmp(b.mul(e, 7), e);
    auto T    = sb.tmp(b.mul(e, 8), e);

    auto off_w1  = off_w;
    auto off_wm1 = b.add(off_w, p);
    auto off_w2  = b.add(off_w, b.mul(p, 2));
    auto off_wm2 = b.add(off_w, b.mul(p, 3));
    auto W1   = sb.tmp(off_w1,  p);
    auto WM1  = sb.tmp(off_wm1, p);
    auto W2   = sb.tmp(off_w2,  p);
    auto WM2  = sb.tmp(off_wm2, p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));
    auto W2HI = sb.tmp(b.add(off_w2, c),     b.add(c, b.c(2)));

    auto C0   = sb.rb(b.c(0),       two_c);
    auto R2   = sb.rb(two_c,        two_c);
    auto R4   = sb.rb(b.mul(c, 4),  c);
    auto C5   = sb.rb(b.mul(c, 5),  st);
    auto R1   = sb.rb(c,            b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),  b.sub(rn, b.mul(c, 3)));
    auto R4X  = sb.rb(b.mul(c, 4),  b.sub(rn, b.mul(c, 4)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 4> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3) };
    constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };

    std::array plan = {
        // Evaluate A (four pieces): A(+-1) = (a0 + a2) +- (a1 + a3), A(+-2) = (a0 + 4 a2) +- 2 (a1 + 4 a3).
        toom_instr{ toom_op::uadd,        T,    u[1], u[3]    },
        toom_instr{ .op = toom_op::eval_pm1, .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = T },
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2) }),
        lincomb(T,    { lc_add(u[1], 1), lc_add(u[3], 3) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 },

        // Evaluate B (three pieces): B(+-x) = (b0 + b2 x^2) +- b1 x.
        toom_instr{ .op = toom_op::eval_pm1, .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = v[1] },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2) }),
        lincomb(T,    { lc_add(v[1], 1) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = T, .dst2 = EBM2 },

        // Pointwise products (signs of the minus points kept in the WM slots); c0 and c5 into rb.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2 },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C5,   u[3], v[2] },

        // Odd / even halves, c0 and c5 taken out: WM1 = Q1, W1 = E1, WM2 = Q2, W2 = E2.
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C5, 1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C5) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2), lc_sub(C5, 6) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C5, 5) }, 2),

        // Both halves at once.
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W1) },
                     WM2, { lc_add(WM2), lc_sub(WM1) }, 0, 3),      // c4, c3
        lincomb_dual(W1,  { lc_add(W1), lc_sub(W2) },
                     WM1, { lc_add(WM1), lc_sub(WM2) }),            // c2, c1

        // Compose: c2 and c4 fill rb's pieces (c4's limbs above its c-limb gap onto c5), then
        // c1 and c3 are added.
        toom_instr{ toom_op::copy_low,  R2,   W1   },
        toom_instr{ toom_op::copy_low,  R4,   W2   },
        toom_instr{ toom_op::uadd_into, C5,   W2HI },
        toom_instr{ toom_op::uadd_into, R4X,  W1HI },
        toom_instr{ toom_op::uadd_into, R1,   WM1  },
        toom_instr{ toom_op::uadd_into, R3,   WM2  },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom43 = make_toom43();

// c = ceil(un/4)
struct toom43_u_traits
{
    static constexpr auto const& plan              = toom43.plan;
    static constexpr auto const& size_exprs        = toom43.exprs;
    static constexpr auto const& slot_layout       = toom43.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom43.slab_expr;
    static constexpr size_t N = 4;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/3)
struct toom43_v_traits
{
    static constexpr auto const& plan              = toom43.plan;
    static constexpr auto const& size_exprs        = toom43.exprs;
    static constexpr auto const& slot_layout       = toom43.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom43.slab_expr;
    static constexpr size_t N = 4;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
