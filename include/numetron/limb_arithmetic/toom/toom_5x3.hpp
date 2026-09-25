// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-5/3 (toom53, GMP's mpn_toom53_mul) splits u into five pieces and v into three of c limbs
// each, c = max(ceil(un/5), ceil(vn/3)), so that both top pieces are <= c. Two traits for the one
// plan, as for toom32 (toom_3x2.hpp); toom53_split_by_u() picks.
inline bool toom53_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 4) / 5 >= (vn + 2) / 3;
}

// Whether toom53 can take un x vn (un >= vn): both top pieces non-empty, s = un - 4c > 0 and
// t = vn - 2c > 0 (about 4/3 < un/vn < 5/2).
inline bool toom53_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 4) / 5, (vn + 2) / 3);
    return un > 4 * c && vn > 2 * c;
}

}

namespace toom_runtime_detail {

// Toom-5/3: u = a0 + a1 W + ... + a4 W^4, v = b0 + b1 W + b2 W^2, W = B^c, s = |a4| = un - 4c,
// t = |b2| = vn - 2c, 0 < s, t <= c. The product r(x) = c0 + ... + c6 x^6 has the degree of the
// balanced Toom-4's, so it is evaluated at the same points, 0, +-1, +-2, 1/2, inf, with 1/2
// scaled so that the product is again 64 r(1/2): 16 A(1/2) = 16 a0 + 8 a1 + 4 a2 + 2 a3 + a4
// and 4 B(1/2) = 4 b0 + 2 b1 + b2. c0 = a0 b0 and c6 = a4 b2 go straight into rb; seven
// products of about vn/3 limbs. From the point values on, the plan is the balanced Toom-4's
// (toom_4x4.hpp: the interpolation, the scratch and the result views), unchanged.
consteval auto make_toom53()
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

    auto off_w1  = b.mul(e, 12);
    auto off_wm1 = b.add(off_w1, p);
    auto off_w2  = b.add(off_wm1, p);
    auto off_wm2 = b.add(off_w2, p);
    auto off_wh  = b.add(off_wm2, p);
    auto slab    = b.add(off_wh, p);

    slot_builder sb;
    auto EA1  = sb.tmp(b.c(0),      e);
    auto EAM1 = sb.tmp(e,           e);
    auto EA2  = sb.tmp(b.mul(e, 2), e);
    auto EAM2 = sb.tmp(b.mul(e, 3), e);
    auto EAH  = sb.tmp(b.mul(e, 4), e);
    auto EB1  = sb.tmp(b.mul(e, 5), e);
    auto EBM1 = sb.tmp(b.mul(e, 6), e);
    auto EB2  = sb.tmp(b.mul(e, 7), e);
    auto EBM2 = sb.tmp(b.mul(e, 8), e);
    auto EBH  = sb.tmp(b.mul(e, 9), e);
    auto TA   = sb.tmp(b.mul(e, 10), e);
    auto TB   = sb.tmp(b.mul(e, 11), e);
    auto W1   = sb.tmp(off_w1,  p);
    auto WM1  = sb.tmp(off_wm1, p);
    auto W2   = sb.tmp(off_w2,  p);
    auto WM2  = sb.tmp(off_wm2, p);
    auto WH   = sb.tmp(off_wh,  p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));
    auto W2HI = sb.tmp(b.add(off_w2, two_c), b.c(2));

    auto C0   = sb.rb(b.c(0),       two_c);
    auto R2   = sb.rb(two_c,        two_c);
    auto R4   = sb.rb(b.mul(c, 4),  two_c);
    auto C6   = sb.rb(b.mul(c, 6),  st);
    auto R1   = sb.rb(c,            b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),  b.sub(rn, b.mul(c, 3)));
    auto R4X  = sb.rb(b.mul(c, 4),  b.sub(rn, b.mul(c, 4)));
    auto R5   = sb.rb(b.mul(c, 5),  b.sub(rn, b.mul(c, 5)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 5> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3), slot_builder::u(4) };
    constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };

    std::array plan = {
        // Evaluate A (five pieces) at +-1, +-2, 1/2: even part into the minus slot, odd part into TA.
        lincomb(EAM1, { lc_add(u[0]), lc_add(u[2]), lc_add(u[4]) }),
        lincomb(TA,   { lc_add(u[1]), lc_add(u[3]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA1, .src0 = EAM1, .src1 = TA, .dst2 = EAM1 },   // A(+-1)
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2), lc_add(u[4], 4) }),
        lincomb(TA,   { lc_add(u[1], 1), lc_add(u[3], 3) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = TA, .dst2 = EAM2 },   // A(+-2)
        lincomb(EAH,  { lc_add(u[0], 4), lc_add(u[1], 3), lc_add(u[2], 2), lc_add(u[3], 1) }),
        lincomb(EAH,  { lc_add(EAH), lc_add(u[4]) }),                                                  // 16 A(1/2)

        // Evaluate B (three pieces): B(+-x) = (b0 + b2 x^2) +- b1 x.
        toom_instr{ .op = toom_op::eval_pm1, .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = v[1] },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2) }),
        lincomb(TB,   { lc_add(v[1], 1) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = TB, .dst2 = EBM2 },   // B(+-2)
        lincomb(EBH,  { lc_add(v[0], 2), lc_add(v[1], 1), lc_add(v[2]) }),                            // 4 B(1/2)

        // Pointwise products; c0 and c6 straight into the result.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1     },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1    },   // sign of r(-1) kept in WM1
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2     },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2    },   // sign of r(-2) kept in WM2
        toom_instr{ toom_op::umul_fixed,  WH,   EAH,  EBH     },   // 64 r(1/2)
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0]    },
        toom_instr{ toom_op::umul_fixed,  C6,   u[4], v[2]    },

        // The balanced Toom-4's interpolation (see toom_4x4.hpp for the steps).
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),                              // D1
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C6) }),                // E1 = c2 + c4
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2) }, 2),                              // D2
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C6, 6) }, 2),       // E2 = c2 + 4 c4
        lincomb(W2,  { lc_add(W2), lc_sub(W1) }, 0, 3),                                   // c4
        toom_instr{ toom_op::usub,        W1,   W1,   W2      },                          // c2
        lincomb(WH,  { lc_add(WH), lc_sub(C0, 6), lc_sub(W1, 4), lc_sub(C6) }),           // 2 H + 4 c4
        lincomb(WH,  { lc_add(WH), lc_sub(W2, 2), lc_sub(WM1, 1) }, 1, 3),                // T1 = 5 c1 + c3
        lincomb(WM2, { lc_add(WM2), lc_sub(WM1) }, 0, 3),                                 // T2 = c3 + 5 c5
        lincomb(WM1, { lc_add(WM1), lc_add(WM1, 2), lc_sub(WH), lc_sub(WM2) }, 0, 3),     // c3
        lincomb(WH,  { lc_add(WH), lc_sub(WM1) }, 0, 5),                                  // c1
        lincomb(WM2, { lc_add(WM2), lc_sub(WM1) }, 0, 5),                                 // c5

        // Compose, as in the balanced plan.
        toom_instr{ toom_op::copy_low,    R2,   W1            },
        toom_instr{ toom_op::copy_low,    R4,   W2            },
        toom_instr{ toom_op::uadd_into,   R4X,  W1HI          },
        toom_instr{ toom_op::uadd_into,   C6,   W2HI          },
        toom_instr{ toom_op::uadd_into,   R1,   WH            },
        toom_instr{ toom_op::uadd_into,   R3,   WM1           },
        toom_instr{ toom_op::uadd_into,   R5,   WM2           },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom53 = make_toom53();

// c = ceil(un/5)
struct toom53_u_traits
{
    static constexpr auto const& plan              = toom53.plan;
    static constexpr auto const& size_exprs        = toom53.exprs;
    static constexpr auto const& slot_layout       = toom53.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom53.slab_expr;
    static constexpr size_t N = 5;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/3)
struct toom53_v_traits
{
    static constexpr auto const& plan              = toom53.plan;
    static constexpr auto const& size_exprs        = toom53.exprs;
    static constexpr auto const& slot_layout       = toom53.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom53.slab_expr;
    static constexpr size_t N = 5;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
