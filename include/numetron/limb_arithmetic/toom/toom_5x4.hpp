// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-5/4 (toom54, GMP's mpn_toom54_mul) splits u into five pieces and v into four of c limbs
// each, c = max(ceil(un/5), ceil(vn/4)), so that both top pieces are <= c. Two traits for the one
// plan, as for toom32 (toom_3x2.hpp); toom54_split_by_u() picks.
inline bool toom54_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 4) / 5 >= (vn + 3) / 4;
}

// Whether toom54 can take un x vn (un >= vn): both top pieces non-empty, s = un - 4c > 0 and
// t = vn - 3c > 0 (about 1 < un/vn < 5/3).
inline bool toom54_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 4) / 5, (vn + 3) / 4);
    return un > 4 * c && vn > 3 * c;
}

}

namespace toom_runtime_detail {

// Toom-5/4: u = a0 + a1 W + ... + a4 W^4, v = b0 + ... + b3 W^3, W = B^c, s = |a4| = un - 4c,
// t = |b3| = vn - 3c, 0 < s, t <= c. The product r(x) = c0 + ... + c7 x^7 has toom63's degree
// (toom_6x3.hpp), so it is evaluated at the same points, 0, +-1, +-2, +-4, inf, with c0 = a0 b0
// and c7 = a4 b3 straight into rb; eight products of about vn/4 limbs. From the point values on,
// the plan is toom63's (the halves, the quadratic at 1, 4, 16, the scratch and the result
// views), unchanged.
consteval auto make_toom54()
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
    constexpr std::array<toom_ref, 5> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3), slot_builder::u(4) };
    constexpr std::array<toom_ref, 4> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2), slot_builder::v(3) };

    std::array plan = {
        // Evaluate A (five pieces): even part into the minus slot, odd part into T, then both points.
        lincomb(EAM1, { lc_add(u[0]), lc_add(u[2]), lc_add(u[4]) }),
        lincomb(T,    { lc_add(u[1]), lc_add(u[3]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA1, .src0 = EAM1, .src1 = T, .dst2 = EAM1 },     // A(+-1)
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2), lc_add(u[4], 4) }),
        lincomb(T,    { lc_add(u[1], 1), lc_add(u[3], 3) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 },     // A(+-2)
        lincomb(EAM4, { lc_add(u[0]), lc_add(u[2], 4), lc_add(u[4], 8) }),
        lincomb(T,    { lc_add(u[1], 2), lc_add(u[3], 6) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA4, .src0 = EAM4, .src1 = T, .dst2 = EAM4 },     // A(+-4)

        // Evaluate B (four pieces) the same way.
        lincomb(EBM1, { lc_add(v[0]), lc_add(v[2]) }),
        lincomb(T,    { lc_add(v[1]), lc_add(v[3]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB1, .src0 = EBM1, .src1 = T, .dst2 = EBM1 },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2) }),
        lincomb(T,    { lc_add(v[1], 1), lc_add(v[3], 3) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = T, .dst2 = EBM2 },
        lincomb(EBM4, { lc_add(v[0]), lc_add(v[2], 4) }),
        lincomb(T,    { lc_add(v[1], 2), lc_add(v[3], 6) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB4, .src0 = EBM4, .src1 = T, .dst2 = EBM4 },

        // Pointwise products (signs of the minus points kept in the WM slots); c0 and c7 into rb.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2 },
        toom_instr{ toom_op::umul_fixed,  W4,   EA4,  EB4  },
        toom_instr{ toom_op::umul_fixed,  WM4,  EAM4, EBM4 },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C7,   u[4], v[3] },

        // toom63's interpolation (see toom_6x3.hpp for the steps).
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C7, 1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C7) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2), lc_sub(C7, 8) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C7, 7) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4), lc_sub(C7, 15) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0), lc_sub(C7, 14) }, 4),
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W2, 2), lc_sub(W2), lc_add(W1, 2) },
                     WM4, { lc_add(WM4), lc_sub(WM2, 2), lc_sub(WM2), lc_add(WM1, 2) }, 2, 45),     // z2: c6, c5
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W1), lc_sub(W4, 4), lc_add(W4) },
                     WM2, { lc_add(WM2), lc_sub(WM1), lc_sub(WM4, 4), lc_add(WM4) }, 0, 3),         // z1: c4, c3
        lincomb_dual(W1,  { lc_add(W1), lc_sub(W2), lc_sub(W4) },
                     WM1, { lc_add(WM1), lc_sub(WM2), lc_sub(WM4) }),                               // z0: c2, c1

        // Compose, as in toom63.
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

static constexpr auto toom54 = make_toom54();

// c = ceil(un/5)
struct toom54_u_traits
{
    static constexpr auto const& plan              = toom54.plan;
    static constexpr auto const& size_exprs        = toom54.exprs;
    static constexpr auto const& slot_layout       = toom54.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom54.slab_expr;
    static constexpr size_t N = 5;
    static constexpr size_t M = 4;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/4)
struct toom54_v_traits
{
    static constexpr auto const& plan              = toom54.plan;
    static constexpr auto const& size_exprs        = toom54.exprs;
    static constexpr auto const& slot_layout       = toom54.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom54.slab_expr;
    static constexpr size_t N = 5;
    static constexpr size_t M = 4;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
