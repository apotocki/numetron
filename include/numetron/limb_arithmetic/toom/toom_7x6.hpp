// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-6.5 7 x 6 (toom76, the "half" case of GMP's toom6h) splits u into seven pieces and v into
// six of c limbs each, c = max(ceil(un/7), ceil(vn/6)), so that both top pieces are <= c. As with
// toom32 (toom_3x2.hpp) there are two traits for the one plan; toom76_split_by_u() picks.
inline bool toom76_split_by_u(size_t un, size_t vn) noexcept
{
    return (un + 6) / 7 >= (vn + 5) / 6;
}

// Whether toom76 can take un x vn (un >= vn): both top pieces non-empty, s = un - 6c > 0 and
// t = vn - 5c > 0.
inline bool toom76_split_fits(size_t un, size_t vn) noexcept
{
    const size_t c = (std::max)((un + 6) / 7, (vn + 5) / 6);
    return un > 6 * c && vn > 5 * c;
}

}

namespace toom_runtime_detail {

// Toom-6.5, 7 x 6: u = a0 + a1 W + ... + a6 W^6, v = b0 + ... + b5 W^5, W = B^c,
// s = |a6| = un - 6c, t = |b5| = vn - 5c, 0 < s, t <= c. The product r(x) = c0 + ... + c11 x^11
// is evaluated at the 11 points of the balanced plan (toom_6x6.hpp) -- 0, +-1, +-2, +-4, +-1/2,
// +-1/4 -- plus infinity, c11 = a6 b5. The fractional points are scaled one step further than
// there: 64 A(+-1/2), 4096 A(+-1/4) (seven pieces), with B as in the balanced plan, so the
// products are r(+-1), r(+-2), r(+-4), 2^11 r(+-1/2), 4^11 r(+-1/4).
//
// Interpolation. The even coefficients c0, c2, ..., c10 are the same six as in the balanced
// case; the odd ones are c1, ..., c9 there and c11 here in addition -- known from infinity. So
// splitting every pair into its halves also takes c11's share out (one more lincomb term each)
// and undoes the extra scaling, and the halves are then exactly the balanced plan's
// P(1), P(4), P(16), P~(4), P~(16) and Q(1), ..., Q~(16):
//   Q(1)   = (r(1) - r(-1) - 2 c11) / 2                   P(1)   = r(1) - Q(1) - c0 - c11
//   Q(4)   = (r(2) - r(-2) - 2^12 c11) / 4                P(4)   = (r(2) - 2 Q(4) - c0 - 2^11 c11) / 4
//   Q(16)  = (r(4) - r(-4) - 2^23 c11) / 8                P(16)  = (r(4) - 4 Q(16) - c0 - 2^22 c11) / 16
//   Q~(4)  = (X(1/2) - X(-1/2) - 2 c11) / 8               P~(4)  = (X(1/2) - 4 Q~(4) - c11 - 2^11 c0) / 2
//   Q~(16) = (Y(1/4) - Y(-1/4) - 2 c11) / 32              P~(16) = (Y(1/4) - 16 Q~(16) - c11 - 2^22 c0) / 4
// with X = 2^11 r, Y = 4^11 r. The rest -- 13 lincomb_dual steps, the division by 189
// included -- is the balanced plan's, unchanged.
//
// Scratch (21e + 10p limbs), e = c + 1, p = 2c + 2: as in the balanced plan, plus
//   W4HI  aliases W4[c, 2c + 2)     c10's limbs above rb's c-limb gap at 10c, added onto c11
// Result views: C0 = rb[0, 2c), R2 .. R8 = the 2c-limb pieces at 2c .. 8c, R10 = rb[10c, 11c),
// C11 = rb[11c, end); they cover rb, so it needn't be zeroed. The accumulation windows R1 .. R9
// and R4X .. R10X run from c, 3c, ... to the end.
consteval auto make_toom76()
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

    auto off_w = b.mul(e, 21);
    auto slab  = b.add(off_w, b.mul(p, 10));

    // (No helper lambdas here: GCC doesn't take consteval calls made inside them.)
    slot_builder sb;
    auto EA1  = sb.tmp(b.c(0),       e);
    auto EAM1 = sb.tmp(e,            e);
    auto EA2  = sb.tmp(b.mul(e, 2),  e);
    auto EAM2 = sb.tmp(b.mul(e, 3),  e);
    auto EA4  = sb.tmp(b.mul(e, 4),  e);
    auto EAM4 = sb.tmp(b.mul(e, 5),  e);
    auto EAH  = sb.tmp(b.mul(e, 6),  e);
    auto EAMH = sb.tmp(b.mul(e, 7),  e);
    auto EAQ  = sb.tmp(b.mul(e, 8),  e);
    auto EAMQ = sb.tmp(b.mul(e, 9),  e);
    auto EB1  = sb.tmp(b.mul(e, 10), e);
    auto EBM1 = sb.tmp(b.mul(e, 11), e);
    auto EB2  = sb.tmp(b.mul(e, 12), e);
    auto EBM2 = sb.tmp(b.mul(e, 13), e);
    auto EB4  = sb.tmp(b.mul(e, 14), e);
    auto EBM4 = sb.tmp(b.mul(e, 15), e);
    auto EBH  = sb.tmp(b.mul(e, 16), e);
    auto EBMH = sb.tmp(b.mul(e, 17), e);
    auto EBQ  = sb.tmp(b.mul(e, 18), e);
    auto EBMQ = sb.tmp(b.mul(e, 19), e);
    auto T    = sb.tmp(b.mul(e, 20), e);

    auto off_w1  = off_w;
    auto off_wm1 = b.add(off_w, p);
    auto off_w2  = b.add(off_w, b.mul(p, 2));
    auto off_wm2 = b.add(off_w, b.mul(p, 3));
    auto off_w4  = b.add(off_w, b.mul(p, 4));
    auto off_wm4 = b.add(off_w, b.mul(p, 5));
    auto off_wh  = b.add(off_w, b.mul(p, 6));
    auto off_wmh = b.add(off_w, b.mul(p, 7));
    auto off_wq  = b.add(off_w, b.mul(p, 8));
    auto off_wmq = b.add(off_w, b.mul(p, 9));
    auto W1   = sb.tmp(off_w1,  p);
    auto WM1  = sb.tmp(off_wm1, p);
    auto W2   = sb.tmp(off_w2,  p);
    auto WM2  = sb.tmp(off_wm2, p);
    auto W4   = sb.tmp(off_w4,  p);
    auto WM4  = sb.tmp(off_wm4, p);
    auto WH   = sb.tmp(off_wh,  p);
    auto WMH  = sb.tmp(off_wmh, p);
    auto WQ   = sb.tmp(off_wq,  p);
    auto WMQ  = sb.tmp(off_wmq, p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));
    auto W2HI = sb.tmp(b.add(off_w2, two_c), b.c(2));
    auto WHHI = sb.tmp(b.add(off_wh, two_c), b.c(2));
    auto WQHI = sb.tmp(b.add(off_wq, two_c), b.c(2));
    auto W4HI = sb.tmp(b.add(off_w4, c),     b.add(c, b.c(2)));

    auto C0   = sb.rb(b.c(0),        two_c);
    auto R2   = sb.rb(b.mul(c, 2),   two_c);
    auto R4   = sb.rb(b.mul(c, 4),   two_c);
    auto R6   = sb.rb(b.mul(c, 6),   two_c);
    auto R8   = sb.rb(b.mul(c, 8),   two_c);
    auto R10  = sb.rb(b.mul(c, 10),  c);
    auto C11  = sb.rb(b.mul(c, 11),  st);
    auto R1   = sb.rb(c,             b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),   b.sub(rn, b.mul(c, 3)));
    auto R5   = sb.rb(b.mul(c, 5),   b.sub(rn, b.mul(c, 5)));
    auto R7   = sb.rb(b.mul(c, 7),   b.sub(rn, b.mul(c, 7)));
    auto R9   = sb.rb(b.mul(c, 9),   b.sub(rn, b.mul(c, 9)));
    auto R4X  = sb.rb(b.mul(c, 4),   b.sub(rn, b.mul(c, 4)));
    auto R6X  = sb.rb(b.mul(c, 6),   b.sub(rn, b.mul(c, 6)));
    auto R8X  = sb.rb(b.mul(c, 8),   b.sub(rn, b.mul(c, 8)));
    auto R10X = sb.rb(b.mul(c, 10),  b.sub(rn, b.mul(c, 10)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 7> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3), slot_builder::u(4), slot_builder::u(5), slot_builder::u(6) };
    constexpr std::array<toom_ref, 6> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2), slot_builder::v(3), slot_builder::v(4), slot_builder::v(5) };

    std::array plan = {
        // Evaluate A (seven pieces): even part into the minus slot, odd part into T, then both points.
        lincomb(EAM1, { lc_add(u[0]), lc_add(u[2]), lc_add(u[4]), lc_add(u[6]) }),
        lincomb(T,    { lc_add(u[1]), lc_add(u[3]), lc_add(u[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA1, .src0 = EAM1, .src1 = T, .dst2 = EAM1 },     // A(+-1)
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2), lc_add(u[4], 4), lc_add(u[6], 6) }),
        lincomb(T,    { lc_add(u[1], 1), lc_add(u[3], 3), lc_add(u[5], 5) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 },     // A(+-2)
        lincomb(EAM4, { lc_add(u[0]), lc_add(u[2], 4), lc_add(u[4], 8), lc_add(u[6], 12) }),
        lincomb(T,    { lc_add(u[1], 2), lc_add(u[3], 6), lc_add(u[5], 10) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA4, .src0 = EAM4, .src1 = T, .dst2 = EAM4 },     // A(+-4)
        lincomb(EAMH, { lc_add(u[0], 6), lc_add(u[2], 4), lc_add(u[4], 2), lc_add(u[6]) }),
        lincomb(T,    { lc_add(u[1], 5), lc_add(u[3], 3), lc_add(u[5], 1) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAH, .src0 = EAMH, .src1 = T, .dst2 = EAMH },     // 64 A(+-1/2)
        lincomb(EAMQ, { lc_add(u[0], 12), lc_add(u[2], 8), lc_add(u[4], 4), lc_add(u[6]) }),
        lincomb(T,    { lc_add(u[1], 10), lc_add(u[3], 6), lc_add(u[5], 2) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAQ, .src0 = EAMQ, .src1 = T, .dst2 = EAMQ },     // 4096 A(+-1/4)

        // B (six pieces), as in the balanced plan.
        lincomb(EBM1, { lc_add(v[0]), lc_add(v[2]), lc_add(v[4]) }),
        lincomb(T,    { lc_add(v[1]), lc_add(v[3]), lc_add(v[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB1, .src0 = EBM1, .src1 = T, .dst2 = EBM1 },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2), lc_add(v[4], 4) }),
        lincomb(T,    { lc_add(v[1], 1), lc_add(v[3], 3), lc_add(v[5], 5) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = T, .dst2 = EBM2 },
        lincomb(EBM4, { lc_add(v[0]), lc_add(v[2], 4), lc_add(v[4], 8) }),
        lincomb(T,    { lc_add(v[1], 2), lc_add(v[3], 6), lc_add(v[5], 10) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB4, .src0 = EBM4, .src1 = T, .dst2 = EBM4 },
        lincomb(EBMH, { lc_add(v[0], 5), lc_add(v[2], 3), lc_add(v[4], 1) }),
        lincomb(T,    { lc_add(v[1], 4), lc_add(v[3], 2), lc_add(v[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EBH, .src0 = EBMH, .src1 = T, .dst2 = EBMH },
        lincomb(EBMQ, { lc_add(v[0], 10), lc_add(v[2], 6), lc_add(v[4], 2) }),
        lincomb(T,    { lc_add(v[1], 8), lc_add(v[3], 4), lc_add(v[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EBQ, .src0 = EBMQ, .src1 = T, .dst2 = EBMQ },

        // Pointwise products (signs of the minus points kept in the WM slots); c0 and c11 into rb.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2 },
        toom_instr{ toom_op::umul_fixed,  W4,   EA4,  EB4  },
        toom_instr{ toom_op::umul_fixed,  WM4,  EAM4, EBM4 },
        toom_instr{ toom_op::umul_fixed,  WH,   EAH,  EBH  },
        toom_instr{ toom_op::umul_fixed,  WMH,  EAMH, EBMH },
        toom_instr{ toom_op::umul_fixed,  WQ,   EAQ,  EBQ  },
        toom_instr{ toom_op::umul_fixed,  WMQ,  EAMQ, EBMQ },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C11,  u[6], v[5] },

        // Odd / even halves of every pair, c11's share and the extra scaling taken out: W* become
        // P(1), P(4), P(16), P~(4), P~(16), WM* Q(1), Q(4), Q(16), Q~(4), Q~(16) -- the balanced
        // plan's values from here on.
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C11, 1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C11) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2), lc_sub(C11, 12) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C11, 11) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4), lc_sub(C11, 23) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0), lc_sub(C11, 22) }, 4),
        lincomb(WMH, { lc_add(WH), lc_sub_signed(WMH), lc_sub(C11, 1) }, 3),
        lincomb(WH,  { lc_add(WH), lc_sub(WMH, 2), lc_sub(C11), lc_sub(C0, 11) }, 1),
        lincomb(WMQ, { lc_add(WQ), lc_sub_signed(WMQ), lc_sub(C11, 1) }, 5),
        lincomb(WQ,  { lc_add(WQ), lc_sub(WMQ, 4), lc_sub(C11), lc_sub(C0, 22) }, 2),

        // Both halves, as in the balanced plan (see toom_6x6.hpp for the steps).
        lincomb_dual_tc(WH, { lc_add(WH), lc_sub(W2) },
                        WMH, { lc_add(WMH), lc_sub(WM2) }, 0, 15),                                       // U
        lincomb_dual(W2,  { lc_add(W2, 1), lc_add_tc(WH, 4), lc_sub_tc(WH), lc_sub(W1, 5) },
                     WM2, { lc_add(WM2, 1), lc_add_tc(WMH, 4), lc_sub_tc(WMH), lc_sub(WM1, 5) }, 0, 9),  // G
        lincomb_dual_tc(WQ, { lc_add(WQ), lc_sub(W4) },
                        WMQ, { lc_add(WMQ), lc_sub(WM4) }, 0, 255),                                      // V
        lincomb_dual(W4,  { lc_add(W4, 1), lc_add_tc(WQ, 8), lc_sub_tc(WQ), lc_sub(W1, 9) },
                     WM4, { lc_add(WM4, 1), lc_add_tc(WMQ, 8), lc_sub_tc(WMQ), lc_sub(WM1, 9) }, 0, 225), // H
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W2, 2) },
                     WM4, { lc_add(WM4), lc_sub(WM2, 2) }, 0, 189),                                      // s1
        lincomb_dual_tc(WQ, { lc_add_tc(WQ), lc_sub_tc(WH, 2) },
                        WMQ, { lc_add_tc(WMQ), lc_sub_tc(WMH, 2) }, 0, 189),                             // d1
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W4, 4), lc_sub(W4, 3), lc_sub(W4) },
                     WM2, { lc_add(WM2), lc_sub(WM4, 4), lc_sub(WM4, 3), lc_sub(WM4) }, 2),              // s2
        lincomb_dual_tc(WH, { lc_add_tc(WH), lc_sub_tc(WQ, 4), lc_sub_tc(WQ) },
                        WMH, { lc_add_tc(WMH), lc_sub_tc(WMQ, 4), lc_sub_tc(WMQ) }, 2),                  // d2
        lincomb_dual(W1,  { lc_add(W1), lc_sub(W4), lc_sub(W2) },
                     WM1, { lc_add(WM1), lc_sub(WM4), lc_sub(WM2) }),                                    // e3: c6, c5
        lincomb_dual(WQ,  { lc_add(W4), lc_add_tc(WQ) },
                     WMQ, { lc_add(WM4), lc_add_tc(WMQ) }, 1),                                           // e1: c2, c1
        lincomb_dual(W4,  { lc_add(W4), lc_sub(WQ) },
                     WM4, { lc_add(WM4), lc_sub(WMQ) }),                                                 // e5: c10, c9
        lincomb_dual(WH,  { lc_add(W2), lc_add_tc(WH) },
                     WMH, { lc_add(WM2), lc_add_tc(WMH) }, 1),                                           // e2: c4, c3
        lincomb_dual(W2,  { lc_add(W2), lc_sub(WH) },
                     WM2, { lc_add(WM2), lc_sub(WMH) }),                                                 // e4: c8, c7

        // Compose: the even coefficients fill rb's pieces (their top limbs land on the next
        // piece; c10's above its c-limb gap on c11), then the odd ones are added.
        toom_instr{ toom_op::copy_low,  R2,   WQ   },
        toom_instr{ toom_op::copy_low,  R4,   WH   },
        toom_instr{ toom_op::copy_low,  R6,   W1   },
        toom_instr{ toom_op::copy_low,  R8,   W2   },
        toom_instr{ toom_op::copy_low,  R10,  W4   },
        toom_instr{ toom_op::uadd_into, C11,  W4HI },
        toom_instr{ toom_op::uadd_into, R4X,  WQHI },
        toom_instr{ toom_op::uadd_into, R6X,  WHHI },
        toom_instr{ toom_op::uadd_into, R8X,  W1HI },
        toom_instr{ toom_op::uadd_into, R10X, W2HI },
        toom_instr{ toom_op::uadd_into, R1,   WMQ  },
        toom_instr{ toom_op::uadd_into, R3,   WMH  },
        toom_instr{ toom_op::uadd_into, R5,   WM1  },
        toom_instr{ toom_op::uadd_into, R7,   WM2  },
        toom_instr{ toom_op::uadd_into, R9,   WM4  },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom76 = make_toom76();

// c = ceil(un/7)
struct toom76_u_traits
{
    static constexpr auto const& plan              = toom76.plan;
    static constexpr auto const& size_exprs        = toom76.exprs;
    static constexpr auto const& slot_layout       = toom76.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom76.slab_expr;
    static constexpr size_t N = 7;
    static constexpr size_t M = 6;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

// c = ceil(vn/6)
struct toom76_v_traits
{
    static constexpr auto const& plan              = toom76.plan;
    static constexpr auto const& size_exprs        = toom76.exprs;
    static constexpr auto const& slot_layout       = toom76.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom76.slab_expr;
    static constexpr size_t N = 7;
    static constexpr size_t M = 6;
    static constexpr bool split_by_u = false;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
