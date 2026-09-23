// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Whether the balanced Toom-6.5 plan can take un x vn (un >= vn): it splits u into sixths of
// c = ceil(un/6) limbs and needs v to reach into its top sixth (vn > 5c), which also makes u's
// top sixth non-empty.
inline bool toom6h_split_fits(size_t un, size_t vn) noexcept
{
    return vn > 5 * ((un + 5) / 6);
}

}

namespace toom_runtime_detail {

// Balanced Toom-6.5 (the 6 x 6 case of GMP's toom6h; split_by_u): u = a0 + a1 W + ... + a5 W^5,
// v likewise with b0..b5, W = B^c, c = ceil(un/6), s = un - 5c = |a5|, t = vn - 5c = |b5|,
// 0 < t <= s <= c. The product r(x) = c0 + c1 x + ... + c10 x^10 is evaluated at the 11 points
// 0, +-1, +-2, +-4, +-1/2, +-1/4 (no infinity: five symmetric pairs keep the interpolation
// split cleanly into an even and an odd half):
//   A(+-x) = E(x) +- O(x), the even and odd parts formed by one lincomb each, then addsub_abs;
//   the fractional points scaled to integers: 32 A(+-1/2) and 1024 A(+-1/4).
// The products are c0 = a0 b0 (straight into rb) and r(+-1), r(+-2), r(+-4), 1024 r(+-1/2),
// 4^10 r(+-1/4) (the minus ones as magnitude and sign).
//
// Interpolation. Each pair splits into its even and odd halves (one lincomb each, c0's share
// taken out of the even one), scaled so that with the even coefficients e1..e5 = c2, c4, ..., c10,
// P(y) = e1 + e2 y + e3 y^2 + e4 y^3 + e5 y^4 and its reverse P~(y) = y^4 P(1/y):
//   A1 = r(1) - D1 - c0                          = P(1)       D1 = (r(1) - r(-1)) / 2
//   A4 = (r(2) - 2 D2 - c0) / 4                  = P(4)       D2 = (r(2) - r(-2)) / 4
//   A16 = (r(4) - 4 D4 - c0) / 16                = P(16)      D4 = (r(4) - r(-4)) / 8
//   Ah = 1024 r(1/2) - 2 Dh - 1024 c0            = P~(4)      Dh = 1024 (r(1/2) - r(-1/2)) / 4
//   Aq = 4^10 r(1/4) - 4 Dq - 4^20 c0            = P~(16)     Dq = 4^10 (r(1/4) - r(-1/4)) / 8
// and the odd coefficients q1..q5 = c1, c3, ..., c9 in Q(y) = q1 + q2 y + ... + q5 y^4 the same
// way: D1 = Q(1), D2 = Q(4), D4 = Q(16), Dh = Q~(4), Dq = Q~(16). Both halves are then the same
// 5-point problem -- a quartic known at 1, 4, 16 and its reverse at 4, 16 -- solved through the
// sums and differences of the mirrored coefficients, s1 = e1 + e5, s2 = e2 + e4, d1 = e1 - e5,
// d2 = e2 - e4 (d1, d2 and the U, V below may be negative: two's complement, lincomb_tc):
//   U  = (P~(4) - P(4)) / 15                      = 17 d1 + 4 d2
//   G  = (2 P(4) + 15 U - 32 P(1)) / 9            = 25 s1 + 4 s2       (2 P(4) + 15 U = P(4) + P~(4))
//   V  = (P~(16) - P(16)) / 255                   = 257 d1 + 16 d2
//   H  = (2 P(16) + 255 V - 512 P(1)) / 225       = 289 s1 + 16 s2
//   s1 = (H - 4 G) / 189,   d1 = (V - 4 U) / 189
//   s2 = (G - 25 s1) / 4,   d2 = (U - 17 d1) / 4
//   e3 = P(1) - s1 - s2
//   e1 = (s1 + d1) / 2,  e5 = s1 - e1,  e2 = (s2 + d2) / 2,  e4 = s2 - e2
// 13 lincomb passes per half. 189 = 27 * 7 doesn't divide B - 1 (no set of 5 points of this kind
// avoids a factor 7), so those four passes divide by Hensel's method, the others by divisors of
// B - 1 (9 = 3 * 3 and 225 = 15 * 15 as two stages).
//
// Scratch (21e + 10p limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EAM2 EA4 EAM4 EAH EAMH EAQ EAMQ    10 x e   A at +-1, +-2, +-4, +-1/2, +-1/4
//   EB1 ... EBMQ                                    10 x e   the same for B
//   T                                               e        the odd part while evaluating
//   W1 WM1 W2 WM2 W4 WM4 WH WMH WQ WMQ              10 x p   the products, then P / Q values,
//                                                            ending as c6 c5 c8 c7 c10 c9 c4 c3 c2 c1
//   W1HI W2HI WHHI WQHI alias [2c, 2c + 2) of W1 W2 WH WQ   the parts of c6 c8 c4 c2 above 2c
// Result views: C0 = rb[0, 2c), R2, R4, R6, R8 = the 2c-limb pieces at 2c, 4c, 6c, 8c,
// C10 = rb[10c, end); they cover rb, so it needn't be zeroed. The accumulation windows R1, R3,
// R5, R7, R9 and R4X, R6X, R8X run from c, 3c, ... to the end.
consteval auto make_toom6h_balanced()
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

    auto C0   = sb.rb(b.c(0),        two_c);
    auto R2   = sb.rb(b.mul(c, 2),   two_c);
    auto R4   = sb.rb(b.mul(c, 4),   two_c);
    auto R6   = sb.rb(b.mul(c, 6),   two_c);
    auto R8   = sb.rb(b.mul(c, 8),   two_c);
    auto C10  = sb.rb(b.mul(c, 10),  st);
    auto R1   = sb.rb(c,             b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),   b.sub(rn, b.mul(c, 3)));
    auto R5   = sb.rb(b.mul(c, 5),   b.sub(rn, b.mul(c, 5)));
    auto R7   = sb.rb(b.mul(c, 7),   b.sub(rn, b.mul(c, 7)));
    auto R9   = sb.rb(b.mul(c, 9),   b.sub(rn, b.mul(c, 9)));
    auto R4X  = sb.rb(b.mul(c, 4),   b.sub(rn, b.mul(c, 4)));
    auto R6X  = sb.rb(b.mul(c, 6),   b.sub(rn, b.mul(c, 6)));
    auto R8X  = sb.rb(b.mul(c, 8),   b.sub(rn, b.mul(c, 8)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 6> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3), slot_builder::u(4), slot_builder::u(5) };
    constexpr std::array<toom_ref, 6> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2), slot_builder::v(3), slot_builder::v(4), slot_builder::v(5) };

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
        lincomb(EAMH, { lc_add(u[0], 5), lc_add(u[2], 3), lc_add(u[4], 1) }),
        lincomb(T,    { lc_add(u[1], 4), lc_add(u[3], 2), lc_add(u[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAH, .src0 = EAMH, .src1 = T, .dst2 = EAMH },     // 32 A(+-1/2)
        lincomb(EAMQ, { lc_add(u[0], 10), lc_add(u[2], 6), lc_add(u[4], 2) }),
        lincomb(T,    { lc_add(u[1], 8), lc_add(u[3], 4), lc_add(u[5]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAQ, .src0 = EAMQ, .src1 = T, .dst2 = EAMQ },     // 1024 A(+-1/4)

        // Same for B.
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

        // Pointwise products (signs of the minus points kept in the WM slots); c0 into rb.
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

        // Odd / even halves of every pair: W* become P(1), P(4), P(16), P~(4), P~(16),
        // WM* Q(1), Q(4), Q(16), Q~(4), Q~(16).
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0) }, 4),
        lincomb(WMH, { lc_add(WH), lc_sub_signed(WMH) }, 2),
        lincomb(WH,  { lc_add(WH), lc_sub(WMH, 1), lc_sub(C0, 10) }),
        lincomb(WMQ, { lc_add(WQ), lc_sub_signed(WMQ) }, 3),
        lincomb(WQ,  { lc_add(WQ), lc_sub(WMQ, 2), lc_sub(C0, 20) }),

        // Even half: (P(1), P(4), P(16), P~(4), P~(16)) in (W1, W2, W4, WH, WQ) -> c6, c8, c10, c4, c2.
        lincomb_tc(WH, { lc_add(WH), lc_sub(W2) }, 0, 15),                                        // U
        lincomb(W2,    { lc_add(W2, 1), lc_add_tc(WH, 4), lc_sub_tc(WH), lc_sub(W1, 5) }, 0, 9),    // G
        lincomb_tc(WQ, { lc_add(WQ), lc_sub(W4) }, 0, 255),                                       // V
        lincomb(W4,    { lc_add(W4, 1), lc_add_tc(WQ, 8), lc_sub_tc(WQ), lc_sub(W1, 9) }, 0, 225),  // H
        lincomb(W4,    { lc_add(W4), lc_sub(W2, 2) }, 0, 189),                                    // s1
        lincomb_tc(WQ, { lc_add_tc(WQ), lc_sub_tc(WH, 2) }, 0, 189),                              // d1
        lincomb(W2,    { lc_add(W2), lc_sub(W4, 4), lc_sub(W4, 3), lc_sub(W4) }, 2),              // s2
        lincomb_tc(WH, { lc_add_tc(WH), lc_sub_tc(WQ, 4), lc_sub_tc(WQ) }, 2),                    // d2
        lincomb(W1,    { lc_add(W1), lc_sub(W4), lc_sub(W2) }),                                   // e3 = c6
        lincomb(WQ,    { lc_add(W4), lc_add_tc(WQ) }, 1),                                         // e1 = c2
        lincomb(W4,    { lc_add(W4), lc_sub(WQ) }),                                               // e5 = c10
        lincomb(WH,    { lc_add(W2), lc_add_tc(WH) }, 1),                                         // e2 = c4
        lincomb(W2,    { lc_add(W2), lc_sub(WH) }),                                               // e4 = c8

        // Odd half: (Q(1), Q(4), Q(16), Q~(4), Q~(16)) in (WM1, WM2, WM4, WMH, WMQ) -> c5, c7, c9, c3, c1.
        lincomb_tc(WMH, { lc_add(WMH), lc_sub(WM2) }, 0, 15),
        lincomb(WM2,    { lc_add(WM2, 1), lc_add_tc(WMH, 4), lc_sub_tc(WMH), lc_sub(WM1, 5) }, 0, 9),
        lincomb_tc(WMQ, { lc_add(WMQ), lc_sub(WM4) }, 0, 255),
        lincomb(WM4,    { lc_add(WM4, 1), lc_add_tc(WMQ, 8), lc_sub_tc(WMQ), lc_sub(WM1, 9) }, 0, 225),
        lincomb(WM4,    { lc_add(WM4), lc_sub(WM2, 2) }, 0, 189),
        lincomb_tc(WMQ, { lc_add_tc(WMQ), lc_sub_tc(WMH, 2) }, 0, 189),
        lincomb(WM2,    { lc_add(WM2), lc_sub(WM4, 4), lc_sub(WM4, 3), lc_sub(WM4) }, 2),
        lincomb_tc(WMH, { lc_add_tc(WMH), lc_sub_tc(WMQ, 4), lc_sub_tc(WMQ) }, 2),
        lincomb(WM1,    { lc_add(WM1), lc_sub(WM4), lc_sub(WM2) }),                               // c5
        lincomb(WMQ,    { lc_add(WM4), lc_add_tc(WMQ) }, 1),                                      // c1
        lincomb(WM4,    { lc_add(WM4), lc_sub(WMQ) }),                                            // c9
        lincomb(WMH,    { lc_add(WM2), lc_add_tc(WMH) }, 1),                                      // c3
        lincomb(WM2,    { lc_add(WM2), lc_sub(WMH) }),                                            // c7

        // Compose: the even coefficients fill rb's pieces (their top limbs land on the next
        // piece), then the odd ones are added.
        toom_instr{ toom_op::copy_low,  R2,  WQ   },
        toom_instr{ toom_op::copy_low,  R4,  WH   },
        toom_instr{ toom_op::copy_low,  R6,  W1   },
        toom_instr{ toom_op::copy_low,  R8,  W2   },
        toom_instr{ toom_op::copy_low,  C10, W4   },
        toom_instr{ toom_op::uadd_into, R4X, WQHI },
        toom_instr{ toom_op::uadd_into, R6X, WHHI },
        toom_instr{ toom_op::uadd_into, R8X, W1HI },
        toom_instr{ toom_op::uadd_into, C10, W2HI },
        toom_instr{ toom_op::uadd_into, R1,  WMQ  },
        toom_instr{ toom_op::uadd_into, R3,  WMH  },
        toom_instr{ toom_op::uadd_into, R5,  WM1  },
        toom_instr{ toom_op::uadd_into, R7,  WM2  },
        toom_instr{ toom_op::uadd_into, R9,  WM4  },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom6h_balanced = make_toom6h_balanced();

struct toom6h_balanced_traits
{
    static constexpr auto const& plan              = toom6h_balanced.plan;
    static constexpr auto const& size_exprs        = toom6h_balanced.exprs;
    static constexpr auto const& slot_layout       = toom6h_balanced.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom6h_balanced.slab_expr;
    static constexpr size_t N = 6;
    static constexpr size_t M = 6;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
