// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Whether the balanced Toom-8.5 plan can take un x vn (un >= vn): it splits u into eighths of
// c = ceil(un/8) limbs and needs v to reach into its top eighth (vn > 7c), which also makes u's
// top eighth non-empty.
inline bool toom8h_split_fits(size_t un, size_t vn) noexcept
{
    return vn > 7 * ((un + 7) / 8);
}

}

namespace toom_runtime_detail {

// Balanced Toom-8.5 (the 8 x 8 case of GMP's toom8h; split_by_u): u = a0 + a1 W + ... + a7 W^7,
// v likewise with b0..b7, W = B^c, c = ceil(un/8), s = un - 7c = |a7|, t = vn - 7c = |b7|,
// 0 < t <= s <= c. The product r(x) = c0 + c1 x + ... + c14 x^14 is evaluated at the 15 points
// 0, +-1, +-2, +-4, +-8, +-1/2, +-1/4, +-1/8 -- the scheme of toom_6x6.hpp with two more pairs:
//   A(+-x) = E(x) +- O(x), the even and odd parts formed by one lincomb each, then addsub_abs;
//   the fractional points scaled to integers: 2^7 A(+-1/2), 4^7 A(+-1/4), 8^7 A(+-1/8).
// The products are c0 = a0 b0 (straight into rb) and r(+-1), r(+-2), r(+-4), r(+-8),
// 2^14 r(+-1/2), 4^14 r(+-1/4), 8^14 r(+-1/8) (the minus ones as magnitude and sign).
//
// Interpolation. Each pair splits into its even and odd halves (one lincomb each, c0's share
// taken out of the even one): with the even coefficients e1..e7 = c2, c4, ..., c14,
// P(y) = e1 + e2 y + ... + e7 y^6 and its reverse P~(y) = y^6 P(1/y), the point x = 2^j gives
// P(4^j) and the point 1/x gives P~(4^j); the odd coefficients c1, c3, ..., c13 likewise give
// Q(4^j) and Q~(4^j). Both halves are then the same 7-point problem -- a sextic known at 1, 4,
// 16, 64 and its reverse at 4, 16, 64 -- solved through the sums and differences of mirrored
// coefficients: s1 = e1 + e7, s2 = e2 + e6, s3 = e3 + e5, d1 = e1 - e7, d2 = e2 - e6,
// d3 = e3 - e5 (the d's and the U, X, Y below may be negative: two's complement, lincomb_tc).
// With y^6 - 1, y^5 - y, y^4 - y^2 all divisible by y^2 - 1 and the symmetric part's
// P + P~ - 2 y^3 P(1) = s1 (y^3 - 1)^2 + s2 y (y^2 - 1)^2 + s3 y^2 (y - 1)^2 by (y - 1)^2:
//   U_y = (P~(y) - P(y)) / (y^2 - 1)            = d1 (y^4 + y^2 + 1) + d2 y (y^2 + 1) + d3 y^2
//   G_y = (P(y) + P~(y) - 2 y^3 P(1)) / (y - 1)^2 = s1 (y^2 + y + 1)^2 + s2 y (y + 1)^2 + s3 y^2
// for y = 4, 16, 64 (P + P~ = 2P + (y^2 - 1) U_y, so U_y can overwrite P~ first), i.e.
//   U4 = 273 d1 + 68 d2 + 16 d3,          G4 = 441 s1 + 100 s2 + 16 s3
//   U16 = 65793 d1 + 4112 d2 + 256 d3,    G16 = 74529 s1 + 4624 s2 + 256 s3
//   U64 = ... + 4096 d3,                  G64 = ... + 4096 s3
// and eliminating the last two unknowns:
//   Y = (U64 - 16 U16) / 3069 = 5125 d1 + 64 d2,   Y' = (G64 - 16 G16) / 3069 = 5253 s1 + 64 s2
//   X = (U16 - 16 U4) / 189   = 325 d1 + 16 d2,    X' = (G16 - 16 G4) / 189   = 357 s1 + 16 s2
//   d1 = (Y - 4 X) / 3825,                         s1 = (Y' - 4 X') / 3825
//   d2 = (X - 325 d1) / 16,  d3 = (U4 - 273 d1 - 68 d2) / 16
//   s2 = (X' - 357 s1) / 16, s3 = (G4 - 441 s1 - 100 s2) / 16,  e4 = P(1) - s1 - s2 - s3
//   e1 = (s1 + d1) / 2, e7 = s1 - e1, and the same for (e2, e6), (e3, e5)
// 28 steps (the multiplications by 325, 357, 273, 441 take two or three passes of at most four
// terms each), each one lincomb_dual pass doing it for both halves at once. 9 = 3 * 3,
// 225 = 15 * 15 and 3825 = 15 * 255 divide as two stages dividing B - 1; the other divisors
// don't factor that way and need Hensel's method, which is latency-bound -- so they are kept to
// four such passes: U64 and G64 are only needed for Y and Y', so they are kept undivided
// (V64 = 4095 U64, H64 = 3969 G64) and their divisors folded into those of Y and Y':
//   Y = (V64 - 65520 U16) / (4095 * 3069),   Y' = (H64 - 63504 G16) / (3969 * 3069)
// leaving Y, Y', X, X' -- and running the halves interleaved hides half of what those cost.
//
// Scratch (29e + 14p limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EAM2 EA4 EAM4 EA8 EAM8 EAH EAMH EAQ EAMQ EAE EAME   14 x e   A at the 14 points
//   EB1 ... EBME                                                    14 x e   the same for B
//   T                                                               e        odd part while evaluating
//   W1 WM1 W2 WM2 W4 WM4 W8 WM8 WH WMH WQ WMQ WE WME                14 x p   the products, then P / Q
//     values, ending as c8 c7 c10 c9 c12 c11 c14 c13 c6 c5 c4 c3 c2 c1
//   W1HI W2HI W4HI WHHI WQHI WEHI alias [2c, 2c + 2) of W1 W2 W4 WH WQ WE: the parts of
//     c8 c10 c12 c6 c4 c2 above 2c
// Result views: C0 = rb[0, 2c), R2 .. R12 = the 2c-limb pieces at 2c .. 12c, C14 = rb[14c, end);
// they cover rb, so it needn't be zeroed. The accumulation windows R1, R3, ..., R13 and
// R4X .. R12X run from c, 3c, ... to the end.
consteval auto make_toom8h_balanced()
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

    auto off_w = b.mul(e, 29);
    auto slab  = b.add(off_w, b.mul(p, 14));

    // (No helper lambdas here: GCC doesn't take consteval calls made inside them.)
    slot_builder sb;
    auto EA1  = sb.tmp(b.c(0),       e);
    auto EAM1 = sb.tmp(e,            e);
    auto EA2  = sb.tmp(b.mul(e, 2),  e);
    auto EAM2 = sb.tmp(b.mul(e, 3),  e);
    auto EA4  = sb.tmp(b.mul(e, 4),  e);
    auto EAM4 = sb.tmp(b.mul(e, 5),  e);
    auto EA8  = sb.tmp(b.mul(e, 6),  e);
    auto EAM8 = sb.tmp(b.mul(e, 7),  e);
    auto EAH  = sb.tmp(b.mul(e, 8),  e);
    auto EAMH = sb.tmp(b.mul(e, 9),  e);
    auto EAQ  = sb.tmp(b.mul(e, 10), e);
    auto EAMQ = sb.tmp(b.mul(e, 11), e);
    auto EAE  = sb.tmp(b.mul(e, 12), e);
    auto EAME = sb.tmp(b.mul(e, 13), e);
    auto EB1  = sb.tmp(b.mul(e, 14), e);
    auto EBM1 = sb.tmp(b.mul(e, 15), e);
    auto EB2  = sb.tmp(b.mul(e, 16), e);
    auto EBM2 = sb.tmp(b.mul(e, 17), e);
    auto EB4  = sb.tmp(b.mul(e, 18), e);
    auto EBM4 = sb.tmp(b.mul(e, 19), e);
    auto EB8  = sb.tmp(b.mul(e, 20), e);
    auto EBM8 = sb.tmp(b.mul(e, 21), e);
    auto EBH  = sb.tmp(b.mul(e, 22), e);
    auto EBMH = sb.tmp(b.mul(e, 23), e);
    auto EBQ  = sb.tmp(b.mul(e, 24), e);
    auto EBMQ = sb.tmp(b.mul(e, 25), e);
    auto EBE  = sb.tmp(b.mul(e, 26), e);
    auto EBME = sb.tmp(b.mul(e, 27), e);
    auto T    = sb.tmp(b.mul(e, 28), e);

    auto off_w1  = off_w;
    auto off_wm1 = b.add(off_w, p);
    auto off_w2  = b.add(off_w, b.mul(p, 2));
    auto off_wm2 = b.add(off_w, b.mul(p, 3));
    auto off_w4  = b.add(off_w, b.mul(p, 4));
    auto off_wm4 = b.add(off_w, b.mul(p, 5));
    auto off_w8  = b.add(off_w, b.mul(p, 6));
    auto off_wm8 = b.add(off_w, b.mul(p, 7));
    auto off_wh  = b.add(off_w, b.mul(p, 8));
    auto off_wmh = b.add(off_w, b.mul(p, 9));
    auto off_wq  = b.add(off_w, b.mul(p, 10));
    auto off_wmq = b.add(off_w, b.mul(p, 11));
    auto off_we  = b.add(off_w, b.mul(p, 12));
    auto off_wme = b.add(off_w, b.mul(p, 13));
    auto W1   = sb.tmp(off_w1,  p);
    auto WM1  = sb.tmp(off_wm1, p);
    auto W2   = sb.tmp(off_w2,  p);
    auto WM2  = sb.tmp(off_wm2, p);
    auto W4   = sb.tmp(off_w4,  p);
    auto WM4  = sb.tmp(off_wm4, p);
    auto W8   = sb.tmp(off_w8,  p);
    auto WM8  = sb.tmp(off_wm8, p);
    auto WH   = sb.tmp(off_wh,  p);
    auto WMH  = sb.tmp(off_wmh, p);
    auto WQ   = sb.tmp(off_wq,  p);
    auto WMQ  = sb.tmp(off_wmq, p);
    auto WE   = sb.tmp(off_we,  p);
    auto WME  = sb.tmp(off_wme, p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));
    auto W2HI = sb.tmp(b.add(off_w2, two_c), b.c(2));
    auto W4HI = sb.tmp(b.add(off_w4, two_c), b.c(2));
    auto WHHI = sb.tmp(b.add(off_wh, two_c), b.c(2));
    auto WQHI = sb.tmp(b.add(off_wq, two_c), b.c(2));
    auto WEHI = sb.tmp(b.add(off_we, two_c), b.c(2));

    auto C0   = sb.rb(b.c(0),        two_c);
    auto R2   = sb.rb(b.mul(c, 2),   two_c);
    auto R4   = sb.rb(b.mul(c, 4),   two_c);
    auto R6   = sb.rb(b.mul(c, 6),   two_c);
    auto R8   = sb.rb(b.mul(c, 8),   two_c);
    auto R10  = sb.rb(b.mul(c, 10),  two_c);
    auto R12  = sb.rb(b.mul(c, 12),  two_c);
    auto C14  = sb.rb(b.mul(c, 14),  st);
    auto R1   = sb.rb(c,             b.sub(rn, c));
    auto R3   = sb.rb(b.mul(c, 3),   b.sub(rn, b.mul(c, 3)));
    auto R5   = sb.rb(b.mul(c, 5),   b.sub(rn, b.mul(c, 5)));
    auto R7   = sb.rb(b.mul(c, 7),   b.sub(rn, b.mul(c, 7)));
    auto R9   = sb.rb(b.mul(c, 9),   b.sub(rn, b.mul(c, 9)));
    auto R11  = sb.rb(b.mul(c, 11),  b.sub(rn, b.mul(c, 11)));
    auto R13  = sb.rb(b.mul(c, 13),  b.sub(rn, b.mul(c, 13)));
    auto R4X  = sb.rb(b.mul(c, 4),   b.sub(rn, b.mul(c, 4)));
    auto R6X  = sb.rb(b.mul(c, 6),   b.sub(rn, b.mul(c, 6)));
    auto R8X  = sb.rb(b.mul(c, 8),   b.sub(rn, b.mul(c, 8)));
    auto R10X = sb.rb(b.mul(c, 10),  b.sub(rn, b.mul(c, 10)));
    auto R12X = sb.rb(b.mul(c, 12),  b.sub(rn, b.mul(c, 12)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 8> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3),
                                            slot_builder::u(4), slot_builder::u(5), slot_builder::u(6), slot_builder::u(7) };
    constexpr std::array<toom_ref, 8> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2), slot_builder::v(3),
                                            slot_builder::v(4), slot_builder::v(5), slot_builder::v(6), slot_builder::v(7) };

    std::array plan = {
        // Evaluate A: even part into the minus slot, odd part into T, then both points.
        lincomb(EAM1, { lc_add(u[0]), lc_add(u[2]), lc_add(u[4]), lc_add(u[6]) }),
        lincomb(T,    { lc_add(u[1]), lc_add(u[3]), lc_add(u[5]), lc_add(u[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA1, .src0 = EAM1, .src1 = T, .dst2 = EAM1 },     // A(+-1)
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2), lc_add(u[4], 4), lc_add(u[6], 6) }),
        lincomb(T,    { lc_add(u[1], 1), lc_add(u[3], 3), lc_add(u[5], 5), lc_add(u[7], 7) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 },     // A(+-2)
        lincomb(EAM4, { lc_add(u[0]), lc_add(u[2], 4), lc_add(u[4], 8), lc_add(u[6], 12) }),
        lincomb(T,    { lc_add(u[1], 2), lc_add(u[3], 6), lc_add(u[5], 10), lc_add(u[7], 14) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA4, .src0 = EAM4, .src1 = T, .dst2 = EAM4 },     // A(+-4)
        lincomb(EAM8, { lc_add(u[0]), lc_add(u[2], 6), lc_add(u[4], 12), lc_add(u[6], 18) }),
        lincomb(T,    { lc_add(u[1], 3), lc_add(u[3], 9), lc_add(u[5], 15), lc_add(u[7], 21) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA8, .src0 = EAM8, .src1 = T, .dst2 = EAM8 },     // A(+-8)
        lincomb(EAMH, { lc_add(u[0], 7), lc_add(u[2], 5), lc_add(u[4], 3), lc_add(u[6], 1) }),
        lincomb(T,    { lc_add(u[1], 6), lc_add(u[3], 4), lc_add(u[5], 2), lc_add(u[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAH, .src0 = EAMH, .src1 = T, .dst2 = EAMH },     // 2^7 A(+-1/2)
        lincomb(EAMQ, { lc_add(u[0], 14), lc_add(u[2], 10), lc_add(u[4], 6), lc_add(u[6], 2) }),
        lincomb(T,    { lc_add(u[1], 12), lc_add(u[3], 8), lc_add(u[5], 4), lc_add(u[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAQ, .src0 = EAMQ, .src1 = T, .dst2 = EAMQ },     // 4^7 A(+-1/4)
        lincomb(EAME, { lc_add(u[0], 21), lc_add(u[2], 15), lc_add(u[4], 9), lc_add(u[6], 3) }),
        lincomb(T,    { lc_add(u[1], 18), lc_add(u[3], 12), lc_add(u[5], 6), lc_add(u[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EAE, .src0 = EAME, .src1 = T, .dst2 = EAME },     // 8^7 A(+-1/8)

        // Same for B.
        lincomb(EBM1, { lc_add(v[0]), lc_add(v[2]), lc_add(v[4]), lc_add(v[6]) }),
        lincomb(T,    { lc_add(v[1]), lc_add(v[3]), lc_add(v[5]), lc_add(v[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB1, .src0 = EBM1, .src1 = T, .dst2 = EBM1 },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2), lc_add(v[4], 4), lc_add(v[6], 6) }),
        lincomb(T,    { lc_add(v[1], 1), lc_add(v[3], 3), lc_add(v[5], 5), lc_add(v[7], 7) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = T, .dst2 = EBM2 },
        lincomb(EBM4, { lc_add(v[0]), lc_add(v[2], 4), lc_add(v[4], 8), lc_add(v[6], 12) }),
        lincomb(T,    { lc_add(v[1], 2), lc_add(v[3], 6), lc_add(v[5], 10), lc_add(v[7], 14) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB4, .src0 = EBM4, .src1 = T, .dst2 = EBM4 },
        lincomb(EBM8, { lc_add(v[0]), lc_add(v[2], 6), lc_add(v[4], 12), lc_add(v[6], 18) }),
        lincomb(T,    { lc_add(v[1], 3), lc_add(v[3], 9), lc_add(v[5], 15), lc_add(v[7], 21) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB8, .src0 = EBM8, .src1 = T, .dst2 = EBM8 },
        lincomb(EBMH, { lc_add(v[0], 7), lc_add(v[2], 5), lc_add(v[4], 3), lc_add(v[6], 1) }),
        lincomb(T,    { lc_add(v[1], 6), lc_add(v[3], 4), lc_add(v[5], 2), lc_add(v[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EBH, .src0 = EBMH, .src1 = T, .dst2 = EBMH },
        lincomb(EBMQ, { lc_add(v[0], 14), lc_add(v[2], 10), lc_add(v[4], 6), lc_add(v[6], 2) }),
        lincomb(T,    { lc_add(v[1], 12), lc_add(v[3], 8), lc_add(v[5], 4), lc_add(v[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EBQ, .src0 = EBMQ, .src1 = T, .dst2 = EBMQ },
        lincomb(EBME, { lc_add(v[0], 21), lc_add(v[2], 15), lc_add(v[4], 9), lc_add(v[6], 3) }),
        lincomb(T,    { lc_add(v[1], 18), lc_add(v[3], 12), lc_add(v[5], 6), lc_add(v[7]) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EBE, .src0 = EBME, .src1 = T, .dst2 = EBME },

        // Pointwise products (signs of the minus points kept in the WM slots); c0 into rb.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2 },
        toom_instr{ toom_op::umul_fixed,  W4,   EA4,  EB4  },
        toom_instr{ toom_op::umul_fixed,  WM4,  EAM4, EBM4 },
        toom_instr{ toom_op::umul_fixed,  W8,   EA8,  EB8  },
        toom_instr{ toom_op::umul_fixed,  WM8,  EAM8, EBM8 },
        toom_instr{ toom_op::umul_fixed,  WH,   EAH,  EBH  },
        toom_instr{ toom_op::umul_fixed,  WMH,  EAMH, EBMH },
        toom_instr{ toom_op::umul_fixed,  WQ,   EAQ,  EBQ  },
        toom_instr{ toom_op::umul_fixed,  WMQ,  EAMQ, EBMQ },
        toom_instr{ toom_op::umul_fixed,  WE,   EAE,  EBE  },
        toom_instr{ toom_op::umul_fixed,  WME,  EAME, EBME },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },

        // Odd / even halves of every pair: W* become P(1), P(4), P(16), P(64), P~(4), P~(16),
        // P~(64), WM* the same for Q.
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0) }, 4),
        lincomb(WM8, { lc_add(W8), lc_sub_signed(WM8) }, 4),
        lincomb(W8,  { lc_add(W8), lc_sub(WM8, 3), lc_sub(C0) }, 6),
        lincomb(WMH, { lc_add(WH), lc_sub_signed(WMH) }, 2),
        lincomb(WH,  { lc_add(WH), lc_sub(WMH, 1), lc_sub(C0, 14) }),
        lincomb(WMQ, { lc_add(WQ), lc_sub_signed(WMQ) }, 3),
        lincomb(WQ,  { lc_add(WQ), lc_sub(WMQ, 2), lc_sub(C0, 28) }),
        lincomb(WME, { lc_add(WE), lc_sub_signed(WME) }, 4),
        lincomb(WE,  { lc_add(WE), lc_sub(WME, 3), lc_sub(C0, 42) }),

        // Both halves step by step, each step one lincomb_dual (the halves are independent and
        // take identical steps). Even half: (P(1), P(4), P(16), P(64), P~(4), P~(16), P~(64)) in
        // (W1, W2, W4, W8, WH, WQ, WE) -> c8, c10, c12, c14, c6, c4, c2; odd half: the same on
        // (WM1, WM2, WM4, WM8, WMH, WMQ, WME) -> c7, c9, c11, c13, c5, c3, c1.
        lincomb_dual_tc(WH, { lc_add(WH), lc_sub(W2) },
                        WMH, { lc_add(WMH), lc_sub(WM2) }, 0, 15),                                          // U4
        lincomb_dual_tc(WQ, { lc_add(WQ), lc_sub(W4) },
                        WMQ, { lc_add(WMQ), lc_sub(WM4) }, 0, 255),                                         // U16
        lincomb_dual_tc(WE, { lc_add(WE), lc_sub(W8) },
                        WME, { lc_add(WME), lc_sub(WM8) }),                                                 // V64 = 4095 U64
        lincomb_dual(W2,  { lc_add(W2, 1), lc_add_tc(WH, 4), lc_sub_tc(WH), lc_sub(W1, 7) },
                     WM2, { lc_add(WM2, 1), lc_add_tc(WMH, 4), lc_sub_tc(WMH), lc_sub(WM1, 7) }, 0, 9),     // G4
        lincomb_dual(W4,  { lc_add(W4, 1), lc_add_tc(WQ, 8), lc_sub_tc(WQ), lc_sub(W1, 13) },
                     WM4, { lc_add(WM4, 1), lc_add_tc(WMQ, 8), lc_sub_tc(WMQ), lc_sub(WM1, 13) }, 0, 225), // G16
        lincomb_dual(W8,  { lc_add(W8, 1), lc_add_tc(WE), lc_sub(W1, 19) },
                     WM8, { lc_add(WM8, 1), lc_add_tc(WME), lc_sub(WM1, 19) }),                             // H64 = 3969 G64
        lincomb_dual(W8,  { lc_add(W8), lc_sub(W4, 16), lc_add(W4, 11), lc_sub(W4, 4) },
                     WM8, { lc_add(WM8), lc_sub(WM4, 16), lc_add(WM4, 11), lc_sub(WM4, 4) }, 0, 3969 * 3069), // Y'
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W2, 4) },
                     WM4, { lc_add(WM4), lc_sub(WM2, 4) }, 0, 189),                                         // X'
        lincomb_dual(W8,  { lc_add(W8), lc_sub(W4, 2) },
                     WM8, { lc_add(WM8), lc_sub(WM4, 2) }, 0, 3825),                                        // s1
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W8, 8), lc_sub(W8, 6), lc_sub(W8, 5) },
                     WM4, { lc_add(WM4), lc_sub(WM8, 8), lc_sub(WM8, 6), lc_sub(WM8, 5) }),                 // X' - 352 s1
        lincomb_dual(W4,  { lc_add(W4), lc_sub(W8, 2), lc_sub(W8) },
                     WM4, { lc_add(WM4), lc_sub(WM8, 2), lc_sub(WM8) }, 4),                                 // s2
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W8, 9), lc_add(W8, 6), lc_add(W8, 3) },
                     WM2, { lc_add(WM2), lc_sub(WM8, 9), lc_add(WM8, 6), lc_add(WM8, 3) }),                 // G4 - 440 s1
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W8), lc_sub(W4, 6), lc_sub(W4, 5) },
                     WM2, { lc_add(WM2), lc_sub(WM8), lc_sub(WM4, 6), lc_sub(WM4, 5) }),                    // 4 s2 + 16 s3
        lincomb_dual(W2,  { lc_add(W2), lc_sub(W4, 2) },
                     WM2, { lc_add(WM2), lc_sub(WM4, 2) }, 4),                                              // s3
        lincomb_dual(W1,  { lc_add(W1), lc_sub(W8), lc_sub(W4), lc_sub(W2) },
                     WM1, { lc_add(WM1), lc_sub(WM8), lc_sub(WM4), lc_sub(WM2) }),                          // e4: c8, c7
        lincomb_dual_tc(WE, { lc_add_tc(WE), lc_sub_tc(WQ, 16), lc_add_tc(WQ, 4) },
                        WME, { lc_add_tc(WME), lc_sub_tc(WMQ, 16), lc_add_tc(WMQ, 4) }, 0, 4095 * 3069),   // Y
        lincomb_dual_tc(WQ, { lc_add_tc(WQ), lc_sub_tc(WH, 4) },
                        WMQ, { lc_add_tc(WMQ), lc_sub_tc(WMH, 4) }, 0, 189),                                // X
        lincomb_dual_tc(WE, { lc_add_tc(WE), lc_sub_tc(WQ, 2) },
                        WME, { lc_add_tc(WME), lc_sub_tc(WMQ, 2) }, 0, 3825),                               // d1
        lincomb_dual_tc(WQ, { lc_add_tc(WQ), lc_sub_tc(WE, 8), lc_sub_tc(WE, 6), lc_sub_tc(WE, 2) },
                        WMQ, { lc_add_tc(WMQ), lc_sub_tc(WME, 8), lc_sub_tc(WME, 6), lc_sub_tc(WME, 2) }),  // X - 324 d1
        lincomb_dual_tc(WQ, { lc_add_tc(WQ), lc_sub_tc(WE) },
                        WMQ, { lc_add_tc(WMQ), lc_sub_tc(WME) }, 4),                                        // d2
        lincomb_dual_tc(WH, { lc_add_tc(WH), lc_sub_tc(WE, 8), lc_sub_tc(WE, 4), lc_sub_tc(WE) },
                        WMH, { lc_add_tc(WMH), lc_sub_tc(WME, 8), lc_sub_tc(WME, 4), lc_sub_tc(WME) }),     // U4 - 273 d1
        lincomb_dual_tc(WH, { lc_add_tc(WH), lc_sub_tc(WQ, 6), lc_sub_tc(WQ, 2) },
                        WMH, { lc_add_tc(WMH), lc_sub_tc(WMQ, 6), lc_sub_tc(WMQ, 2) }, 4),                  // d3
        lincomb_dual(WE,  { lc_add(W8), lc_add_tc(WE) },
                     WME, { lc_add(WM8), lc_add_tc(WME) }, 1),                                              // e1: c2, c1
        lincomb_dual(W8,  { lc_add(W8), lc_sub(WE) },
                     WM8, { lc_add(WM8), lc_sub(WME) }),                                                    // e7: c14, c13
        lincomb_dual(WQ,  { lc_add(W4), lc_add_tc(WQ) },
                     WMQ, { lc_add(WM4), lc_add_tc(WMQ) }, 1),                                              // e2: c4, c3
        lincomb_dual(W4,  { lc_add(W4), lc_sub(WQ) },
                     WM4, { lc_add(WM4), lc_sub(WMQ) }),                                                    // e6: c12, c11
        lincomb_dual(WH,  { lc_add(W2), lc_add_tc(WH) },
                     WMH, { lc_add(WM2), lc_add_tc(WMH) }, 1),                                              // e3: c6, c5
        lincomb_dual(W2,  { lc_add(W2), lc_sub(WH) },
                     WM2, { lc_add(WM2), lc_sub(WMH) }),                                                    // e5: c10, c9

        // Compose: the even coefficients fill rb's pieces (their top limbs land on the next
        // piece), then the odd ones are added.
        toom_instr{ toom_op::copy_low,  R2,   WE   },
        toom_instr{ toom_op::copy_low,  R4,   WQ   },
        toom_instr{ toom_op::copy_low,  R6,   WH   },
        toom_instr{ toom_op::copy_low,  R8,   W1   },
        toom_instr{ toom_op::copy_low,  R10,  W2   },
        toom_instr{ toom_op::copy_low,  R12,  W4   },
        toom_instr{ toom_op::copy_low,  C14,  W8   },
        toom_instr{ toom_op::uadd_into, R4X,  WEHI },
        toom_instr{ toom_op::uadd_into, R6X,  WQHI },
        toom_instr{ toom_op::uadd_into, R8X,  WHHI },
        toom_instr{ toom_op::uadd_into, R10X, W1HI },
        toom_instr{ toom_op::uadd_into, R12X, W2HI },
        toom_instr{ toom_op::uadd_into, C14,  W4HI },
        toom_instr{ toom_op::uadd_into, R1,   WME  },
        toom_instr{ toom_op::uadd_into, R3,   WMQ  },
        toom_instr{ toom_op::uadd_into, R5,   WMH  },
        toom_instr{ toom_op::uadd_into, R7,   WM1  },
        toom_instr{ toom_op::uadd_into, R9,   WM2  },
        toom_instr{ toom_op::uadd_into, R11,  WM4  },
        toom_instr{ toom_op::uadd_into, R13,  WM8  },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom8h_balanced = make_toom8h_balanced();

struct toom8h_balanced_traits
{
    static constexpr auto const& plan              = toom8h_balanced.plan;
    static constexpr auto const& size_exprs        = toom8h_balanced.exprs;
    static constexpr auto const& slot_layout       = toom8h_balanced.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom8h_balanced.slab_expr;
    static constexpr size_t N = 8;
    static constexpr size_t M = 8;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
