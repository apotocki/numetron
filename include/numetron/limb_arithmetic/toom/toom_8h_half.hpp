// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <array>
#include <algorithm>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Toom-8.5 N x (17 - N) (the "half" cases of GMP's toom8h: 9 x 8, 10 x 7, 11 x 6, ...) splits u
// into N pieces and v into M = 17 - N of c limbs each, c = max(ceil(un/N), ceil(vn/M)), so that
// both top pieces are <= c. As with toom32 (toom_3x2.hpp) there are two traits for one plan;
// toom8h_half_split_by_u() picks.
template <size_t N>
inline bool toom8h_half_split_by_u(size_t un, size_t vn) noexcept
{
    constexpr size_t M = 17 - N;
    return (un + N - 1) / N >= (vn + M - 1) / M;
}

// Whether N x (17 - N) can take un x vn (un >= vn): both top pieces non-empty,
// s = un - (N - 1) c > 0 and t = vn - (M - 1) c > 0. For N = 9 that is 1 < un/vn < 9/7, for
// N = 10 9/7 < un/vn < 5/3, for N = 11 5/3 < un/vn < 11/5 (approximately, at the edges).
template <size_t N>
inline bool toom8h_half_split_fits(size_t un, size_t vn) noexcept
{
    constexpr size_t M = 17 - N;
    const size_t c = (std::max)((un + N - 1) / N, (vn + M - 1) / M);
    return un > (N - 1) * c && vn > (M - 1) * c;
}

}

namespace toom_runtime_detail {

// lincomb passes for a sum of n terms: four in the first, then the partial sum and three more.
consteval size_t toom8h_half_passes(size_t n)
{
    return n <= 4 ? 1 : 1 + (n - 4 + 2) / 3;
}

// dst <- sum of src[j] << shift[j], j < n, in toom8h_half_passes(n) lincomb passes.
template <size_t K>
consteval void toom8h_half_push_sum(std::array<toom_instr, K>& plan, size_t& i, toom_ref dst,
    const toom_ref* src, const unsigned* shift, size_t n)
{
    size_t j = 0;
    bool first = true;
    while (j < n) {
        toom_term t[4] = {};
        size_t k = 0;
        if (!first) t[k++] = lc_add(dst);
        while (k < 4 && j < n) {
            t[k++] = lc_add(src[j], shift[j]);
            ++j;
        }
        switch (k) {
        case 1: plan[i++] = lincomb(dst, { t[0] }); break;
        case 2: plan[i++] = lincomb(dst, { t[0], t[1] }); break;
        case 3: plan[i++] = lincomb(dst, { t[0], t[1], t[2] }); break;
        default: plan[i++] = lincomb(dst, { t[0], t[1], t[2], t[3] }); break;
        }
        first = false;
    }
}

// Evaluation of a P-piece operand at +-2^j (frac = false) or at +-2^-j scaled by 2^(j (P - 1))
// (frac = true): the even part into EM, the odd one into T, then E = EM + T, EM = |EM - T| with
// its sign.
template <size_t K, size_t P>
consteval void toom8h_half_push_point(std::array<toom_instr, K>& plan, size_t& i, toom_ref E, toom_ref EM, toom_ref T,
    std::array<toom_ref, P> const& a, unsigned j, bool frac)
{
    toom_ref se[P] = {}, so[P] = {};
    unsigned he[P] = {}, ho[P] = {};
    size_t ne = 0, no = 0;
    for (size_t k = 0; k < P; ++k) {
        const unsigned sh = static_cast<unsigned>(frac ? j * (P - 1 - k) : j * k);
        if (k % 2 == 0) { se[ne] = a[k]; he[ne++] = sh; }
        else            { so[no] = a[k]; ho[no++] = sh; }
    }
    toom8h_half_push_sum(plan, i, EM, se, he, ne);
    toom8h_half_push_sum(plan, i, T, so, ho, no);
    plan[i++] = toom_instr{ .op = toom_op::addsub_abs, .dst = E, .src0 = EM, .src1 = T, .dst2 = EM };
}

template <size_t N>
consteval size_t toom8h_half_plan_size()
{
    constexpr size_t M = 17 - N;
    const size_t eval = 7 * (toom8h_half_passes((N + 1) / 2) + toom8h_half_passes(N / 2) + 1)
                      + 7 * (toom8h_half_passes((M + 1) / 2) + toom8h_half_passes(M / 2) + 1);
    return eval + 16 /* products */ + 14 /* halves */ + 28 /* interpolation */ + 21 /* compose */;
}

// Toom-8.5 N x M, N + M = 17: u = a0 + a1 W + ... + a_{N-1} W^(N-1), v = b0 + ... + b_{M-1} W^(M-1),
// W = B^c, s = |a_{N-1}| = un - (N - 1) c, t = |b_{M-1}| = vn - (M - 1) c, 0 < s, t <= c. The
// product r(x) = c0 + ... + c15 x^15 is evaluated at the 15 points of the balanced plan
// (toom_8x8.hpp) -- 0, +-1, +-2, +-4, +-8, +-1/2, +-1/4, +-1/8 -- plus infinity,
// c15 = a_{N-1} b_{M-1}. The fractional points are scaled as 2^(j (N - 1)) A(+-2^-j),
// 2^(j (M - 1)) B(+-2^-j), so the products are r(+-1), r(+-2), r(+-4), r(+-8), 2^15 r(+-1/2),
// 4^15 r(+-1/4), 8^15 r(+-1/8) whatever the split -- everything after the evaluation is the
// same for every N.
//
// Interpolation (as toom_7x6.hpp does for Toom-6.5): the even coefficients c0, c2, ..., c14 are
// the balanced plan's eight; the odd ones are its c1, ..., c13 and c15, known from infinity. So
// splitting every pair into its halves also takes c15's share out and undoes the extra scaling,
// and the halves are then exactly the balanced plan's P(1), P(4), P(16), P(64), P~(4), P~(16),
// P~(64) and the same for Q:
//   Q(1)   = (r(1) - r(-1) - 2 c15) / 2            P(1)   = r(1) - Q(1) - c0 - c15
//   Q(4)   = (r(2) - r(-2) - 2^16 c15) / 4         P(4)   = (r(2) - 2 Q(4) - c0 - 2^15 c15) / 4
//   Q(16)  = (r(4) - r(-4) - 2^31 c15) / 8         P(16)  = (r(4) - 4 Q(16) - c0 - 2^30 c15) / 16
//   Q(64)  = (r(8) - r(-8) - 2^46 c15) / 16        P(64)  = (r(8) - 8 Q(64) - c0 - 2^45 c15) / 64
//   Q~(4)  = (X(1/2) - X(-1/2) - 2 c15) / 8        P~(4)  = (X(1/2) - 4 Q~(4) - c15 - 2^15 c0) / 2
//   Q~(16) = (Y(1/4) - Y(-1/4) - 2 c15) / 32       P~(16) = (Y(1/4) - 16 Q~(16) - c15 - 2^30 c0) / 4
//   Q~(64) = (Z(1/8) - Z(-1/8) - 2 c15) / 128      P~(64) = (Z(1/8) - 64 Q~(64) - c15 - 2^45 c0) / 8
// with X = 2^15 r, Y = 4^15 r, Z = 8^15 r. The 28 lincomb_dual steps that follow are the
// balanced plan's, unchanged.
//
// Scratch (29e + 14p limbs), e = c + 1, p = 2c + 2: as in the balanced plan, plus
//   W8HI  aliases W8[c, 2c + 2)     c14's limbs above rb's c-limb gap at 14c, added onto c15
// Result views: C0 = rb[0, 2c), R2 .. R12 = the 2c-limb pieces at 2c .. 12c, R14 = rb[14c, 15c),
// C15 = rb[15c, end); they cover rb, so it needn't be zeroed. The accumulation windows R1 .. R13
// and R4X .. R14X run from c, 3c, ... to the end.
template <size_t N>
consteval auto make_toom8h_half()
{
    static_assert(N >= 9 && N <= 13, "Toom-8.5 half: 9 x 8 .. 13 x 4");
    constexpr size_t M = 17 - N;

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
    auto W8HI = sb.tmp(b.add(off_w8, c),     b.add(c, b.c(2)));

    auto C0   = sb.rb(b.c(0),        two_c);
    auto R2   = sb.rb(b.mul(c, 2),   two_c);
    auto R4   = sb.rb(b.mul(c, 4),   two_c);
    auto R6   = sb.rb(b.mul(c, 6),   two_c);
    auto R8   = sb.rb(b.mul(c, 8),   two_c);
    auto R10  = sb.rb(b.mul(c, 10),  two_c);
    auto R12  = sb.rb(b.mul(c, 12),  two_c);
    auto R14  = sb.rb(b.mul(c, 14),  c);
    auto C15  = sb.rb(b.mul(c, 15),  st);
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
    auto R14X = sb.rb(b.mul(c, 14),  b.sub(rn, b.mul(c, 14)));

    std::array<toom_ref, N> u = {};
    for (size_t k = 0; k < N; ++k) u[k] = slot_builder::u(static_cast<unsigned short>(k));
    std::array<toom_ref, M> v = {};
    for (size_t k = 0; k < M; ++k) v[k] = slot_builder::v(static_cast<unsigned short>(k));

    constexpr size_t K = toom8h_half_plan_size<N>();
    std::array<toom_instr, K> plan = {};
    size_t i = 0;

    // Evaluate A (N pieces) and B (M pieces) at the 14 points.
    toom8h_half_push_point(plan, i, EA1, EAM1, T, u, 0, false);        // A(+-1)
    toom8h_half_push_point(plan, i, EA2, EAM2, T, u, 1, false);        // A(+-2)
    toom8h_half_push_point(plan, i, EA4, EAM4, T, u, 2, false);        // A(+-4)
    toom8h_half_push_point(plan, i, EA8, EAM8, T, u, 3, false);        // A(+-8)
    toom8h_half_push_point(plan, i, EAH, EAMH, T, u, 1, true);         // 2^(N-1) A(+-1/2)
    toom8h_half_push_point(plan, i, EAQ, EAMQ, T, u, 2, true);         // 4^(N-1) A(+-1/4)
    toom8h_half_push_point(plan, i, EAE, EAME, T, u, 3, true);         // 8^(N-1) A(+-1/8)
    toom8h_half_push_point(plan, i, EB1, EBM1, T, v, 0, false);
    toom8h_half_push_point(plan, i, EB2, EBM2, T, v, 1, false);
    toom8h_half_push_point(plan, i, EB4, EBM4, T, v, 2, false);
    toom8h_half_push_point(plan, i, EB8, EBM8, T, v, 3, false);
    toom8h_half_push_point(plan, i, EBH, EBMH, T, v, 1, true);
    toom8h_half_push_point(plan, i, EBQ, EBMQ, T, v, 2, true);
    toom8h_half_push_point(plan, i, EBE, EBME, T, v, 3, true);

    const std::array rest = {
        // Pointwise products (signs of the minus points kept in the WM slots); c0 and c15 into rb.
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
        toom_instr{ toom_op::umul_fixed,  C15,  u[N - 1], v[M - 1] },

        // Odd / even halves of every pair, c15's share and the extra scaling taken out: W* become
        // P(1), P(4), P(16), P(64), P~(4), P~(16), P~(64), WM* the same for Q -- the balanced
        // plan's values from here on.
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1), lc_sub(C15, 1) }, 1),
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C15) }),
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2), lc_sub(C15, 16) }, 2),
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C15, 15) }, 2),
        lincomb(WM4, { lc_add(W4), lc_sub_signed(WM4), lc_sub(C15, 31) }, 3),
        lincomb(W4,  { lc_add(W4), lc_sub(WM4, 2), lc_sub(C0), lc_sub(C15, 30) }, 4),
        lincomb(WM8, { lc_add(W8), lc_sub_signed(WM8), lc_sub(C15, 46) }, 4),
        lincomb(W8,  { lc_add(W8), lc_sub(WM8, 3), lc_sub(C0), lc_sub(C15, 45) }, 6),
        lincomb(WMH, { lc_add(WH), lc_sub_signed(WMH), lc_sub(C15, 1) }, 3),
        lincomb(WH,  { lc_add(WH), lc_sub(WMH, 2), lc_sub(C15), lc_sub(C0, 15) }, 1),
        lincomb(WMQ, { lc_add(WQ), lc_sub_signed(WMQ), lc_sub(C15, 1) }, 5),
        lincomb(WQ,  { lc_add(WQ), lc_sub(WMQ, 4), lc_sub(C15), lc_sub(C0, 30) }, 2),
        lincomb(WME, { lc_add(WE), lc_sub_signed(WME), lc_sub(C15, 1) }, 7),
        lincomb(WE,  { lc_add(WE), lc_sub(WME, 6), lc_sub(C15), lc_sub(C0, 45) }, 3),

        // Both halves, as in the balanced plan (see toom_8x8.hpp for the steps).
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
        // piece; c14's above its c-limb gap on c15), then the odd ones are added.
        toom_instr{ toom_op::copy_low,  R2,   WE   },
        toom_instr{ toom_op::copy_low,  R4,   WQ   },
        toom_instr{ toom_op::copy_low,  R6,   WH   },
        toom_instr{ toom_op::copy_low,  R8,   W1   },
        toom_instr{ toom_op::copy_low,  R10,  W2   },
        toom_instr{ toom_op::copy_low,  R12,  W4   },
        toom_instr{ toom_op::copy_low,  R14,  W8   },
        toom_instr{ toom_op::uadd_into, C15,  W8HI },
        toom_instr{ toom_op::uadd_into, R4X,  WEHI },
        toom_instr{ toom_op::uadd_into, R6X,  WQHI },
        toom_instr{ toom_op::uadd_into, R8X,  WHHI },
        toom_instr{ toom_op::uadd_into, R10X, W1HI },
        toom_instr{ toom_op::uadd_into, R12X, W2HI },
        toom_instr{ toom_op::uadd_into, R14X, W4HI },
        toom_instr{ toom_op::uadd_into, R1,   WME  },
        toom_instr{ toom_op::uadd_into, R3,   WMQ  },
        toom_instr{ toom_op::uadd_into, R5,   WMH  },
        toom_instr{ toom_op::uadd_into, R7,   WM1  },
        toom_instr{ toom_op::uadd_into, R9,   WM2  },
        toom_instr{ toom_op::uadd_into, R11,  WM4  },
        toom_instr{ toom_op::uadd_into, R13,  WM8  },
    };
    for (toom_instr const& in : rest) plan[i++] = in;
    if (i != K) throw "toom8h_half: plan size mismatch";

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

template <size_t N>
inline constexpr auto toom8h_half = make_toom8h_half<N>();

// c = ceil(un/N) (SplitByU) or ceil(vn/(17 - N))
template <size_t NV, bool SplitByU>
struct toom8h_half_traits
{
    static constexpr auto const& plan              = toom8h_half<NV>.plan;
    static constexpr auto const& size_exprs        = toom8h_half<NV>.exprs;
    static constexpr auto const& slot_layout       = toom8h_half<NV>.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom8h_half<NV>.slab_expr;
    static constexpr size_t N = NV;
    static constexpr size_t M = 17 - NV;
    static constexpr bool split_by_u = SplitByU;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
