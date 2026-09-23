// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace detail {

// Whether the balanced Toom-4 plan can take un x vn (un >= vn): it splits u into quarters of
// c = ceil(un/4) limbs and needs v to reach into its top quarter (vn > 3c), which also makes
// u's top quarter non-empty.
inline bool toom4_split_fits(size_t un, size_t vn) noexcept
{
    return vn > 3 * ((un + 3) / 4);
}

}

namespace toom_runtime_detail {

// Balanced Toom-4 (split_by_u): u = a0 + a1 W + a2 W^2 + a3 W^3, v likewise with b0..b3,
// W = B^c, c = ceil(un/4), s = un - 3c = |a3|, t = vn - 3c = |b3|, 0 < t <= s <= c.
// The product r(x) = c0 + c1 x + ... + c6 x^6 is evaluated at 0, 1, -1, 2, -2, 1/2 and inf:
//   A(+-1) = (a0 + a2) +- (a1 + a3)                    eval_pm1 over a ready odd sum
//   A(+-2) = (a0 + 4 a2) +- 2 (a1 + 4 a3)              addsub_abs
//   8 A(1/2) = 8 a0 + 4 a1 + 2 a2 + a3                 one lincomb
// so the seven point values are c0 = a0 b0, c6 = a3 b3 (both straight into rb), r(1), |r(-1)|,
// r(2), |r(-2)| and 64 r(1/2). Interpolation, each line one lincomb pass (all results
// non-negative, shifts and divisions exact):
//   D1 = (r(1) - r(-1)) / 2                     = c1 + c3 + c5
//   E1 = r(1) - D1 - c0 - c6                    = c2 + c4
//   D2 = (r(2) - r(-2)) / 4                     = c1 + 4c3 + 16c5
//   E2 = (r(2) - 2 D2 - c0 - 64 c6) / 4         = c2 + 4c4
//   c4 = (E2 - E1) / 3,  c2 = E1 - c4
//   G  = 64 r(1/2) - 64 c0 - 16 c2 - c6         = 32c1 + 8c3 + 4c4 + 2c5
//   T1 = (G - 4 c4 - 2 D1) / 2 / 3              = 5c1 + c3
//   T2 = (D2 - D1) / 3                          = c3 + 5c5
//   c3 = (D1 + 4 D1 - T1 - T2) / 3,  c1 = (T1 - c3) / 5,  c5 = (T2 - c3) / 5
// and c2, c4 fill rb's untouched middle [2c, 6c) (their top limbs land on the next piece),
// then c1, c3, c5 are added at c, 3c, 5c.
//
// Scratch (12e + 5p limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EAM2 EAH, EB1 EBM1 EB2 EBM2 EBH   10 x e    point values of A and B
//   TA TB                                          2 x e     evaluation temporaries
//   W1 WM1 W2 WM2 WH                               5 x p     r(1), |r(-1)|, r(2), |r(-2)|, 64 r(1/2),
//                                                            ending as c2, c3, c4, c5, c1
//   W1HI, W2HI alias W1 / W2 [2c, 2c + 2)                    the parts of c2 / c4 above 2c
// Result views: C0 = rb[0, 2c), C6 = rb[6c, end), R2 = rb[2c, 4c), R4 = rb[4c, 6c), and the
// accumulation windows R1, R3, R4X, R5 from c, 3c, 4c, 5c to the end. C0, R2, R4 and C6 cover
// all of rb, so it needn't be zeroed first.
consteval auto make_toom4_balanced()
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

    // Plain refs rather than calls through a pointer to slot_builder::u/v: GCC rejects such calls
    // inside the arguments of lincomb() ("does not designate a 'constexpr' function").
    constexpr std::array<toom_ref, 4> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2), slot_builder::u(3) };
    constexpr std::array<toom_ref, 4> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2), slot_builder::v(3) };

    std::array plan = {
        // Evaluate A at +-1, +-2, 1/2.
        toom_instr{ toom_op::uadd,        TA,   u[1], u[3]    },
        toom_instr{ .op = toom_op::eval_pm1,   .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = TA },
        lincomb(EAM2, { lc_add(u[0]), lc_add(u[2], 2) }),                                    // a0 + 4 a2
        lincomb(TA,   { lc_add(u[1], 1), lc_add(u[3], 3) }),                                 // 2 (a1 + 4 a3)
        toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = TA, .dst2 = EAM2 },
        lincomb(EAH,  { lc_add(u[0], 3), lc_add(u[1], 2), lc_add(u[2], 1), lc_add(u[3]) }),  // 8 A(1/2)

        // Same for B.
        toom_instr{ toom_op::uadd,        TB,   v[1], v[3]    },
        toom_instr{ .op = toom_op::eval_pm1,   .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = TB },
        lincomb(EBM2, { lc_add(v[0]), lc_add(v[2], 2) }),
        lincomb(TB,   { lc_add(v[1], 1), lc_add(v[3], 3) }),
        toom_instr{ .op = toom_op::addsub_abs, .dst = EB2, .src0 = EBM2, .src1 = TB, .dst2 = EBM2 },
        lincomb(EBH,  { lc_add(v[0], 3), lc_add(v[1], 2), lc_add(v[2], 1), lc_add(v[3]) }),

        // Pointwise products; c0 and c6 straight into the result.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1     },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1    },   // sign of r(-1) kept in WM1
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2     },
        toom_instr{ toom_op::umul_fixed,  WM2,  EAM2, EBM2    },   // sign of r(-2) kept in WM2
        toom_instr{ toom_op::umul_fixed,  WH,   EAH,  EBH     },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0]    },
        toom_instr{ toom_op::umul_fixed,  C6,   u[3], v[3]    },

        // Even coefficients (the +-1 and +-2 pairs split into odd and even parts on the way).
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),                              // D1
        lincomb(W1,  { lc_add(W1), lc_sub(WM1), lc_sub(C0), lc_sub(C6) }),                // E1 = c2 + c4
        lincomb(WM2, { lc_add(W2), lc_sub_signed(WM2) }, 2),                              // D2
        lincomb(W2,  { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C6, 6) }, 2),       // E2 = c2 + 4 c4
        lincomb(W2,  { lc_add(W2), lc_sub(W1) }, 0, 3),                                   // c4
        toom_instr{ toom_op::usub,        W1,   W1,   W2      },                          // c2

        // Odd coefficients.
        lincomb(WH,  { lc_add(WH), lc_sub(C0, 6), lc_sub(W1, 4), lc_sub(C6) }),           // 2 H + 4 c4
        lincomb(WH,  { lc_add(WH), lc_sub(W2, 2), lc_sub(WM1, 1) }, 1, 3),                // T1 = 5 c1 + c3
        lincomb(WM2, { lc_add(WM2), lc_sub(WM1) }, 0, 3),                                 // T2 = c3 + 5 c5
        lincomb(WM1, { lc_add(WM1), lc_add(WM1, 2), lc_sub(WH), lc_sub(WM2) }, 0, 3),     // c3
        lincomb(WH,  { lc_add(WH), lc_sub(WM1) }, 0, 5),                                  // c1
        lincomb(WM2, { lc_add(WM2), lc_sub(WM1) }, 0, 5),                                 // c5

        // Compose: c2 and c4 fill rb[2c, 6c) (their tops land on the next piece), then c1, c3,
        // c5 are added.
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

static constexpr auto toom4_balanced = make_toom4_balanced();

struct toom4_balanced_traits
{
    static constexpr auto const& plan              = toom4_balanced.plan;
    static constexpr auto const& size_exprs        = toom4_balanced.exprs;
    static constexpr auto const& slot_layout       = toom4_balanced.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom4_balanced.slab_expr;
    static constexpr size_t N = 4;
    static constexpr size_t M = 4;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
