// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

consteval auto make_toom3()
{
    // chunk = c = ceil(vn/3), u_hi = h = un - 2*c, v_hi = g = vn - 2*c.
    //
    // Capacity analysis (with 1-limb carry margin):
    //   EA1/EAM1  : u0+u1+u2      => 2c + h + 1
    //   EA2       : u0+2u1+4u2    => 3c + 4h + 1
    //   EAINF     : u2             => h
    //   EB1/EBM1  : v0+v1+v2      => 3c + 1
    //   EB2       : v0+2v1+4v2    => 7c + 1
    //   EBINF     : v2             => g
    //   W1/WM1    : EA*EB pts 1,-1 => 5c + h + 2
    //   W2        : EA2*EB2        => 10c + 4h + 2
    //   WINF      : EAINF*EBINF   => h + g
    //   R4        : copy of WINF   => h + g
    //   R1..R3,T* : bounded by un+vn = 4c+h+g+1

    expr_builder b;

    auto c = b.v(toom_size_var::chunk);
    auto h = b.v(toom_size_var::u_hi);
    auto g = b.v(toom_size_var::v_hi);

    auto cap_ea1   = b.add(b.add(b.mul(c, 2), h), b.c(1));
    auto cap_ea2   = b.add(b.add(b.mul(c, 3), b.mul(h, 4)), b.c(1));
    auto cap_eainf = b.max2(h, b.c(1));
    auto cap_eb1   = b.add(b.mul(c, 3), b.c(1));
    auto cap_eb2   = b.add(b.mul(c, 7), b.c(1));
    auto cap_ebinf = b.max2(g, b.c(1));
    auto cap_w1    = b.add(b.add(b.mul(c, 5), h), b.c(2));
    auto cap_w2    = b.add(b.add(b.mul(c, 10), b.mul(h, 4)), b.c(2));
    auto cap_winf  = b.max2(b.add(h, g), b.c(1));
    auto cap_r     = b.add(b.add(b.add(b.mul(c, 4), h), g), b.c(1));
    auto cap_r4    = cap_winf;

    auto off_ea1   = b.c(0);
    auto off_eam1  = b.add(off_ea1,   cap_ea1);
    auto off_ea2   = b.add(off_eam1,  cap_ea1);
    auto off_eainf = b.add(off_ea2,   cap_ea2);
    auto off_eb1   = b.add(off_eainf, cap_eainf);
    auto off_ebm1  = b.add(off_eb1,   cap_eb1);
    auto off_eb2   = b.add(off_ebm1,  cap_eb1);
    auto off_ebinf = b.add(off_eb2,   cap_eb2);
    auto off_w1    = b.add(off_ebinf, cap_ebinf);
    auto off_wm1   = b.add(off_w1,    cap_w1);
    auto off_w2    = b.add(off_wm1,   cap_w1);
    auto off_winf  = b.add(off_w2,    cap_w2);
    auto off_r1    = b.add(off_winf,  cap_winf);
    auto off_r2    = b.add(off_r1,    cap_r);
    auto off_r3    = b.add(off_r2,    cap_r);
    auto off_r4    = b.add(off_r3,    cap_r);
    auto off_t0    = b.add(off_r4,    cap_r4);
    auto off_t1    = b.add(off_t0,    cap_r);
    auto off_t2    = b.add(off_t1,    cap_r);
    auto off_t3    = b.add(off_t2,    cap_r);
    auto slab      = b.add(off_t3,    cap_r);

    auto two_c = b.mul(c, 2);

    slot_builder sb;
    // tmp slots
    auto EA1   = sb.tmp(off_ea1,   cap_ea1);
    auto EAM1  = sb.tmp(off_eam1,  cap_ea1);
    auto EA2   = sb.tmp(off_ea2,   cap_ea2);
    auto EAINF = sb.tmp(off_eainf, cap_eainf);
    auto EB1   = sb.tmp(off_eb1,   cap_eb1);
    auto EBM1  = sb.tmp(off_ebm1,  cap_eb1);
    auto EB2   = sb.tmp(off_eb2,   cap_eb2);
    auto EBINF = sb.tmp(off_ebinf, cap_ebinf);
    auto W1    = sb.tmp(off_w1,    cap_w1);
    auto WM1   = sb.tmp(off_wm1,   cap_w1);
    auto W2    = sb.tmp(off_w2,    cap_w2);
    auto WINF  = sb.tmp(off_winf,  cap_winf);
    auto R1    = sb.tmp(off_r1,    cap_r);
    auto R2    = sb.tmp(off_r2,    cap_r);
    auto R3    = sb.tmp(off_r3,    cap_r);
    auto R4    = sb.tmp(off_r4,    cap_r4);
    auto T0    = sb.tmp(off_t0,    cap_r);
    auto T1    = sb.tmp(off_t1,    cap_r);
    auto T2    = sb.tmp(off_t2,    cap_r);
    auto T3    = sb.tmp(off_t3,    cap_r);
    // rb slot: r0 = u[0]*v[0]
    auto R0    = sb.rb (b.c(0),    two_c);

    auto u = slot_builder::u;
    auto v = slot_builder::v;

    std::array plan = {
        // Evaluate A at x = 0, 1, -1, 2, inf.
        toom_instr{ toom_op::copy,       EAINF, u(2)                    },
        toom_instr{ toom_op::add,        T0,    u(0),  u(1)             },
        toom_instr{ toom_op::add,        EA1,   T0,    u(2)             },
        toom_instr{ toom_op::sub,        T0,    u(0),  u(1)             },
        toom_instr{ toom_op::add,        EAM1,  T0,    u(2)             },
        toom_instr{ toom_op::mul_small,  T0,    u(1),  {}, 2            },
        toom_instr{ toom_op::mul_small,  T1,    u(2),  {}, 4            },
        toom_instr{ toom_op::add,        T0,    u(0),  T0               },
        toom_instr{ toom_op::add,        EA2,   T0,    T1               },

        // Evaluate B at x = 0, 1, -1, 2, inf.
        toom_instr{ toom_op::copy,       EBINF, v(2)                    },
        toom_instr{ toom_op::add,        T0,    v(0),  v(1)             },
        toom_instr{ toom_op::add,        EB1,   T0,    v(2)             },
        toom_instr{ toom_op::sub,        T0,    v(0),  v(1)             },
        toom_instr{ toom_op::add,        EBM1,  T0,    v(2)             },
        toom_instr{ toom_op::mul_small,  T0,    v(1),  {}, 2            },
        toom_instr{ toom_op::mul_small,  T1,    v(2),  {}, 4            },
        toom_instr{ toom_op::add,        T0,    v(0),  T0               },
        toom_instr{ toom_op::add,        EB2,   T0,    T1               },

        // Pointwise products.
        toom_instr{ toom_op::mul_block,  R0,    u(0),  v(0)             },
        toom_instr{ toom_op::mul_block,  W1,    EA1,   EB1              },
        toom_instr{ toom_op::mul_block,  WM1,   EAM1,  EBM1             },
        toom_instr{ toom_op::mul_block,  W2,    EA2,   EB2              },
        toom_instr{ toom_op::mul_block,  WINF,  EAINF, EBINF            },

        // Interpolation to recover r0..r4.
        toom_instr{ toom_op::copy,         R4,  WINF                    },
        toom_instr{ toom_op::add,          T0,  W1,    WM1              },
        toom_instr{ toom_op::divexact_small, T1, T0,   {}, 2            },
        toom_instr{ toom_op::sub,          T0,  W1,    WM1              },
        toom_instr{ toom_op::divexact_small, T2, T0,   {}, 2            },
        toom_instr{ toom_op::sub,          T0,  T1,    R0               },
        toom_instr{ toom_op::sub,          R2,  T0,    R4               },
        toom_instr{ toom_op::mul_small,    T0,  R4,    {}, 16           },
        toom_instr{ toom_op::sub,          T3,  W2,    R0               },
        toom_instr{ toom_op::sub,          T3,  T3,    T0               },
        toom_instr{ toom_op::mul_small,    T0,  R2,    {}, 4            },
        toom_instr{ toom_op::sub,          T3,  T3,    T0               },
        toom_instr{ toom_op::divexact_small, T3, T3,   {}, 2            },
        toom_instr{ toom_op::sub,          T0,  T3,    T2               },
        toom_instr{ toom_op::divexact_small, R3, T0,   {}, 3            },
        toom_instr{ toom_op::sub,          R1,  T2,    R3               },

        // Compose to result rb.
        toom_instr{ toom_op::compose_shifted, R0, R1,  {}, 1            },
        toom_instr{ toom_op::compose_shifted, R0, R2,  {}, 2            },
        toom_instr{ toom_op::compose_shifted, R0, R3,  {}, 3            },
        toom_instr{ toom_op::compose_shifted, R0, R4,  {}, 4            },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom3 = make_toom3();

template <>
struct toom_stage_traits<3, 3>
{
    static constexpr auto const& plan              = toom3.plan;
    static constexpr auto const& size_exprs        = toom3.exprs;
    static constexpr auto const& slot_layout       = toom3.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom3.slab_expr;
    static constexpr size_t N = 3;
    static constexpr size_t M = 3;
};

// Balanced Toom-3 on the fixed-width ops: the same algorithm and memory layout as the
// hand-written detail::umul_toom3_impl() (umul_toom3.hpp), expressed as a plan, with the
// evaluation at 2 and the interpolation fused into lincomb passes.
//
// Split by u (split_by_u): c = ceil(un/3), s = un - 2c = |u2|, t = vn - 2c = |v2|,
// 0 < t <= s <= c (the dispatch only takes this plan when detail::toom3_split_fits(un, vn)).
//
// Scratch (12c + 12 limbs), e = c + 1, p = 2c + 2:
//   EA1 EAM1 EA2 EB1 EBM1 EB2   6 x e     A(1), |A(-1)|, A(2) and the same for B
//   W1 WM1 W2                   3 x p     r(1), |r(-1)|, r(2), then the interpolated c2, c1, c3
//   W1HI  aliases W1[2c, 2c+2)            the part of c2 above rb's 2c-limb gap
// Result views: C0 = rb[0, 2c), R2 = rb[2c, 4c), C4 = rb[4c, end), R1 = rb[c, end),
// R3 = rb[3c, end). rb is written in full by the plan (C0, R2, C4 cover it), so it doesn't
// need zeroing first.
consteval auto make_toom3_balanced()
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
    auto W1HI  = sb.tmp(b.add(off_w1, two_c), b.c(2));

    auto C0    = sb.rb(b.c(0),          two_c);
    auto R2    = sb.rb(two_c,           two_c);
    auto C4    = sb.rb(b.mul(c, 4),     st);
    auto R1    = sb.rb(c,               b.sub(rn, c));
    auto R3    = sb.rb(b.mul(c, 3),     b.sub(rn, b.mul(c, 3)));

    // Plain refs, see make_toom4_balanced() (GCC and consteval calls through a pointer).
    constexpr std::array<toom_ref, 3> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2) };
    constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };

    std::array plan = {
        // Evaluate A at 1, -1 (one fused op: even parts a0 + a2, odd part a1), then at 2.
        toom_instr{ .op = toom_op::eval_pm1, .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = u[1] },
        lincomb(EA2, { lc_add(u[0]), lc_add(u[1], 1), lc_add(u[2], 2) }),                  // A(2)

        // Same for B.
        toom_instr{ .op = toom_op::eval_pm1, .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = v[1] },
        lincomb(EB2, { lc_add(v[0]), lc_add(v[1], 1), lc_add(v[2], 2) }),

        // Pointwise products; c0 and c4 straight into the result.
        toom_instr{ toom_op::umul_fixed,  W1,   EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed,  WM1,  EAM1, EBM1 },   // sign of r(-1) kept in WM1
        toom_instr{ toom_op::umul_fixed,  W2,   EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed,  C0,   u[0], v[0] },
        toom_instr{ toom_op::umul_fixed,  C4,   u[2], v[2] },

        // Interpolation, one lincomb pass per line.
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

static constexpr auto toom3_balanced = make_toom3_balanced();

struct toom3_balanced_traits
{
    static constexpr auto const& plan              = toom3_balanced.plan;
    static constexpr auto const& size_exprs        = toom3_balanced.exprs;
    static constexpr auto const& slot_layout       = toom3_balanced.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom3_balanced.slab_expr;
    static constexpr size_t N = 3;
    static constexpr size_t M = 3;
    static constexpr bool split_by_u = true;
    static constexpr bool zero_result = false;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic