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

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic