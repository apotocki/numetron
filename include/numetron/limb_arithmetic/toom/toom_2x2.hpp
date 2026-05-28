// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

consteval auto make_toom2()
{
    expr_builder b;

    auto n2                  = b.v(toom_size_var::chunk);
    auto d_buf_n             = b.v(toom_size_var::d_buf_n);
    auto two_n2              = b.mul(n2, 2);
    auto three_n2            = b.mul(n2, 3);
    auto four_n2             = b.mul(n2, 4);
    auto d_plus_n2           = b.add(d_buf_n, n2);
    auto d_plus_3n2          = b.add(d_buf_n, three_n2);
    auto two_d               = b.mul(d_buf_n, 2);
    auto slab                = b.add(two_d, four_n2);
    auto v_hi_plus_u_hi      = b.add(b.v(toom_size_var::v_hi), b.v(toom_size_var::u_hi));
    auto un_plus_vn          = b.add(b.v(toom_size_var::un), b.v(toom_size_var::vn));
    auto un_plus_vn_minus_n2 = b.sub(un_plus_vn, n2);

    // Build slot layout via slot_builder.
    // Karatsuba-style scratch map:
    //   [0 .. d_buf_n)               -> d_u = |u0 - u1|
    //   [d_buf_n .. d_buf_n+n2)      -> d_v = |v0 - v1|
    //   [d_buf_n+n2 .. d_buf_n+3n2)  -> TC0 (copy of c0)
    //   [d_buf_n+3n2 .. 2d+4n2)      -> TC2 = d_u * d_v
    // Result-buffer views:
    //   rb[0 .. 2n2)                 -> C0 = u0*v0
    //   rb[2n2 .. 2n2 + u_hi+v_hi)  -> C1 = u1*v1
    //   rb[n2 .. un+vn)              -> RM (middle accumulation window)
    slot_builder sb;
    auto d_u = sb.tmp(b.c(0),     d_buf_n);
    auto d_v = sb.tmp(d_buf_n,    n2);
    auto TC0 = sb.tmp(d_plus_n2,  two_n2);
    auto TC2 = sb.tmp(d_plus_3n2, d_plus_n2);
    auto C0  = sb.rb (b.c(0),     two_n2);
    auto C1  = sb.rb (two_n2,     v_hi_plus_u_hi);
    auto RM  = sb.rb (n2,         un_plus_vn_minus_n2);

    auto u = slot_builder::u;
    auto v = slot_builder::v;

    std::array plan = {
        // Pointwise products at x=0 and x=inf.
        toom_instr{ toom_op::mul_block,   C0,  u(0), v(0) },
        toom_instr{ toom_op::mul_block,   C1,  u(1), v(1) },

        // Evaluate at x=-1: (u0-u1)*(v0-v1).
        toom_instr{ toom_op::sub,         d_u, u(0), u(1) },
        toom_instr{ toom_op::sub,         d_v, v(0), v(1) },
        toom_instr{ toom_op::mul_block,   TC2, d_u,  d_v  },

        // Middle accumulation: RM += C1 + copy(C0) - TC2.
        toom_instr{ toom_op::copy,        TC0, C0         },
        toom_instr{ toom_op::inplace_add, RM,  C1         },
        toom_instr{ toom_op::inplace_add, RM,  TC0        },
        toom_instr{ toom_op::inplace_sub, RM,  TC2        },
    };

    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom2 = make_toom2();

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan              = toom2.plan;
    static constexpr auto const& size_exprs        = toom2.exprs;
    static constexpr auto const& slot_layout       = toom2.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom2.slab_expr;
    static constexpr size_t N = 2;
    static constexpr size_t M = 2;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic