// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "toom_core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom2_slots {

// Pointwise products/interpolation storage:
// W1 = u1 * v1  (the x=inf product, later reused as R2)
// W2 = (u0 - u1) * (v0 - v1)  (the x=-1 product, later reused in R1 formula)
inline constexpr unsigned short W1 = 0;
inline constexpr unsigned short W2 = 1;

// Interpolation outputs kept in tmp before final composition:
// R1 = (W0 + W1) - W2, where W0 lives in rb[0..)
// R2 = W1
inline constexpr unsigned short R1 = 2;
inline constexpr unsigned short R2 = 3;

// Temporary arithmetic slots:
// T0: reused for u-diff and interpolation accumulator
// T1: reused for v-diff
inline constexpr unsigned short T0 = 4;
inline constexpr unsigned short T1 = 5;

} // namespace toom2_slots

inline constexpr std::array<toom_size_expr, 11> toom2_size_exprs{
    // 0: 0
    toom_size_expr{ toom_size_expr_op::constant, 0, 0 },
    // 1: n2 == chunk
    toom_size_expr{ toom_size_expr_op::variable, static_cast<unsigned short>(toom_size_var::chunk), 0 },
    // 2: d_buf_n == max(n2, u_hi)
    toom_size_expr{ toom_size_expr_op::variable, static_cast<unsigned short>(toom_size_var::d_buf_n), 0 },
    // 3: 2*n2
    toom_size_expr{ toom_size_expr_op::mul_const, 1, 2 },
    // 4: 3*n2
    toom_size_expr{ toom_size_expr_op::mul_const, 1, 3 },
    // 5: 4*n2
    toom_size_expr{ toom_size_expr_op::mul_const, 1, 4 },
    // 6: d_buf_n + n2
    toom_size_expr{ toom_size_expr_op::add, 2, 1 },
    // 7: d_buf_n + 2*n2
    toom_size_expr{ toom_size_expr_op::add, 2, 3 },
    // 8: d_buf_n + 3*n2
    toom_size_expr{ toom_size_expr_op::add, 2, 4 },
    // 9: 2*d_buf_n
    toom_size_expr{ toom_size_expr_op::mul_const, 2, 2 },
    // 10: 2*d_buf_n + 4*n2
    toom_size_expr{ toom_size_expr_op::add, 9, 5 },
};

inline constexpr std::array<toom_tmp_layout, 6> toom2_tmp_layout{
    // Karatsuba-style scratch map:
    // [0 .. d_buf_n)                    -> T0 (d_u = |u0-u1|)
    // [d_buf_n .. d_buf_n+n2)           -> T1 (d_v = |v0-v1|)
    // [d_buf_n+n2 .. d_buf_n+3*n2)      -> W1 / R2
    // [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)  -> W2 / R1 / T0 (interpolation reuse)
    toom_tmp_layout{ toom2_slots::T0, 0, 2 },
    toom_tmp_layout{ toom2_slots::T1, 2, 1 },
    toom_tmp_layout{ toom2_slots::W1, 6, 3 },
    toom_tmp_layout{ toom2_slots::R2, 6, 3 },
    toom_tmp_layout{ toom2_slots::W2, 8, 6 },
    toom_tmp_layout{ toom2_slots::R1, 8, 6 },
};

inline constexpr std::array toom2_plan{
    // Evaluate at x=0 and x=inf directly from source chunks.
    // - W0 = u0*v0 is written to rb[0 .. 2*n2)
    // - W1 = u1*v1 is written to tmp range [d_buf_n+n2 .. d_buf_n+3*n2)
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::v, 0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::W1), make_ref(toom_mem_kind::u, 1), make_ref(toom_mem_kind::v, 1), 0 },

    // Evaluate at x=-1: (u0-u1)*(v0-v1).
    // - T0 uses [0 .. d_buf_n) as d_u
    // - T1 uses [d_buf_n .. d_buf_n+n2) as d_v
    // - W2 uses [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::u, 1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::T1), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::v, 1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::W2), make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::T1), 0 },

    // Interpolate coefficients r1/r2 (r0 is already in rb).
    // - R2 aliases W1 range [d_buf_n+n2 .. d_buf_n+3*n2)
    // - T0 is reused as accumulator in [0 .. d_buf_n)
    // - R1 aliases W2 range [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom2_slots::R2), make_ref(toom_mem_kind::tmp, toom2_slots::W1), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::W1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::R1), make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::W2), 0 },

    // Compose into result buffer rb:
    // rb += R1 << n2, rb += R2 << (2*n2), while rb already contains R0 at shift 0.
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::R1), make_ref(toom_mem_kind::tmp, 0), 1 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::R2), make_ref(toom_mem_kind::tmp, 0), 2 },
};

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan = toom2_plan;
    static constexpr auto const& size_exprs = toom2_size_exprs;
    static constexpr auto const& tmp_layout = toom2_tmp_layout;
    static constexpr unsigned short slab_size_expr_id = 10;
    static constexpr size_t N = 2;
    static constexpr size_t M = 2;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
