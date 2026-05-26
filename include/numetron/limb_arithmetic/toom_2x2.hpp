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
//inline constexpr unsigned short W1 = 0;


// Temporary arithmetic slots:
// d_u: reused for u-diff and interpolation accumulator
// d_v: reused for v-diff
// TC0: for u0 * v0
// TC2 = (u0 - u1) * (v0 - v1)  (the x=-1 product, later reused in R1 formula)
inline constexpr unsigned short d_u = 0;
inline constexpr unsigned short d_v = 1;
inline constexpr unsigned short TC0 = 2;
inline constexpr unsigned short TC2 = 3;

// C0 slot for direct u0*v0 product, which is used in interpolation and also serves as the final output buffer. Must be last to allow reuse of all tmp slots for interpolation.
inline constexpr unsigned short C0 = 4;
// C1 slot for direct u1*v1 product
inline constexpr unsigned short C1 = 5;
// RM slot for r + mid in interpolation, which is used in the final recomposition step. Must be last to allow reuse of all tmp slots for interpolation before
inline constexpr unsigned short RM = 6;
// R slot for final result
inline constexpr unsigned short R = 7;

} // namespace toom2_slots


inline constexpr auto toom2_size_exprs = std::array{
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
    // 11: un + vn
    toom_size_expr{ toom_size_expr_op::add, make_variable(toom_size_var::un), make_variable(toom_size_var::vn) },
    // 12: un + vn - n2
    toom_size_expr{ toom_size_expr_op::sub, 11, make_variable(toom_size_var::chunk) },
    // 13: v_hi + u_hi
    toom_size_expr{ toom_size_expr_op::add, make_variable(toom_size_var::v_hi), make_variable(toom_size_var::u_hi) },
};

inline constexpr auto toom2_slot_layout = std::array{
    // Karatsuba-style scratch map:
    // [0 .. d_buf_n)                    -> T0 (d_u = |u0-u1|)
    // [d_buf_n .. d_buf_n+n2)           -> T1 (d_v = |v0-v1|)
    // [d_buf_n+n2 .. d_buf_n+3*n2)      -> W1 / R2
    // [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)  -> W2 / R1 / T0 (interpolation reuse)
    toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::d_u, 0, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::d_v, 2, 1 },
    toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::TC0, 6, 3 },
    toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::TC2, 8, 6 },

    // reuse result buffer for pointwise products to save slots
    toom_slot_layout{ toom_mem_kind::rb, toom2_slots::C0, 0, 3 },
    toom_slot_layout{ toom_mem_kind::rb, toom2_slots::C1, 3, 13 },
    toom_slot_layout{ toom_mem_kind::rb, toom2_slots::RM, 1, 12 },
    toom_slot_layout{ toom_mem_kind::rb, toom2_slots::R, 0, 11 }
};

inline constexpr std::array toom2_plan{
    // Evaluate at x=0 and x=inf directly from source chunks.
    // - C0 = u0*v0 is written to rb[0 .. 2*n2)
    // - C1 = u1*v1 is written to rb[2*n2 .. 4*n2)
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, toom2_slots::C0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::v, 0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, toom2_slots::C1), make_ref(toom_mem_kind::u, 1), make_ref(toom_mem_kind::v, 1), 0 },

    // Evaluate at x=-1: (u0-u1)*(v0-v1).
    // - T0 uses [0 .. d_buf_n) as d_u
    // - T1 uses [d_buf_n .. d_buf_n+n2) as d_v
    // - TC2 uses [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::d_u), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::u, 1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::d_v), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::v, 1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::TC2), make_ref(toom_mem_kind::tmp, toom2_slots::d_u), make_ref(toom_mem_kind::tmp, toom2_slots::d_v), 0 },
    
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom2_slots::TC0), make_ref(toom_mem_kind::rb, toom2_slots::C0) },
    toom_instr{ toom_op::inplace_add, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::rb, toom2_slots::C1) },
    toom_instr{ toom_op::inplace_add, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::tmp, toom2_slots::TC0) },
    toom_instr{ toom_op::inplace_sub, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::tmp, toom2_slots::TC2) },
};

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan = toom2_plan;
    static constexpr auto const& size_exprs = toom2_size_exprs;
    static constexpr auto const& slot_layout = toom2_slot_layout;
    static constexpr unsigned short slab_size_expr_id = 10;
    static constexpr size_t N = 2;
    //static constexpr size_t M = 2;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
