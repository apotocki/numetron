// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "toom_core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom2_slots {

inline constexpr unsigned short A0 = 0;
inline constexpr unsigned short A1 = 1;
inline constexpr unsigned short B0 = 2;
inline constexpr unsigned short B1 = 3;

inline constexpr unsigned short W0 = 4;
inline constexpr unsigned short W1 = 5;
inline constexpr unsigned short W2 = 6;

inline constexpr unsigned short R0 = 7;
inline constexpr unsigned short R1 = 8;
inline constexpr unsigned short R2 = 9;

inline constexpr unsigned short T0 = 10;
inline constexpr unsigned short T1 = 11;

} // namespace toom2_slots

inline constexpr std::array<toom_size_expr, 1> toom2_size_exprs{
    toom_size_expr{ toom_size_expr_op::constant, 0, 0 },
};

inline constexpr std::array<toom_tmp_layout, 12> toom2_tmp_layout{
    toom_tmp_layout{ toom2_slots::A0, 0, 1 },
    toom_tmp_layout{ toom2_slots::A1, 1, 1 },
    toom_tmp_layout{ toom2_slots::B0, 2, 1 },
    toom_tmp_layout{ toom2_slots::B1, 3, 1 },
    toom_tmp_layout{ toom2_slots::W0, 4, 2 },
    toom_tmp_layout{ toom2_slots::W1, 6, 2 },
    toom_tmp_layout{ toom2_slots::W2, 8, 2 },
    toom_tmp_layout{ toom2_slots::R0, 4, 2 },
    toom_tmp_layout{ toom2_slots::R1, 10, 2 },
    toom_tmp_layout{ toom2_slots::R2, 6, 2 },
    toom_tmp_layout{ toom2_slots::T0, 10, 2 },
    toom_tmp_layout{ toom2_slots::T1, 10, 2 },
};

inline constexpr std::array toom2_plan{
    toom_instr{ toom_op::load_part_u, make_ref(toom_mem_kind::tmp, toom2_slots::A0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, 0), 0, 0, 0 },
    toom_instr{ toom_op::load_part_u, make_ref(toom_mem_kind::tmp, toom2_slots::A1), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, 0), 1, 0, 0 },
    toom_instr{ toom_op::load_part_v, make_ref(toom_mem_kind::tmp, toom2_slots::B0), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, 0), 0, 0, 0 },
    toom_instr{ toom_op::load_part_v, make_ref(toom_mem_kind::tmp, toom2_slots::B1), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, 0), 1, 0, 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::W0), make_ref(toom_mem_kind::tmp, toom2_slots::A0), make_ref(toom_mem_kind::tmp, toom2_slots::B0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::W1), make_ref(toom_mem_kind::tmp, toom2_slots::A1), make_ref(toom_mem_kind::tmp, toom2_slots::B1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::A0), make_ref(toom_mem_kind::tmp, toom2_slots::A1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::T1), make_ref(toom_mem_kind::tmp, toom2_slots::B0), make_ref(toom_mem_kind::tmp, toom2_slots::B1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::W2), make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::T1), 0 },

    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom2_slots::R0), make_ref(toom_mem_kind::tmp, toom2_slots::W0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom2_slots::R2), make_ref(toom_mem_kind::tmp, toom2_slots::W1), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::W0), make_ref(toom_mem_kind::tmp, toom2_slots::W1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::R1), make_ref(toom_mem_kind::tmp, toom2_slots::T0), make_ref(toom_mem_kind::tmp, toom2_slots::W2), 0 },

    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::R0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::R1), make_ref(toom_mem_kind::tmp, 0), 1 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom2_slots::R2), make_ref(toom_mem_kind::tmp, 0), 2 },
};

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan = toom2_plan;
    static constexpr auto const& size_exprs = toom2_size_exprs;
    static constexpr auto const& tmp_layout = toom2_tmp_layout;
    static constexpr unsigned short slab_size_expr_id = 12;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
