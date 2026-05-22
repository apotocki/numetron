// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "toom_core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom3_slots {

inline constexpr unsigned short A0 = 0;
inline constexpr unsigned short A1 = 1;
inline constexpr unsigned short A2 = 2;
inline constexpr unsigned short B0 = 3;
inline constexpr unsigned short B1 = 4;
inline constexpr unsigned short B2 = 5;

inline constexpr unsigned short EA0 = 6;
inline constexpr unsigned short EA1 = 7;
inline constexpr unsigned short EAM1 = 8;
inline constexpr unsigned short EA2 = 9;
inline constexpr unsigned short EAINF = 10;

inline constexpr unsigned short EB0 = 11;
inline constexpr unsigned short EB1 = 12;
inline constexpr unsigned short EBM1 = 13;
inline constexpr unsigned short EB2 = 14;
inline constexpr unsigned short EBINF = 15;

inline constexpr unsigned short W0 = 16;
inline constexpr unsigned short W1 = 17;
inline constexpr unsigned short WM1 = 18;
inline constexpr unsigned short W2 = 19;
inline constexpr unsigned short WINF = 20;

inline constexpr unsigned short R0 = 21;
inline constexpr unsigned short R1 = 22;
inline constexpr unsigned short R2 = 23;
inline constexpr unsigned short R3 = 24;
inline constexpr unsigned short R4 = 25;

inline constexpr unsigned short T0 = 26;
inline constexpr unsigned short T1 = 27;
inline constexpr unsigned short T2 = 28;
inline constexpr unsigned short T3 = 29;

} // namespace toom3_slots

inline constexpr std::array<toom_size_expr, 1> toom3_size_exprs{
    toom_size_expr{ toom_size_expr_op::constant, 0, 0 },
};

inline constexpr std::array<toom_tmp_layout, 30> toom3_tmp_layout{
    toom_tmp_layout{ toom3_slots::A0, 0, 1 },
    toom_tmp_layout{ toom3_slots::A1, 1, 1 },
    toom_tmp_layout{ toom3_slots::A2, 2, 1 },
    toom_tmp_layout{ toom3_slots::B0, 3, 1 },
    toom_tmp_layout{ toom3_slots::B1, 4, 1 },
    toom_tmp_layout{ toom3_slots::B2, 5, 1 },
    toom_tmp_layout{ toom3_slots::EA0, 6, 2 },
    toom_tmp_layout{ toom3_slots::EA1, 8, 2 },
    toom_tmp_layout{ toom3_slots::EAM1, 10, 2 },
    toom_tmp_layout{ toom3_slots::EA2, 12, 3 },
    toom_tmp_layout{ toom3_slots::EAINF, 15, 1 },
    toom_tmp_layout{ toom3_slots::EB0, 16, 2 },
    toom_tmp_layout{ toom3_slots::EB1, 18, 2 },
    toom_tmp_layout{ toom3_slots::EBM1, 20, 2 },
    toom_tmp_layout{ toom3_slots::EB2, 22, 3 },
    toom_tmp_layout{ toom3_slots::EBINF, 25, 1 },
    toom_tmp_layout{ toom3_slots::W0, 26, 4 },
    toom_tmp_layout{ toom3_slots::W1, 30, 4 },
    toom_tmp_layout{ toom3_slots::WM1, 34, 4 },
    toom_tmp_layout{ toom3_slots::W2, 38, 4 },
    toom_tmp_layout{ toom3_slots::WINF, 42, 2 },
    toom_tmp_layout{ toom3_slots::R0, 44, 4 },
    toom_tmp_layout{ toom3_slots::R1, 48, 4 },
    toom_tmp_layout{ toom3_slots::R2, 52, 4 },
    toom_tmp_layout{ toom3_slots::R3, 56, 4 },
    toom_tmp_layout{ toom3_slots::R4, 60, 2 },
    toom_tmp_layout{ toom3_slots::T0, 62, 4 },
    toom_tmp_layout{ toom3_slots::T1, 66, 4 },
    toom_tmp_layout{ toom3_slots::T2, 70, 4 },
    toom_tmp_layout{ toom3_slots::T3, 74, 4 },
};

inline constexpr std::array toom3_plan{
    toom_instr{ toom_op::load_part_u, make_ref(toom_mem_kind::tmp, toom3_slots::A0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, 0), 0, 0, 0 },
    toom_instr{ toom_op::load_part_u, make_ref(toom_mem_kind::tmp, toom3_slots::A1), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, 0), 1, 0, 0 },
    toom_instr{ toom_op::load_part_u, make_ref(toom_mem_kind::tmp, toom3_slots::A2), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, 0), 2, 0, 0 },
    toom_instr{ toom_op::load_part_v, make_ref(toom_mem_kind::tmp, toom3_slots::B0), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, 0), 0, 0, 0 },
    toom_instr{ toom_op::load_part_v, make_ref(toom_mem_kind::tmp, toom3_slots::B1), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, 0), 1, 0, 0 },
    toom_instr{ toom_op::load_part_v, make_ref(toom_mem_kind::tmp, toom3_slots::B2), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, 0), 2, 0, 0 },
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EA0), make_ref(toom_mem_kind::tmp, toom3_slots::A0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EAINF), make_ref(toom_mem_kind::tmp, toom3_slots::A2), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A0), make_ref(toom_mem_kind::tmp, toom3_slots::A1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EA1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A2), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A0), make_ref(toom_mem_kind::tmp, toom3_slots::A1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EAM1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A2), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A1), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::tmp, toom3_slots::A2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::A0), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EA2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), 0 },

    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EB0), make_ref(toom_mem_kind::tmp, toom3_slots::B0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EBINF), make_ref(toom_mem_kind::tmp, toom3_slots::B2), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B0), make_ref(toom_mem_kind::tmp, toom3_slots::B1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EB1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B2), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B0), make_ref(toom_mem_kind::tmp, toom3_slots::B1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EBM1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B2), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B1), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::tmp, toom3_slots::B2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::B0), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EB2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), 0 },

    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::W0), make_ref(toom_mem_kind::tmp, toom3_slots::EA0), make_ref(toom_mem_kind::tmp, toom3_slots::EB0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::EA1), make_ref(toom_mem_kind::tmp, toom3_slots::EB1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::WM1), make_ref(toom_mem_kind::tmp, toom3_slots::EAM1), make_ref(toom_mem_kind::tmp, toom3_slots::EBM1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::W2), make_ref(toom_mem_kind::tmp, toom3_slots::EA2), make_ref(toom_mem_kind::tmp, toom3_slots::EB2), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::WINF), make_ref(toom_mem_kind::tmp, toom3_slots::EAINF), make_ref(toom_mem_kind::tmp, toom3_slots::EBINF), 0 },

    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::R0), make_ref(toom_mem_kind::tmp, toom3_slots::W0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, toom3_slots::WINF), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::WM1), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::WM1), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::tmp, toom3_slots::R0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, 0), 16 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::W2), make_ref(toom_mem_kind::tmp, toom3_slots::R0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T2), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::R3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 3 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R1), make_ref(toom_mem_kind::tmp, toom3_slots::T2), make_ref(toom_mem_kind::tmp, toom3_slots::R3), 0 },

    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom3_slots::R0), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom3_slots::R1), make_ref(toom_mem_kind::tmp, 0), 1 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom3_slots::R3), make_ref(toom_mem_kind::tmp, 0), 3 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, 0), 4 },
};

template <>
struct toom_stage_traits<3, 3>
{
    static constexpr auto const& plan = toom3_plan;
    static constexpr auto const& size_exprs = toom3_size_exprs;
    static constexpr auto const& tmp_layout = toom3_tmp_layout;
    static constexpr unsigned short slab_size_expr_id = 80;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
