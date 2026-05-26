// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "toom_core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom3_slots {

inline constexpr unsigned short EA1 = 0;
inline constexpr unsigned short EAM1 = 1;
inline constexpr unsigned short EA2 = 2;
inline constexpr unsigned short EAINF = 3;

inline constexpr unsigned short EB1 = 4;
inline constexpr unsigned short EBM1 = 5;
inline constexpr unsigned short EB2 = 6;
inline constexpr unsigned short EBINF = 7;

inline constexpr unsigned short W1 = 8;
inline constexpr unsigned short WM1 = 9;
inline constexpr unsigned short W2 = 10;
inline constexpr unsigned short WINF = 11;

inline constexpr unsigned short R1 = 12;
inline constexpr unsigned short R2 = 13;
inline constexpr unsigned short R3 = 14;
inline constexpr unsigned short R4 = 15;

inline constexpr unsigned short T0 = 16;
inline constexpr unsigned short T1 = 17;
inline constexpr unsigned short T2 = 18;
inline constexpr unsigned short T3 = 19;

} // namespace toom3_slots

inline constexpr std::array<toom_size_expr, 1> toom3_size_exprs{
    toom_size_expr{ toom_size_expr_op::constant, 0, 0 },
};

inline constexpr std::array<toom_slot_layout, 21> toom3_slot_layout{
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EA1, 8, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EAM1, 10, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EA2, 12, 3 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EAINF, 15, 1 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EB1, 18, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EBM1, 20, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EB2, 22, 3 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EBINF, 25, 1 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::W1, 30, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::WM1, 34, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::W2, 38, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::WINF, 42, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R1, 48, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R2, 52, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R3, 56, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R4, 60, 2 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T0, 62, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T1, 66, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T2, 70, 4 },
    toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T3, 74, 4 },
    toom_slot_layout{ toom_mem_kind::rb, 0, 0, 2 },
};

inline constexpr std::array toom3_plan{
    // Evaluate A at x = 0, 1, -1, 2, inf.
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EAINF), make_ref(toom_mem_kind::u, 2), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::u, 1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EA1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 2), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::u, 1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EAM1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 2), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 1), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::u, 2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EA2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), 0 },

    // Evaluate B at x = 0, 1, -1, 2, inf.
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::EBINF), make_ref(toom_mem_kind::v, 2), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::v, 1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EB1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 2), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::v, 1), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EBM1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 2), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 1), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::v, 2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::EB2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), 0 },

    // Pointwise products.
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, 0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::v, 0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::EA1), make_ref(toom_mem_kind::tmp, toom3_slots::EB1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::WM1), make_ref(toom_mem_kind::tmp, toom3_slots::EAM1), make_ref(toom_mem_kind::tmp, toom3_slots::EBM1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::W2), make_ref(toom_mem_kind::tmp, toom3_slots::EA2), make_ref(toom_mem_kind::tmp, toom3_slots::EB2), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom3_slots::WINF), make_ref(toom_mem_kind::tmp, toom3_slots::EAINF), make_ref(toom_mem_kind::tmp, toom3_slots::EBINF), 0 },

    // Interpolation to recover r0..r4.
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, toom3_slots::WINF), make_ref(toom_mem_kind::tmp, 0), 0 },
    toom_instr{ toom_op::add, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::WM1), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::W1), make_ref(toom_mem_kind::tmp, toom3_slots::WM1), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::rb, 0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, 0), 16 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::W2), make_ref(toom_mem_kind::rb, 0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T2), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::R3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 3 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R1), make_ref(toom_mem_kind::tmp, toom3_slots::T2), make_ref(toom_mem_kind::tmp, toom3_slots::R3), 0 },

    // Compose to result rb.
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
    static constexpr auto const& slot_layout = toom3_slot_layout;
    static constexpr unsigned short slab_size_expr_id = 80;
    static constexpr size_t N = 3;
    static constexpr size_t M = 3;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
