// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom3_slots {

// Temporary slots inside scratch slab.
// EA*: A(x) evaluations at points {1, -1, 2, inf}.
// EB*: B(x) evaluations at points {1, -1, 2, inf}.
// W*:  pointwise products at matching evaluation points.
// R*:  interpolation outputs r1..r4 (r0 is rb slot below).
// T*:  interpolation temporaries.
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

// View into result buffer rb for r0 = u0 * v0.
// Keep rb slot ids disjoint from tmp ids because both kinds share one slot array.
inline constexpr unsigned short R0 = 20;

} // namespace toom3_slots

template <typename ExprPackT, typename SlotLayoutT>
struct toom3_spec_t
{
    ExprPackT exprs{};
    SlotLayoutT slot_layout{};
    expr_handle slab_expr{};
};

template <typename ExprPackT, typename SlotLayoutT>
consteval auto make_toom3_spec_t(ExprPackT exprs, SlotLayoutT slot_layout, expr_handle slab_expr)
{
    return toom3_spec_t<ExprPackT, SlotLayoutT>{ exprs, slot_layout, slab_expr };
}

consteval auto make_toom3_spec()
{
    // Build named size expressions once and reuse typed handles in slot layout.
    expr_builder<64> b;

    auto chunk = b.v(toom_size_var::chunk);
    auto one = chunk;
    auto two = b.mul(chunk, 2);
    auto three = b.mul(chunk, 3);
    auto four = b.mul(chunk, 4);

    auto off8 = b.mul(chunk, 8);
    auto off10 = b.mul(chunk, 10);
    auto off12 = b.mul(chunk, 12);
    auto off15 = b.mul(chunk, 15);
    auto off18 = b.mul(chunk, 18);
    auto off20 = b.mul(chunk, 20);
    auto off22 = b.mul(chunk, 22);
    auto off25 = b.mul(chunk, 25);
    auto off30 = b.mul(chunk, 30);
    auto off34 = b.mul(chunk, 34);
    auto off38 = b.mul(chunk, 38);
    auto off42 = b.mul(chunk, 42);
    auto off48 = b.mul(chunk, 48);
    auto off52 = b.mul(chunk, 52);
    auto off56 = b.mul(chunk, 56);
    auto off60 = b.mul(chunk, 60);
    auto off62 = b.mul(chunk, 62);
    auto off66 = b.mul(chunk, 66);
    auto off70 = b.mul(chunk, 70);
    auto off74 = b.mul(chunk, 74);
    auto slab = b.mul(chunk, 80);

    return make_toom3_spec_t(b.finish(slab), std::array{
        // Scratch map (tmp):
        // [8n .. 17n)  : A/B evaluations
        // [30n .. 44n) : pointwise products W*
        // [48n .. 62n) : interpolation outputs R1..R4
        // [62n .. 78n) : interpolation temporaries T0..T3
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EA1, off8, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EAM1, off10, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EA2, off12, three },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EAINF, off15, one },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EB1, off18, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EBM1, off20, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EB2, off22, three },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::EBINF, off25, one },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::W1, off30, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::WM1, off34, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::W2, off38, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::WINF, off42, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R1, off48, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R2, off52, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R3, off56, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::R4, off60, two },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T0, off62, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T1, off66, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T2, off70, four },
        toom_slot_layout{ toom_mem_kind::tmp, toom3_slots::T3, off74, four },

        // Result-buffer views used by rb-kind refs in the plan.
        toom_slot_layout{ toom_mem_kind::rb, toom3_slots::R0, 0, two },
    }, slab);
}

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
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, toom3_slots::R0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::v, 0), 0 },
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
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T1), make_ref(toom_mem_kind::rb, toom3_slots::R0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, 0), 16 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::W2), make_ref(toom_mem_kind::rb, toom3_slots::R0), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::mul_small, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, 0), 4 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, toom3_slots::T3), make_ref(toom_mem_kind::tmp, toom3_slots::T2), 0 },
    toom_instr{ toom_op::divexact_small, make_ref(toom_mem_kind::tmp, toom3_slots::R3), make_ref(toom_mem_kind::tmp, toom3_slots::T0), make_ref(toom_mem_kind::tmp, 0), 3 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom3_slots::R1), make_ref(toom_mem_kind::tmp, toom3_slots::T2), make_ref(toom_mem_kind::tmp, toom3_slots::R3), 0 },

    // Compose to result rb.
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, toom3_slots::R0), make_ref(toom_mem_kind::tmp, toom3_slots::R1), make_ref(toom_mem_kind::tmp, 0), 1 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, toom3_slots::R0), make_ref(toom_mem_kind::tmp, toom3_slots::R2), make_ref(toom_mem_kind::tmp, 0), 2 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, toom3_slots::R0), make_ref(toom_mem_kind::tmp, toom3_slots::R3), make_ref(toom_mem_kind::tmp, 0), 3 },
    toom_instr{ toom_op::compose_shifted, make_ref(toom_mem_kind::rb, toom3_slots::R0), make_ref(toom_mem_kind::tmp, toom3_slots::R4), make_ref(toom_mem_kind::tmp, 0), 4 },
};

static constexpr auto toom3_spec = make_toom3_spec();

template <>
struct toom_stage_traits<3, 3>
{
    static constexpr auto const& plan = toom3_plan;
    static constexpr auto const& size_exprs = toom3_spec.exprs;
    static constexpr auto const& slot_layout = toom3_spec.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom3_spec.slab_expr;
    static constexpr size_t N = 3;
    static constexpr size_t M = 3;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
