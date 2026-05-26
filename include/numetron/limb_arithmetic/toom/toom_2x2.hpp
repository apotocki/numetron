// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "core.hpp"

namespace numetron::limb_arithmetic {

namespace toom_runtime_detail {

namespace toom2_slots {

// Temporary slots inside scratch slab.
// d_u: |u0 - u1|
// d_v: |v0 - v1|
// TC0: copy of c0 = u0*v0 (used to avoid overlap during in-place middle accumulation)
// TC2: c2 = d_u * d_v
inline constexpr unsigned short d_u = 0;
inline constexpr unsigned short d_v = 1;
inline constexpr unsigned short TC0 = 2;
inline constexpr unsigned short TC2 = 3;

// Views into result buffer rb.
// C0: c0 = u0*v0 at rb[0 .. 2*n2)
inline constexpr unsigned short C0 = 4;
// C1: c1 = u1*v1 at rb[2*n2 .. 2*n2 + (u_hi+v_hi))
inline constexpr unsigned short C1 = 5;
// RM: middle window rb[n2 .. un+vn), where c0/c1/c2 are accumulated
inline constexpr unsigned short RM = 6;
// R: optional full-result window rb[0 .. un+vn) (not used in current plan)
inline constexpr unsigned short R = 7;

} // namespace toom2_slots

template <typename ExprPackT, typename SlotLayoutT>
struct toom_spec
{
    ExprPackT exprs{};
    SlotLayoutT slot_layout{};
    expr_handle slab_expr{};
};

template <typename ExprPackT, typename SlotLayoutT>
consteval auto make_toom_spec(ExprPackT exprs, SlotLayoutT slot_layout, expr_handle slab_expr)
{
    return toom_spec<ExprPackT, SlotLayoutT>{ exprs, slot_layout, slab_expr };
}

consteval auto make_toom2_spec()
{
    // Build named size expressions once and reuse typed handles in slot layout.
    expr_builder<16> b;

    auto n2 = b.v(toom_size_var::chunk);
    auto d_buf_n = b.v(toom_size_var::d_buf_n);

    auto two_n2 = b.mul(n2, 2);
    auto three_n2 = b.mul(n2, 3);
    auto four_n2 = b.mul(n2, 4);
    auto d_plus_n2 = b.add(d_buf_n, n2);
    auto d_plus_2n2 = b.add(d_buf_n, two_n2);
    auto d_plus_3n2 = b.add(d_buf_n, three_n2);
    auto two_d = b.mul(d_buf_n, 2);
    auto slab = b.add(two_d, four_n2);
    auto v_hi_plus_u_hi = b.add(b.v(toom_size_var::v_hi), b.v(toom_size_var::u_hi));
    auto un_plus_vn = b.add(b.v(toom_size_var::un), b.v(toom_size_var::vn));
    auto un_plus_vn_minus_n2 = b.sub(un_plus_vn, n2);

    return make_toom_spec(b.finish(slab), std::array{
        // Karatsuba-style scratch map:
        // [0 .. d_buf_n)                    -> d_u = |u0-u1|
        // [d_buf_n .. d_buf_n+n2)           -> d_v = |v0-v1|
        // [d_buf_n+n2 .. d_buf_n+3*n2)      -> TC0 (copy of c0)
        // [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)  -> TC2 = d_u*d_v
        toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::d_u, 0, d_buf_n },
        toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::d_v, d_buf_n, n2 },
        toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::TC0, d_plus_n2, two_n2 },
        toom_slot_layout{ toom_mem_kind::tmp, toom2_slots::TC2, d_plus_3n2, d_plus_n2 },

        // Result-buffer views used by rb-kind refs in the plan.
        toom_slot_layout{ toom_mem_kind::rb, toom2_slots::C0, 0, two_n2 },
        toom_slot_layout{ toom_mem_kind::rb, toom2_slots::C1, two_n2, v_hi_plus_u_hi },
        toom_slot_layout{ toom_mem_kind::rb, toom2_slots::RM, n2, un_plus_vn_minus_n2 }
        //, toom_slot_layout{ toom_mem_kind::rb, toom2_slots::R, 0, un_plus_vn }
    }, slab);
}

inline constexpr std::array toom2_plan{
    // Pointwise products at x=0 and x=inf.
    // - C0 = u0*v0 is written to rb[0 .. 2*n2)
    // - C1 = u1*v1 is written to rb[2*n2 .. 2*n2 + (u_hi+v_hi))
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, toom2_slots::C0), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::v, 0), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::rb, toom2_slots::C1), make_ref(toom_mem_kind::u, 1), make_ref(toom_mem_kind::v, 1), 0 },

    // Evaluate at x=-1: (u0-u1)*(v0-v1).
    // - d_u uses [0 .. d_buf_n)
    // - d_v uses [d_buf_n .. d_buf_n+n2)
    // - TC2 uses [d_buf_n+3*n2 .. 2*d_buf_n+4*n2)
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::d_u), make_ref(toom_mem_kind::u, 0), make_ref(toom_mem_kind::u, 1), 0 },
    toom_instr{ toom_op::sub, make_ref(toom_mem_kind::tmp, toom2_slots::d_v), make_ref(toom_mem_kind::v, 0), make_ref(toom_mem_kind::v, 1), 0 },
    toom_instr{ toom_op::mul_block, make_ref(toom_mem_kind::tmp, toom2_slots::TC2), make_ref(toom_mem_kind::tmp, toom2_slots::d_u), make_ref(toom_mem_kind::tmp, toom2_slots::d_v), 0 },

    // Middle accumulation in rb window RM:
    // RM += C1; RM += TC0(copy of C0); RM -= TC2.
    toom_instr{ toom_op::copy, make_ref(toom_mem_kind::tmp, toom2_slots::TC0), make_ref(toom_mem_kind::rb, toom2_slots::C0) },
    toom_instr{ toom_op::inplace_add, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::rb, toom2_slots::C1) },
    toom_instr{ toom_op::inplace_add, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::tmp, toom2_slots::TC0) },
    toom_instr{ toom_op::inplace_sub, make_ref(toom_mem_kind::rb, toom2_slots::RM), make_ref(toom_mem_kind::tmp, toom2_slots::TC2) },
};

static constexpr auto toom2_spec = make_toom2_spec();

template <>
struct toom_stage_traits<2, 2>
{
    static constexpr auto const& plan = toom2_plan;
    static constexpr auto const& size_exprs = toom2_spec.exprs;
    static constexpr auto const& slot_layout = toom2_spec.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom2_spec.slab_expr;
    static constexpr size_t N = 2;
    static constexpr size_t M = 2;
};

} // namespace toom_runtime_detail

} // namespace numetron::limb_arithmetic
