// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <new>
#include <array>
#include <span>
#include <tuple>
#include <memory>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <limits>
#include <algorithm>

#include "numetron/detail/scope_exit.hpp"

#include "toom_2x2.hpp"
#include "toom_3x3.hpp"
#include "toom_4x4.hpp"
#include "toom_6x6.hpp"

#include "numetron/limb_arithmetic/toom/slot.hpp"

namespace numetron::limb_arithmetic::toom_runtime_detail {

struct toom_size_eval_context
{
    size_t un = 0;
    size_t vn = 0;
    size_t chunk = 0;
    size_t u_hi = 0;
    size_t v_hi = 0;
    size_t d_buf_n = 0;
};

template <std::unsigned_integral LimbT>
struct stage_memory_state
{
    LimbT* slab = nullptr;
    toom_slot<LimbT>* slots = nullptr;
    // Views of the operand parts u(i) / v(i), computed once per stage (see run_toom_stage)
    // rather than on every reference to them.
    toom_slot<LimbT> const* u_parts = nullptr;
    toom_slot<LimbT> const* v_parts = nullptr;
    size_t slab_len = 0;
    size_t slab_alloc_len = 0;
    bool slab_owned = false;
};

template <auto LayoutV, toom_mem_kind KindV>
consteval size_t required_slot_count()
{
    unsigned short max_idx = 0;
    bool found = false;
    for (auto const& e : LayoutV) {
        if (e.kind == KindV) {
            max_idx = (std::max)(max_idx, e.var);
            found = true;
        }
    }
    return found ? static_cast<size_t>(max_idx) + 1 : 0;
}

template <auto LayoutV>
consteval bool has_unique_slot_vars()
{
    for (size_t i = 0; i < LayoutV.size(); ++i) {
        for (size_t j = i + 1; j < LayoutV.size(); ++j) {
            if (LayoutV[i].var == LayoutV[j].var) {
                return false;
            }
        }
    }
    return true;
}

template <toom_size_var V>
NUMETRON_FORCEINLINE size_t eval_size_var(toom_size_eval_context const& c)
{
    if constexpr (V == toom_size_var::un) return c.un;
    else if constexpr (V == toom_size_var::vn) return c.vn;
    else if constexpr (V == toom_size_var::chunk) return c.chunk;
    else if constexpr (V == toom_size_var::u_hi) return c.u_hi;
    else if constexpr (V == toom_size_var::v_hi) return c.v_hi;
    else if constexpr (V == toom_size_var::d_buf_n) return c.d_buf_n;
    else return 0;
}

template <auto ExpressionsV, uint_least64_t ExprId>
NUMETRON_FORCEINLINE size_t eval_size_expr_ct(toom_size_eval_context const& c)
{
    constexpr uint_least64_t expr_type = (ExprId >> 62) & 3;
    if constexpr (expr_type == 0) {
        // Constant value
        return static_cast<size_t>(ExprId & 0x3FFFFFFFFFFFFFFF);
    } else if constexpr (expr_type == 1) {
        // Variable
        return eval_size_var<static_cast<toom_size_var>(ExprId & 0x3FFFFFFFFFFFFFFF)>(c);
    } else {
        static_assert(expr_type == 2, "Unsupported expression handle tag");
        constexpr size_t idx = static_cast<size_t>(ExprId & expr_payload_mask);
        static_assert(idx < ExpressionsV.count, "Expression ID out of bounds");
        constexpr expr_node e = ExpressionsV.nodes[idx];
        constexpr toom_size_expr_op op = e.op;
        if constexpr (op == toom_size_expr_op::constant) {
            return eval_size_expr_ct<ExpressionsV, e.a.raw>(c);
        } else if constexpr (op == toom_size_expr_op::variable) {
            return eval_size_expr_ct<ExpressionsV, e.a.raw>(c);
        } else if constexpr (op == toom_size_expr_op::add) {
            return eval_size_expr_ct<ExpressionsV, e.a.raw>(c) + eval_size_expr_ct<ExpressionsV, e.b.raw>(c);
        } else if constexpr (op == toom_size_expr_op::sub) {
            return eval_size_expr_ct<ExpressionsV, e.a.raw>(c) - eval_size_expr_ct<ExpressionsV, e.b.raw>(c);
        } else if constexpr (op == toom_size_expr_op::mul_const) {
            return eval_size_expr_ct<ExpressionsV, e.a.raw>(c) * eval_size_expr_ct<ExpressionsV, e.b.raw>(c);
        } else if constexpr (op == toom_size_expr_op::max2) {
            return (std::max)(eval_size_expr_ct<ExpressionsV, e.a.raw>(c), eval_size_expr_ct<ExpressionsV, e.b.raw>(c));
        } else {
            static_assert(false, "Unsupported toom size expression op");
        }
    }
}

#ifndef NDEBUG
template <auto ExpressionsV>
inline size_t eval_size_expr_rt(toom_size_eval_context const& c, uint_least64_t expr_id)
{
    uint_least64_t expr_type = (expr_id >> 62) & 3;
    if (expr_type == 0) {
        // Constant value
        return static_cast<size_t>(expr_id & 0x3FFFFFFFFFFFFFFF);
    } else if (expr_type == 1) {
        // Variable
        switch (static_cast<toom_size_var>(expr_id & 0x3FFFFFFFFFFFFFFF)) {
        case toom_size_var::un: return c.un;
        case toom_size_var::vn: return c.vn;
        case toom_size_var::chunk: return c.chunk;
        case toom_size_var::u_hi: return c.u_hi;
        case toom_size_var::v_hi: return c.v_hi;
        case toom_size_var::d_buf_n: return c.d_buf_n;
        }
    }

    if (expr_type != 2) {
        return 0;
    }

    constexpr auto const& exprs = ExpressionsV;
    const size_t idx = static_cast<size_t>(expr_id & expr_payload_mask);
    NUMETRON_ASSERT(idx < exprs.count);
    const expr_node e = exprs.nodes[idx];
    switch (e.op) {
    case toom_size_expr_op::add:
        return eval_size_expr_rt<ExpressionsV>(c, e.a.raw) + eval_size_expr_rt<ExpressionsV>(c, e.b.raw);
    case toom_size_expr_op::sub:
        return eval_size_expr_rt<ExpressionsV>(c, e.a.raw) - eval_size_expr_rt<ExpressionsV>(c, e.b.raw);
    case toom_size_expr_op::mul_const:
        return eval_size_expr_rt<ExpressionsV>(c, e.a.raw) * eval_size_expr_rt<ExpressionsV>(c, e.b.raw);
    case toom_size_expr_op::max2:
        return (std::max)(eval_size_expr_rt<ExpressionsV>(c, e.a.raw), eval_size_expr_rt<ExpressionsV>(c, e.b.raw));
    case toom_size_expr_op::constant:
        return eval_size_expr_rt<ExpressionsV>(c, e.a.raw);
    case toom_size_expr_op::variable:
        return eval_size_expr_rt<ExpressionsV>(c, e.a.raw);
    }
    return 0;
}
#endif

template <std::unsigned_integral LimbT, typename TraitsT, size_t I>
NUMETRON_FORCEINLINE void init_slot_state_entry(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    LimbT* rb,
    size_t rsz)
{
    constexpr auto const& exprs = TraitsT::size_exprs;
    constexpr toom_slot_layout e = TraitsT::slot_layout[I];

    if constexpr (e.kind == toom_mem_kind::tmp) {
        const size_t off = eval_size_expr_ct<exprs, e.off_expr.raw>(size_ctx);
        const size_t cap = eval_size_expr_ct<exprs, e.cap_expr.raw>(size_ctx);
#ifndef NDEBUG
        NUMETRON_ASSERT((off == eval_size_expr_rt<exprs>(size_ctx, e.off_expr.raw)));
        NUMETRON_ASSERT((cap == eval_size_expr_rt<exprs>(size_ctx, e.cap_expr.raw)));
#endif
        NUMETRON_ASSERT(mem.slots != nullptr);
        NUMETRON_ASSERT(off + cap <= mem.slab_len);
        auto& s = mem.slots[e.var];
        s.ptr = mem.slab + off;
        s.cap = cap;
        s.len = 0;
        s.sign = 0;
    } else if constexpr (e.kind == toom_mem_kind::rb) {
        const size_t off = eval_size_expr_ct<exprs, e.off_expr.raw>(size_ctx);
        const size_t cap = eval_size_expr_ct<exprs, e.cap_expr.raw>(size_ctx);
#ifndef NDEBUG
        NUMETRON_ASSERT((off == eval_size_expr_rt<exprs>(size_ctx, e.off_expr.raw)));
        NUMETRON_ASSERT((cap == eval_size_expr_rt<exprs>(size_ctx, e.cap_expr.raw)));
#endif
        NUMETRON_ASSERT(mem.slots != nullptr);
        auto& s = mem.slots[e.var];
        if (off >= rsz) {
            s.ptr = rb + rsz;
            s.cap = 0;
            s.len = 0;
            s.sign = 0;
            return;
        }

        const size_t clamped_cap = (std::min)(cap, rsz - off);
        s.ptr = rb + off;
        s.cap = clamped_cap;
        s.len = clamped_cap;
        s.sign = clamped_cap ? 1 : 0;
    }
}

template <std::unsigned_integral LimbT, typename TraitsT, size_t... Is>
NUMETRON_FORCEINLINE void init_slot_state_impl(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    LimbT* rb,
    size_t rsz,
    std::index_sequence<Is...>)
{
    (init_slot_state_entry<LimbT, TraitsT, Is>(mem, size_ctx, rb, rsz), ...);
}

template <std::unsigned_integral LimbT, typename TraitsT>
NUMETRON_FORCEINLINE void init_slot_state(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    LimbT* rb,
    size_t rsz)
{
    init_slot_state_impl<LimbT, TraitsT>(
        mem,
        size_ctx,
        rb,
        rsz,
        std::make_index_sequence<TraitsT::slot_layout.size()>{});
}

template <std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE toom_slot<LimbT> make_positive_view(std::span<const LimbT> part) noexcept
{
    return toom_slot<LimbT>{ const_cast<LimbT*>(part.data()), part.size(), part.size(), part.empty() ? 0 : 1 };
}

template <size_t Parts, std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE std::span<const LimbT> resolve_input_part(std::span<const LimbT> src, size_t chunk, unsigned short part_id)
{
    const size_t idx = static_cast<size_t>(part_id);
    const size_t start = idx * chunk;
    if (start >= src.size()) {
        return {};
    }

    if (idx + 1 < Parts) {
        const size_t n = (std::min)(chunk, src.size() - start);
        return { src.data() + start, n };
    }

    return { src.data() + start, src.size() - start };
}

template <toom_ref Ref, std::unsigned_integral LimbT, size_t N, size_t M>
NUMETRON_FORCEINLINE toom_slot<LimbT> resolve_ref_read(
    stage_memory_state<LimbT>& mem,
    [[maybe_unused]] std::span<const LimbT> u,
    [[maybe_unused]] std::span<const LimbT> v,
    [[maybe_unused]] size_t chunk)
{
    constexpr toom_mem_kind kind = ref_kind(Ref);
    constexpr unsigned short expr_id = ref_expr_id(Ref);
    if constexpr (kind == toom_mem_kind::u) {
        static_assert(expr_id < N, "u part index out of range");
        return mem.u_parts[expr_id];
    } else if constexpr (kind == toom_mem_kind::v) {
        static_assert(expr_id < M, "v part index out of range");
        return mem.v_parts[expr_id];
    } else {
        static_assert(kind == toom_mem_kind::tmp || kind == toom_mem_kind::rb);
        return mem.slots[expr_id];
    }
}

// A reference to a source slot's state instead of a copy (used by the fixed-width ops, which read
// a source's ptr/cap/sign before writing dst's len/sign, so dst may be the same slot). Copying
// the 32-byte slot right after its fields were stored one by one would be a wide load over
// narrow stores -- a blocked store forward -- on nearly every op.
template <toom_ref Ref, std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE toom_slot<LimbT> const& resolve_ref_cref(stage_memory_state<LimbT> const& mem) noexcept
{
    constexpr toom_mem_kind kind = ref_kind(Ref);
    constexpr unsigned short id = ref_expr_id(Ref);
    if constexpr (kind == toom_mem_kind::u) return mem.u_parts[id];
    else if constexpr (kind == toom_mem_kind::v) return mem.v_parts[id];
    else {
        static_assert(kind == toom_mem_kind::tmp || kind == toom_mem_kind::rb);
        return mem.slots[id];
    }
}

template <toom_ref Ref, std::unsigned_integral LimbT>
NUMETRON_FORCEINLINE toom_slot<LimbT>& resolve_ref_write(
    stage_memory_state<LimbT>& mem)
{
    static_assert(ref_kind(Ref) == toom_mem_kind::tmp || ref_kind(Ref) == toom_mem_kind::rb, "Invalid dst ref for this op");
    return mem.slots[ref_expr_id(Ref)];
}

// Whether two tmp slots of the plan have the same capacity expression. A fixed-width op between
// such slots works on equal, fully written widths and can call its kernel directly -- no
// clipping, zero-extension or tail fill. Compared by expression handle: equal handles always
// mean equal sizes; different handles that happen to be equal at runtime just take the general
// path. rb slots never qualify, their caps are clamped to the result at runtime.
template <typename TraitsT, toom_ref A, toom_ref B>
consteval bool same_tmp_width()
{
    if (ref_kind(A) != toom_mem_kind::tmp || ref_kind(B) != toom_mem_kind::tmp) return false;
    uint64_t ca = 0, cb = 0;
    bool fa = false, fb = false;
    for (auto const& e : TraitsT::slot_layout) {
        if (e.kind != toom_mem_kind::tmp) continue;
        if (e.var == ref_expr_id(A)) { ca = e.cap_expr.raw; fa = true; }
        if (e.var == ref_expr_id(B)) { cb = e.cap_expr.raw; fb = true; }
    }
    return fa && fb && ca == cb;
}

// Kept small per plan instruction: resolve the refs and call the op's shared implementation
// (the slot_fx_* functions are NUMETRON_NOINLINE -- one copy per op type, not per instruction).
template <std::unsigned_integral LimbT, typename TraitsT, size_t I, typename ScratchAllocatorT>
inline void run_toom_op(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    ScratchAllocatorT scratch_alloc)
{
    constexpr toom_instr op = TraitsT::plan[I];
    if constexpr (op.op == toom_op::clear) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        slot_clear(dst);
    } else if constexpr (op.op == toom_op::copy) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_copy(dst, s0);
    } else if constexpr (op.op == toom_op::add) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_add_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::inplace_add) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_inplace_add(dst, s0);
    } else if constexpr (op.op == toom_op::sub) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_add_signed(dst, s0, toom_slot<LimbT>{ s1.ptr, s1.len, s1.cap, -s1.sign });
    } else if constexpr (op.op == toom_op::inplace_sub) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_inplace_add(dst, toom_slot<LimbT>{ s0.ptr, s0.len, s0.cap, -s0.sign });
    } else if constexpr (op.op == toom_op::mul_small) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_mul_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::divexact_small) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_divexact_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::mul_block) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        slot_mul_dispatch(dst, s0, s1, scratch_alloc);
    } else if constexpr (op.op == toom_op::compose_shifted) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        slot_add_shifted_to_result(rb, rsz, s0, static_cast<size_t>(op.imm) * size_ctx.chunk);
    } else if constexpr (op.op == toom_op::print) {
        auto const s0 = resolve_ref_read<op.dst, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, size_ctx.chunk);
        print_limbs(s0.ptr, s0.cap, "print"sv);
    } else if constexpr (op.op == toom_op::uadd || op.op == toom_op::usub || op.op == toom_op::usub_signed
        || op.op == toom_op::abs_sub || op.op == toom_op::umul_fixed) {
        // Binary fixed-width ops. dst may be one of the sources (see resolve_ref_cref) --
        // including usub_signed reading the sign of the slot it overwrites, before writing it.
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto const& s1 = resolve_ref_cref<op.src1, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        constexpr bool exact = same_tmp_width<TraitsT, op.dst, op.src0>() && same_tmp_width<TraitsT, op.dst, op.src1>();
        if constexpr (op.op == toom_op::uadd) {
            if constexpr (exact) slot_fx_add_exact(dst, s0, s1); else slot_fx_add(dst, s0, s1);
        } else if constexpr (op.op == toom_op::usub) {
            if constexpr (exact) slot_fx_sub_exact(dst, s0, s1); else slot_fx_sub(dst, s0, s1);
        } else if constexpr (op.op == toom_op::usub_signed) {
            if constexpr (exact) slot_fx_sub_signed_exact(dst, s0, s1); else slot_fx_sub_signed(dst, s0, s1);
        } else if constexpr (op.op == toom_op::abs_sub) {
            slot_fx_abs_sub(dst, s0, s1);
        } else {
            slot_fx_mul(dst, s0, s1, scratch_alloc);
        }
    } else if constexpr (op.op == toom_op::addsub_abs) {
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto const& s1 = resolve_ref_cref<op.src1, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto& dst2 = resolve_ref_write<op.dst2, LimbT>(mem);
        slot_fx_addsub_abs(dst, dst2, s0, s1);
    } else if constexpr (op.op == toom_op::uaddlsh || op.op == toom_op::usublsh) {
        static_assert(op.imm > 0 && op.imm < std::numeric_limits<LimbT>::digits, "shift out of range");
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto const& s1 = resolve_ref_cref<op.src1, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        if constexpr (op.op == toom_op::uaddlsh) slot_fx_addlsh(dst, s0, s1, op.imm);
        else slot_fx_sublsh(dst, s0, s1, op.imm);
    } else if constexpr (op.op == toom_op::shl || op.op == toom_op::shr || op.op == toom_op::divexact) {
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        if constexpr (op.op == toom_op::divexact) {
            slot_fx_divexact_by<LimbT, op.imm>(dst, s0);
        } else {
            static_assert(op.imm > 0 && op.imm < std::numeric_limits<LimbT>::digits, "shift out of range");
            if constexpr (op.op == toom_op::shl) slot_fx_shl(dst, s0, op.imm);
            else slot_fx_shr(dst, s0, op.imm);
        }
    } else if constexpr (op.op == toom_op::lincomb) {
        // Unused source positions just repeat src0; the kernel doesn't read them.
        constexpr unsigned n = op.lc.count;
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto const& s1 = resolve_ref_cref<(n > 1 ? op.src1 : op.src0), LimbT>(mem);
        auto const& s2 = resolve_ref_cref<(n > 2 ? op.src2 : op.src0), LimbT>(mem);
        auto const& s3 = resolve_ref_cref<(n > 3 ? op.src3 : op.src0), LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        slot_fx_lincomb<LimbT, op.lc>(dst, s0, s1, s2, s3);
    } else if constexpr (op.op == toom_op::eval_pm1) {
        auto const& e0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto const& e1 = resolve_ref_cref<op.src1, LimbT>(mem);
        auto const& o = resolve_ref_cref<op.src2, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto& dst2 = resolve_ref_write<op.dst2, LimbT>(mem);
        slot_fx_eval_pm1(dst, dst2, e0, e1, o);
    } else if constexpr (op.op == toom_op::shl1 || op.op == toom_op::shr1 || op.op == toom_op::divexact3
        || op.op == toom_op::copy_low || op.op == toom_op::uadd_into) {
        // Unary fixed-width ops.
        auto const& s0 = resolve_ref_cref<op.src0, LimbT>(mem);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        constexpr bool exact = same_tmp_width<TraitsT, op.dst, op.src0>();
        if constexpr (op.op == toom_op::shl1) {
            if constexpr (exact) slot_fx_shl1_exact(dst, s0); else slot_fx_shl1(dst, s0);
        } else if constexpr (op.op == toom_op::shr1) {
            if constexpr (exact) slot_fx_shr1_exact(dst, s0); else slot_fx_shr1(dst, s0);
        } else if constexpr (op.op == toom_op::divexact3) {
            if constexpr (exact) slot_fx_divexact3_exact(dst, s0); else slot_fx_divexact3(dst, s0);
        } else if constexpr (op.op == toom_op::copy_low) {
            slot_fx_copy_low(dst, s0);
        } else {
            if constexpr (exact) slot_fx_add_into_exact(dst, s0); else slot_fx_add_into(dst, s0);
        }
    } else {
        static_assert(op.op == toom_op::compose_shifted, "Unsupported toom op");
    }
}

// Optional plan traits (a plan that doesn't declare them gets the original behavior):
//   split_by_u  -- chunk = ceil(un/N) instead of ceil(vn/M); both operands are then cut into
//                  equal chunks with the remainder in the top part, which keeps every part
//                  <= chunk limbs (the dispatch must make sure v still reaches its top part).
//   zero_result -- whether rb has to be zeroed before the plan runs; a plan that writes every
//                  limb of rb itself (and only then accumulates into it) sets this to false.
template <typename TraitsT>
consteval bool traits_split_by_u()
{
    if constexpr (requires { TraitsT::split_by_u; }) return TraitsT::split_by_u;
    else return false;
}

template <typename TraitsT>
consteval bool traits_zero_result()
{
    if constexpr (requires { TraitsT::zero_result; }) return TraitsT::zero_result;
    else return true;
}

template <typename TraitsT>
inline size_t traits_chunk(size_t un, size_t vn) noexcept
{
    if constexpr (traits_split_by_u<TraitsT>()) return (un + (TraitsT::N - 1)) / TraitsT::N;
    else return (vn + (TraitsT::M - 1)) / TraitsT::M;
}

template <std::unsigned_integral LimbT, typename TraitsT, typename ScratchAllocatorT, size_t... Is>
inline void run_toom_stage(
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT scratch_alloc,
    std::index_sequence<Is...>)
{
    constexpr size_t N = TraitsT::N;
    constexpr size_t M = TraitsT::M;
    const size_t u_hi = u.size() - (N - 1) * chunk;
    const toom_size_eval_context size_ctx{
        u.size(),
        v.size(),
        chunk,
        u_hi,
        v.size() - (M - 1) * chunk,
        (std::max)(chunk, u_hi),
    };

    using traits_t = TraitsT;
    static_assert(has_unique_slot_vars<traits_t::slot_layout>(),
        "toom slot ids must be unique across slot_layout entries");
    // Slot ids are unique across kinds (slot_builder numbers tmp and rb slots from one counter),
    // so the table needs max(id) + 1 entries -- not the per-kind maxima added together.
    constexpr size_t slot_count = (std::max)(
        required_slot_count<traits_t::slot_layout, toom_mem_kind::tmp>(),
        required_slot_count<traits_t::slot_layout, toom_mem_kind::rb>());
    stage_memory_state<LimbT> mem;

    std::array<toom_slot<LimbT>, N> u_parts;
    std::array<toom_slot<LimbT>, M> v_parts;
    for (size_t i = 0; i < N; ++i) {
        u_parts[i] = make_positive_view<LimbT>(resolve_input_part<N>(u, chunk, static_cast<unsigned short>(i)));
    }
    for (size_t i = 0; i < M; ++i) {
        v_parts[i] = make_positive_view<LimbT>(resolve_input_part<M>(v, chunk, static_cast<unsigned short>(i)));
    }
    mem.u_parts = u_parts.data();
    mem.v_parts = v_parts.data();

    const size_t slab_len = eval_size_expr_ct<traits_t::size_exprs, traits_t::slab_size_expr_id.raw>(size_ctx);
    mem.slab = std::allocator_traits<ScratchAllocatorT>::allocate(scratch_alloc, slab_len);
    mem.slab_len = slab_len;
    mem.slab_alloc_len = slab_len;
    mem.slab_owned = true;

    // The slot table lives in this frame, uninitialized: init_slot_state() below writes every
    // field of every slot (the ids are dense -- each one has exactly one layout entry), so
    // constructing/zeroing it first would be wasted work on every recursion node. A std::byte
    // array implicitly creates the (implicit-lifetime, aggregate) toom_slot objects in it.
    static_assert(slot_count == traits_t::slot_layout.size(), "toom slot ids must be dense (0 .. count-1)");
    static_assert(std::is_trivially_destructible_v<toom_slot<LimbT>>);
    alignas(toom_slot<LimbT>) std::byte slot_storage[(slot_count ? slot_count : 1) * sizeof(toom_slot<LimbT>)];
    mem.slots = std::launder(reinterpret_cast<toom_slot<LimbT>*>(slot_storage));

    NUMETRON_SCOPE_EXIT([&] {
        if (mem.slab_owned) {
            std::allocator_traits<ScratchAllocatorT>::deallocate(scratch_alloc, mem.slab, mem.slab_alloc_len);
            mem.slab = nullptr;
            mem.slab_len = 0;
            mem.slab_alloc_len = 0;
            mem.slots = nullptr;
            mem.slab_owned = false;
        }
    });

    init_slot_state<LimbT, traits_t>(mem, size_ctx, rb, rsz);
    (run_toom_op<LimbT, traits_t, Is>(mem, size_ctx, u, v, rb, rsz, scratch_alloc), ...);
}

} // namespace numetron::limb_arithmetic::toom_runtime_detail

namespace numetron::limb_arithmetic {

// Runs one Toom stage described by a plan (TraitsT: see toom_stage_traits and the optional
// traits in toom_runtime_detail above).
template <typename TraitsT>
struct toom_engine_t
{
    static_assert(TraitsT::N > 1, "toom_engine requires N > 1");
    static_assert(TraitsT::M > 1, "toom_engine requires M > 1");

    // Allocates the result buffer via alloc; all scratch of the recursion comes from
    // scratch_alloc, which must serve allocations in LIFO order (see umul() in umul.hpp for the
    // one place it is chosen).
    template <std::unsigned_integral LimbT, typename AllocatorT, typename ScratchAllocatorT>
    requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
    static std::tuple<LimbT*, size_t, size_t>
    umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc, ScratchAllocatorT scratch_alloc)
    {
        const size_t un = u.size();
        const size_t vn = v.size();

        NUMETRON_ASSERT(un > 0 && vn > 0 && un >= vn);

        const size_t alloc_sz = un + vn;
        LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
        try {
            LimbT* re = umul(u.data(), un, v.data(), vn, rb, scratch_alloc);
            while (re != rb && *(re - 1) == 0) --re;
            return { rb, static_cast<size_t>(re - rb), alloc_sz };
        }
        catch (...) {
            std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
            throw;
        }
    }

    template <std::unsigned_integral LimbT, typename AllocatorT>
    static LimbT* umul(
        const LimbT* u, size_t un,
        const LimbT* v, size_t vn,
        LimbT* rb,
        AllocatorT alloc)
    {
        using namespace toom_runtime_detail;

        NUMETRON_ASSERT(un > 0 && vn > 0 && un >= vn);

        const size_t r_sz = un + vn;

        const size_t chunk = traits_chunk<TraitsT>(un, vn);
        NUMETRON_ASSERT(chunk > 0);

        if constexpr (traits_zero_result<TraitsT>()) {
            std::memset(rb, 0, r_sz * sizeof(LimbT));
        }
        run_toom_stage<LimbT, TraitsT>(std::span{u, un}, std::span{v, vn}, rb, r_sz, chunk, alloc,
            std::make_index_sequence<TraitsT::plan.size()>{});
        return rb + r_sz;
    }
};

template <size_t N, size_t M>
struct toom_engine : toom_engine_t<toom_runtime_detail::toom_stage_traits<N, M>> {};

// Balanced Toom-3 plan (toom_3x3.hpp), for operands detail::toom3_split_fits() accepts.
using toom3_balanced_engine = toom_engine_t<toom_runtime_detail::toom3_balanced_traits>;

// Balanced Toom-4 plan (toom_4x4.hpp), for operands detail::toom4_split_fits() accepts.
using toom4_balanced_engine = toom_engine_t<toom_runtime_detail::toom4_balanced_traits>;

// Balanced Toom-6.5 plan (toom_6x6.hpp), for operands detail::toom6h_split_fits() accepts.
using toom6h_balanced_engine = toom_engine_t<toom_runtime_detail::toom6h_balanced_traits>;

} // namespace numetron::limb_arithmetic
