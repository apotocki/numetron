// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <span>
#include <tuple>
#include <memory>
#include <cassert>
#include <cstring>

#include "toom_plan.hpp"
#include "toom_core.hpp"
#include "toom_2x2.hpp"
#include "toom_3x3.hpp"

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
    size_t slab_len = 0;
    bool slab_owned = false;
    std::array<toom_slot<LimbT>, 64> slots{};
};

template <toom_size_var V>
inline size_t eval_size_var(toom_size_eval_context const& c)
{
    if constexpr (V == toom_size_var::un) return c.un;
    else if constexpr (V == toom_size_var::vn) return c.vn;
    else if constexpr (V == toom_size_var::chunk) return c.chunk;
    else if constexpr (V == toom_size_var::u_hi) return c.u_hi;
    else if constexpr (V == toom_size_var::v_hi) return c.v_hi;
    else if constexpr (V == toom_size_var::d_buf_n) return c.d_buf_n;
    else return 0;
}

#if 0
template <size_t PlanSize>
consteval size_t required_slot_count(std::array<toom_instr, PlanSize> const& plan)
{
    unsigned short max_idx = 0;
    for (auto const& op : plan) {
        if (ref_kind(op.dst) == toom_mem_kind::tmp)
            max_idx = (std::max)(max_idx, ref_expr_id(op.dst));
        if (ref_kind(op.src0) == toom_mem_kind::tmp)
            max_idx = (std::max)(max_idx, ref_expr_id(op.src0));
        if (ref_kind(op.src1) == toom_mem_kind::tmp)
            max_idx = (std::max)(max_idx, ref_expr_id(op.src1));
    }
    return static_cast<size_t>(max_idx) + 1;
}
#endif

template <size_t N, size_t M, size_t ExprId>
inline size_t eval_size_expr_ct(toom_size_eval_context const& c)
{
    constexpr auto const& exprs = toom_stage_traits<N, M>::size_exprs;
    if constexpr (ExprId >= exprs.size()) {
        return ExprId * c.chunk;
    } else {
        constexpr toom_size_expr e = exprs[ExprId];
        if constexpr (e.op == toom_size_expr_op::constant) {
            return static_cast<size_t>(e.a);
        } else if constexpr (e.op == toom_size_expr_op::variable) {
            return eval_size_var<static_cast<toom_size_var>(e.a)>(c);
        } else if constexpr (e.op == toom_size_expr_op::add) {
            return eval_size_expr_ct<N, M, e.a>(c) + eval_size_expr_ct<N, M, e.b>(c);
        } else if constexpr (e.op == toom_size_expr_op::mul_const) {
            return eval_size_expr_ct<N, M, e.a>(c) * static_cast<size_t>(e.b);
        } else if constexpr (e.op == toom_size_expr_op::max2) {
            return (std::max)(eval_size_expr_ct<N, M, e.a>(c), eval_size_expr_ct<N, M, e.b>(c));
        } else {
            return 0;
        }
    }
}

#ifndef NDEBUG
template <size_t N, size_t M>
inline size_t eval_size_expr_rt(toom_size_eval_context const& c, unsigned short expr_id)
{
    constexpr auto const& exprs = toom_stage_traits<N, M>::size_exprs;
    if (expr_id >= exprs.size()) {
        return static_cast<size_t>(expr_id) * c.chunk;
    }

    const toom_size_expr e = exprs[expr_id];
    switch (e.op) {
    case toom_size_expr_op::constant:
        return static_cast<size_t>(e.a);
    case toom_size_expr_op::variable:
        switch (static_cast<toom_size_var>(e.a)) {
        case toom_size_var::un: return c.un;
        case toom_size_var::vn: return c.vn;
        case toom_size_var::chunk: return c.chunk;
        case toom_size_var::u_hi: return c.u_hi;
        case toom_size_var::v_hi: return c.v_hi;
        case toom_size_var::d_buf_n: return c.d_buf_n;
        }
        return 0;
    case toom_size_expr_op::add:
        return eval_size_expr_rt<N, M>(c, e.a) + eval_size_expr_rt<N, M>(c, e.b);
    case toom_size_expr_op::mul_const:
        return eval_size_expr_rt<N, M>(c, e.a) * static_cast<size_t>(e.b);
    case toom_size_expr_op::max2:
        return (std::max)(eval_size_expr_rt<N, M>(c, e.a), eval_size_expr_rt<N, M>(c, e.b));
    }
    return 0;
}
#endif

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t I>
inline void init_tmp_state_entry(stage_memory_state<LimbT>& mem, toom_size_eval_context const& size_ctx)
{
    constexpr auto e = toom_stage_traits<N, M>::tmp_layout[I];
    const size_t off = eval_size_expr_ct<N, M, e.off_expr>(size_ctx);
    const size_t cap = eval_size_expr_ct<N, M, e.cap_expr>(size_ctx);
#ifndef NDEBUG
    assert((off == eval_size_expr_rt<N, M>(size_ctx, e.off_expr)));
    assert((cap == eval_size_expr_rt<N, M>(size_ctx, e.cap_expr)));
#endif
    assert(e.var < mem.slots.size());
    assert(off + cap <= mem.slab_len);
    auto& s = mem.slots[e.var];
    s.ptr = mem.slab + off;
    s.cap = cap;
    s.len = 0;
    s.sign = 0;
}

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t... Is>
inline void init_tmp_state_impl(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::index_sequence<Is...>)
{
    (init_tmp_state_entry<LimbT, N, M, Is>(mem, size_ctx), ...);
}

template <std::unsigned_integral LimbT, size_t N, size_t M>
inline void init_tmp_state(stage_memory_state<LimbT>& mem, toom_size_eval_context const& size_ctx)
{
    init_tmp_state_impl<LimbT, N, M>(
        mem,
        size_ctx,
        std::make_index_sequence<toom_stage_traits<N, M>::tmp_layout.size()>{});
}

template <std::unsigned_integral LimbT, size_t N, size_t M>
inline toom_slot<LimbT> resolve_ref_read(
    stage_memory_state<LimbT>& mem,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    toom_ref ref)
{
    switch (ref_kind(ref)) {
    case toom_mem_kind::tmp:
        return mem.slots[ref_expr_id(ref)];
    case toom_mem_kind::u:
        return make_positive_view<LimbT>(resolve_input_part<N>(u, chunk, ref_expr_id(ref)));
    case toom_mem_kind::v:
        return make_positive_view<LimbT>(resolve_input_part<M>(v, chunk, ref_expr_id(ref)));
    case toom_mem_kind::rb:
        return toom_slot<LimbT>{ rb, rsz, rsz, rsz ? 1 : 0 };
    }
    return {};
}

template <std::unsigned_integral LimbT>
inline toom_slot<LimbT>& resolve_ref_write(stage_memory_state<LimbT>& mem, toom_ref ref)
{
    assert(ref_kind(ref) == toom_mem_kind::tmp);
    return mem.slots[ref_expr_id(ref)];
}

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t I>
inline void run_toom_op(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk)
{
    constexpr toom_instr op = toom_stage_traits<N, M>::plan[I];

    if constexpr (op.op == toom_op::load_part_u) {
        auto& dst = resolve_ref_write(mem, op.dst);
        const size_t idx = op.imm;
        const size_t start = idx * chunk;
        std::span<const LimbT> part;
        if (start < u.size()) {
            if (idx + 1 < N) {
                const size_t n = (std::min)(chunk, u.size() - start);
                part = { u.data() + start, n };
            } else {
                part = { u.data() + start, u.size() - start };
            }
        }
        slot_set_positive(dst, part);
    } else if constexpr (op.op == toom_op::load_part_v) {
        auto& dst = resolve_ref_write(mem, op.dst);
        const size_t idx = op.imm;
        const size_t start = idx * chunk;
        std::span<const LimbT> part;
        if (start < v.size()) {
            if (idx + 1 < M) {
                const size_t n = (std::min)(chunk, v.size() - start);
                part = { v.data() + start, n };
            } else {
                part = { v.data() + start, v.size() - start };
            }
        }
        slot_set_positive(dst, part);
    } else if constexpr (op.op == toom_op::clear) {
        auto& dst = resolve_ref_write(mem, op.dst);
        slot_clear(dst);
    } else if constexpr (op.op == toom_op::copy) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        slot_copy(dst, s0);
    } else if constexpr (op.op == toom_op::add) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        auto const s1 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src1);
        slot_add_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::sub) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        auto const s1 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src1);
        slot_sub_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::mul_small) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        slot_mul_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::divexact_small) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        slot_divexact_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::mul_block) {
        auto& dst = resolve_ref_write(mem, op.dst);
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        auto const s1 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src1);
        slot_mul_dispatch(dst, s0, s1);
    } else if constexpr (op.op == toom_op::compose_shifted) {
        auto const s0 = resolve_ref_read<LimbT, N, M>(mem, u, v, rb, rsz, chunk, op.src0);
        slot_add_shifted_to_result(rb, rsz, s0, static_cast<size_t>(op.imm) * chunk);
    } else {
        static_assert(op.op == toom_op::compose_shifted, "Unsupported toom op");
    }
}

template <std::unsigned_integral LimbT, size_t N, size_t M, size_t... Is>
inline void run_toom_stage(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    std::index_sequence<Is...>)
{
    (run_toom_op<LimbT, N, M, Is>(mem, size_ctx, u, v, rb, rsz, chunk), ...);
}

template <std::unsigned_integral LimbT, size_t N, size_t M, typename ScratchAllocatorT>
inline void run_toom_stage(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT& scratch_alloc)
{
    init_tmp_state<LimbT, N, M>(mem, size_ctx);
    run_toom_stage<LimbT, N, M>(mem, size_ctx, u, v, rb, rsz, chunk,
        std::make_index_sequence<toom_stage_traits<N, M>::plan.size()>{});
}

template <std::unsigned_integral LimbT, size_t N, size_t M, typename ScratchAllocatorT>
inline void run_toom_stage(
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT scratch_alloc)
{
    const toom_size_eval_context size_ctx{
        u.size(),
        v.size(),
        chunk,
        u.size() - chunk,
        v.size() - chunk,
        (std::max)(chunk, u.size() - chunk),
    };

    stage_memory_state<LimbT> mem;
    const size_t slab_len = eval_size_expr_ct<N, M, toom_stage_traits<N, M>::slab_size_expr_id>(size_ctx);
    mem.slab = std::allocator_traits<ScratchAllocatorT>::allocate(scratch_alloc, slab_len);
    mem.slab_len = slab_len;
    mem.slab_owned = true;
    NUMETRON_SCOPE_EXIT([&] {
        if (mem.slab_owned) {
            std::allocator_traits<ScratchAllocatorT>::deallocate(scratch_alloc, mem.slab, mem.slab_len);
            mem.slab = nullptr;
            mem.slab_len = 0;
            mem.slab_owned = false;
        }
    });
    run_toom_stage<LimbT, N, M>(mem, size_ctx, u, v, rb, rsz, chunk, scratch_alloc);
}

} // namespace toom_runtime_detail


namespace numetron::limb_arithmetic {

template <size_t N, size_t M>
struct toom_engine
{
    using plan = toom_plan<N, M>;

    template <std::unsigned_integral LimbT, typename AllocatorT>
    requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
    static std::tuple<LimbT*, size_t, size_t>
    umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
    {
        const size_t un = u.size();
        const size_t vn = v.size();

        assert(un > 0 && vn > 0 && un >= vn);

        static_assert(detail::has_toom_plan<N, M>(),
            "numetron: toom_engine requested unsupported (N, M). Add row to toom_plan_table and executor path.");

        const size_t alloc_sz = un + vn;
        LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
        try {
            LimbT* re = rb;

            const size_t chunk = (vn + plan::split_rounding_bias) / plan::split_den;
            assert(chunk > 0);

            std::memset(rb, 0, alloc_sz * sizeof(LimbT));
            toom_runtime_detail::run_toom_stage<LimbT, N, M>(u, v, rb, alloc_sz, chunk, alloc);
            re = rb + alloc_sz;

            while (re != rb && *(re - 1) == 0) --re;
            return { rb, static_cast<size_t>(re - rb), alloc_sz };
        }
        catch (...) {
            std::allocator_traits<AllocatorT>::deallocate(alloc, rb, alloc_sz);
            throw;
        }
    }
};

} // namespace numetron::limb_arithmetic
