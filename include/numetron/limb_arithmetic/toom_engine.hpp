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

template <std::unsigned_integral LimbT, size_t SlotSzV>
struct stage_memory_state
{
    LimbT* slab = nullptr;
    size_t slab_len = 0;
    bool slab_owned = false;
    std::array<toom_slot<LimbT>, SlotSzV> slots{};
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

template <auto ExpressionsV, size_t ExprId>
//requires (ExprId < ExpressionsV.size())
inline size_t eval_size_expr_ct(toom_size_eval_context const& c)
{
    static_assert(ExprId < ExpressionsV.size(), "Expression ID out of bounds");
    constexpr toom_size_expr e = ExpressionsV[ExprId];
    constexpr toom_size_expr_op op = e.op;
    if constexpr (op == toom_size_expr_op::constant) {
        return static_cast<size_t>(e.a);
    } else if constexpr (op == toom_size_expr_op::variable) {
        return eval_size_var<static_cast<toom_size_var>(e.a)>(c);
    } else if constexpr (op == toom_size_expr_op::add) {
        return eval_size_expr_ct<ExpressionsV, e.a>(c) + eval_size_expr_ct<ExpressionsV, e.b>(c);
    } else if constexpr (op == toom_size_expr_op::mul_const) {
        return eval_size_expr_ct<ExpressionsV, e.a>(c) * static_cast<size_t>(e.b);
    } else if constexpr (op == toom_size_expr_op::max2) {
        return (std::max)(eval_size_expr_ct<ExpressionsV, e.a>(c), eval_size_expr_ct<ExpressionsV, e.b>(c));
    } else {
        static_assert(false, "Unsupported toom size expression op");
    }
}

//template <size_t ESzV, std::array<toom_size_expr, ESzV> exprs, size_t ExprId>
//requires (ExprId < ESzV)
//inline size_t eval_size_expr_ct(toom_size_eval_context const& c)
//{
//    //if constexpr (ExprId >= exprs.size()) {
//    //    return ExprId * c.chunk;
//    //} else {
//        constexpr toom_size_expr e = exprs[ExprId];
//        if constexpr (e.op == toom_size_expr_op::constant) {
//            return static_cast<size_t>(e.a);
//        } else if constexpr (e.op == toom_size_expr_op::variable) {
//            return eval_size_var<static_cast<toom_size_var>(e.a)>(c);
//        } else if constexpr (e.op == toom_size_expr_op::add) {
//            return eval_size_expr_ct<ESzV, exprs, e.a>(c) + eval_size_expr_ct<ESzV, exprs, e.b>(c);
//        } else if constexpr (e.op == toom_size_expr_op::mul_const) {
//            return eval_size_expr_ct<ESzV, exprs, e.a>(c) * static_cast<size_t>(e.b);
//        } else if constexpr (e.op == toom_size_expr_op::max2) {
//            return (std::max)(eval_size_expr_ct<ESzV, exprs, e.a>(c), eval_size_expr_ct<ESzV, exprs, e.b>(c));
//        } else {
//            return 0;
//        }
//    //}
//}

#ifndef NDEBUG
template <auto ExpressionsV>
inline size_t eval_size_expr_rt(toom_size_eval_context const& c, unsigned short expr_id)
{
    constexpr auto const& exprs = ExpressionsV;
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
        return eval_size_expr_rt<ExpressionsV>(c, e.a) + eval_size_expr_rt<ExpressionsV>(c, e.b);
    case toom_size_expr_op::mul_const:
        return eval_size_expr_rt<ExpressionsV>(c, e.a) * static_cast<size_t>(e.b);
    case toom_size_expr_op::max2:
        return (std::max)(eval_size_expr_rt<ExpressionsV>(c, e.a), eval_size_expr_rt<ExpressionsV>(c, e.b));
    }
    return 0;
}
#endif

template <std::unsigned_integral LimbT, typename TraitsT, size_t I, size_t SlotSzV>
inline void init_tmp_state_entry(stage_memory_state<LimbT, SlotSzV>& mem, toom_size_eval_context const& size_ctx)
{
    constexpr auto const& exprs = TraitsT::size_exprs;
    constexpr auto e = TraitsT::tmp_layout[I];
    const size_t off = eval_size_expr_ct<exprs, e.off_expr>(size_ctx);
    const size_t cap = eval_size_expr_ct<exprs, e.cap_expr>(size_ctx);
#ifndef NDEBUG
    NUMETRON_ASSERT((off == eval_size_expr_rt<exprs>(size_ctx, e.off_expr)));
    NUMETRON_ASSERT((cap == eval_size_expr_rt<exprs>(size_ctx, e.cap_expr)));
#endif
    NUMETRON_ASSERT(e.var < mem.slots.size());
    NUMETRON_ASSERT(off + cap <= mem.slab_len);
    auto& s = mem.slots[e.var];
    s.ptr = mem.slab + off;
    s.cap = cap;
    s.len = 0;
    s.sign = 0;
}

template <std::unsigned_integral LimbT, typename TraitsT, size_t SlotSzV, size_t... Is>
inline void init_tmp_state_impl(
    stage_memory_state<LimbT, SlotSzV>& mem,
    toom_size_eval_context const& size_ctx,
    std::index_sequence<Is...>)
{
    (init_tmp_state_entry<LimbT, TraitsT, Is>(mem, size_ctx), ...);
}

template <std::unsigned_integral LimbT, typename TraitsT, size_t SlotSzV>
inline void init_tmp_state(stage_memory_state<LimbT, SlotSzV>& mem, toom_size_eval_context const& size_ctx)
{
    init_tmp_state_impl<LimbT, TraitsT>(
        mem,
        size_ctx,
        std::make_index_sequence<TraitsT::tmp_layout.size()>{});
}

template <std::unsigned_integral LimbT>
inline toom_slot<LimbT> make_positive_view(std::span<const LimbT> part) noexcept
{
    return toom_slot<LimbT>{ const_cast<LimbT*>(part.data()), part.size(), part.size(), part.empty() ? 0 : 1 };
}

template <toom_ref Ref, std::unsigned_integral LimbT, size_t N, size_t M, size_t SlotSzV>
inline toom_slot<LimbT> resolve_ref_read(
    stage_memory_state<LimbT, SlotSzV>& mem,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk)
{
    if constexpr (ref_kind(Ref) == toom_mem_kind::tmp) {
        return mem.slots[ref_expr_id(Ref)];
    } else if constexpr (ref_kind(Ref) == toom_mem_kind::u) {
        return make_positive_view<LimbT>(resolve_input_part<N>(u, chunk, ref_expr_id(Ref)));
    } else if constexpr (ref_kind(Ref) == toom_mem_kind::v) {
        return make_positive_view<LimbT>(resolve_input_part<M>(v, chunk, ref_expr_id(Ref)));
    } else if constexpr (ref_kind(Ref) == toom_mem_kind::rb) {
        const size_t off = static_cast<size_t>(ref_expr_id(Ref)) * chunk;
        if (off >= rsz) return {};
        const size_t cap = (std::min)(rsz - off, 2 * chunk);
        toom_slot<LimbT> view{ rb + off, cap, cap, cap ? 1 : 0 };
        slot_trim(view);
        if (!view.len) view.sign = 0;
        return view;
    }
    return {};
}

//template <std::unsigned_integral LimbT>
//inline toom_slot<LimbT>& resolve_ref_write(stage_memory_state<LimbT>& mem, toom_ref ref)
//{
//    NUMETRON_ASSERT(ref_kind(ref) == toom_mem_kind::tmp);
//    return mem.slots[ref_expr_id(ref)];
//}

template <toom_ref Ref, std::unsigned_integral LimbT, size_t SlotSzV>
inline toom_slot<LimbT> resolve_ref_write(
    stage_memory_state<LimbT, SlotSzV>& mem,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk)
{
    if constexpr (ref_kind(Ref) == toom_mem_kind::rb) {
        const size_t off = static_cast<size_t>(ref_expr_id(Ref)) * chunk;
        //NUMETRON_ASSERT(off <= rsz);
        NUMETRON_ASSERT(off + 2 * chunk < rsz);
        return toom_slot<LimbT> { rb + off, 0, 2 * chunk, 0 };
    } else {
        static_assert(ref_kind(Ref) == toom_mem_kind::tmp, "Invalid dst ref for this op");
        return mem.slots[ref_expr_id(Ref)];
    }
}

template <std::unsigned_integral LimbT, typename TraitsT, size_t I, size_t SlotSzV, typename ScratchAllocatorT>
inline void run_toom_op(
    stage_memory_state<LimbT, SlotSzV>& mem,
    toom_size_eval_context const& size_ctx,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    ScratchAllocatorT scratch_alloc)
{
    constexpr toom_instr op = TraitsT::plan[I];
    constexpr size_t N = TraitsT::N;
    constexpr size_t M = TraitsT::M;
    if constexpr (op.op == toom_op::clear) {
        auto& dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_clear(dst);
    } else if constexpr (op.op == toom_op::copy) {
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_copy(dst, s0);
    } else if constexpr (op.op == toom_op::add) {
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_add_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::sub) {
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_sub_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::mul_small) {
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_mul_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::divexact_small) {
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_divexact_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::mul_block) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto dst = resolve_ref_write<op.dst>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_mul_dispatch(dst, s0, s1, scratch_alloc);
    } else if constexpr (op.op == toom_op::compose_shifted) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, N, M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_add_shifted_to_result(rb, rsz, s0, static_cast<size_t>(op.imm) * size_ctx.chunk);
    } else {
        static_assert(op.op == toom_op::compose_shifted, "Unsupported toom op");
    }
}

template <std::unsigned_integral LimbT, size_t N, size_t M, typename ScratchAllocatorT, size_t... Is>
inline void run_toom_stage(
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk,
    ScratchAllocatorT scratch_alloc,
    std::index_sequence<Is...>)
{
    const toom_size_eval_context size_ctx{
        u.size(),
        v.size(),
        chunk,
        u.size() - chunk,
        v.size() - chunk,
        (std::max)(chunk, u.size() - chunk),
    };

    using traits_t = toom_stage_traits<N, M>;
    stage_memory_state<LimbT, traits_t::tmp_layout.size()> mem;
    const size_t slab_len = eval_size_expr_ct<traits_t::size_exprs, traits_t::slab_size_expr_id>(size_ctx);
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
    init_tmp_state<LimbT, traits_t>(mem, size_ctx);
    (run_toom_op<LimbT, traits_t, Is>(mem, size_ctx, u, v, rb, rsz, scratch_alloc), ...);
}

} // namespace numetron::limb_arithmetic::toom_runtime_detail

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
        using namespace toom_runtime_detail;

        const size_t un = u.size();
        const size_t vn = v.size();

        NUMETRON_ASSERT(un > 0 && vn > 0 && un >= vn);

        static_assert(detail::has_toom_plan<N, M>(),
            "numetron: toom_engine requested unsupported (N, M). Add row to toom_plan_table and executor path.");

        const size_t alloc_sz = un + vn;
        LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
        try {
            LimbT* re = rb;

            const size_t chunk = (vn + plan::split_rounding_bias) / plan::split_den;
            NUMETRON_ASSERT(chunk > 0);

            std::memset(rb, 0, alloc_sz * sizeof(LimbT));
            run_toom_stage<LimbT, N, M>(u, v, rb, alloc_sz, chunk, alloc,
                std::make_index_sequence<toom_stage_traits<N, M>::plan.size()>{});
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
