// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <new>
#include <span>
#include <tuple>
#include <memory>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <algorithm>

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
    toom_slot<LimbT>* slots = nullptr;
    size_t slab_len:16 = 0;
    size_t slab_alloc_len:16 = 0;
    size_t slab_owned:1 = 0;
};

//template <std::unsigned_integral LimbT>
//inline toom_slot<LimbT>* get_slots(stage_memory_state<LimbT>& mem)
//{
//    constexpr size_t slot_count = SlotCountV;
//    if constexpr (slot_count == 0) {
//        return nullptr;
//    } else {
//        NUMETRON_ASSERT(mem.slab != nullptr);
//        NUMETRON_ASSERT(mem.slab_alloc_len >= mem.slab_len);
//
//        std::byte* slots_storage = reinterpret_cast<std::byte*>(mem.slab + mem.slab_len);
//        size_t slots_storage_space = (mem.slab_alloc_len - mem.slab_len) * sizeof(LimbT);
//        void* slots_ptr = slots_storage;
//        slots_ptr = std::align(alignof(toom_slot<LimbT>), slot_count * sizeof(toom_slot<LimbT>), slots_ptr, slots_storage_space);
//        NUMETRON_ASSERT(slots_ptr != nullptr);
//        return static_cast<toom_slot<LimbT>*>(slots_ptr);
//    }
//}

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

template <auto ExpressionsV, uint_least64_t ExprId>
inline size_t eval_size_expr_ct(toom_size_eval_context const& c)
{
    constexpr uint_least64_t expr_type = (ExprId >> 62) & 3;
    if constexpr (expr_type == 1) {
        // Constant value
        return static_cast<size_t>(ExprId & 0x3FFFFFFFFFFFFFFF);
    } else if constexpr (expr_type == 3) {
        // Variable
        return eval_size_var<static_cast<toom_size_var>(ExprId & 0x3FFFFFFFFFFFFFFF)>(c);
    } else {
        static_assert(ExprId < ExpressionsV.size(), "Expression ID out of bounds");
        constexpr toom_size_expr e = ExpressionsV[ExprId];
        constexpr toom_size_expr_op op = e.op;
        if constexpr (op == toom_size_expr_op::constant) {
            return static_cast<size_t>(e.a);
        } else if constexpr (op == toom_size_expr_op::variable) {
            return eval_size_var<static_cast<toom_size_var>(e.a)>(c);
        } else if constexpr (op == toom_size_expr_op::add) {
            return eval_size_expr_ct<ExpressionsV, e.a>(c) + eval_size_expr_ct<ExpressionsV, e.b>(c);
        } else if constexpr (op == toom_size_expr_op::sub) {
            return eval_size_expr_ct<ExpressionsV, e.a>(c) - eval_size_expr_ct<ExpressionsV, e.b>(c);
        } else if constexpr (op == toom_size_expr_op::mul_const) {
            return eval_size_expr_ct<ExpressionsV, e.a>(c) * static_cast<size_t>(e.b);
        } else if constexpr (op == toom_size_expr_op::max2) {
            return (std::max)(eval_size_expr_ct<ExpressionsV, e.a>(c), eval_size_expr_ct<ExpressionsV, e.b>(c));
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
    if (expr_type == 1) {
        // Constant value
        return static_cast<size_t>(expr_id & 0x3FFFFFFFFFFFFFFF);
    } else if (expr_type == 3) {
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
    case toom_size_expr_op::sub:
        return eval_size_expr_rt<ExpressionsV>(c, e.a) - eval_size_expr_rt<ExpressionsV>(c, e.b);
    case toom_size_expr_op::mul_const:
        return eval_size_expr_rt<ExpressionsV>(c, e.a) * static_cast<size_t>(e.b);
    case toom_size_expr_op::max2:
        return (std::max)(eval_size_expr_rt<ExpressionsV>(c, e.a), eval_size_expr_rt<ExpressionsV>(c, e.b));
    }
    return 0;
}
#endif

template <std::unsigned_integral LimbT, typename TraitsT, size_t I>
inline void init_slot_state_entry(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    LimbT* rb,
    size_t rsz)
{
    constexpr auto const& exprs = TraitsT::size_exprs;
    constexpr toom_slot_layout e = TraitsT::slot_layout[I];

    if constexpr (e.kind == toom_mem_kind::tmp) {
        const size_t off = eval_size_expr_ct<exprs, e.off_expr>(size_ctx);
        const size_t cap = eval_size_expr_ct<exprs, e.cap_expr>(size_ctx);
#ifndef NDEBUG
        NUMETRON_ASSERT((off == eval_size_expr_rt<exprs>(size_ctx, e.off_expr)));
        NUMETRON_ASSERT((cap == eval_size_expr_rt<exprs>(size_ctx, e.cap_expr)));
#endif
        NUMETRON_ASSERT(mem.slots != nullptr);
        NUMETRON_ASSERT(off + cap <= mem.slab_len);
        auto& s = mem.slots[e.var];
        s.ptr = mem.slab + off;
        s.cap = cap;
        s.len = 0;
        s.sign = 0;
    } else if constexpr (e.kind == toom_mem_kind::rb) {
        const size_t off = eval_size_expr_ct<exprs, e.off_expr>(size_ctx);
        const size_t cap = eval_size_expr_ct<exprs, e.cap_expr>(size_ctx);
#ifndef NDEBUG
        NUMETRON_ASSERT((off == eval_size_expr_rt<exprs>(size_ctx, e.off_expr)));
        NUMETRON_ASSERT((cap == eval_size_expr_rt<exprs>(size_ctx, e.cap_expr)));
#endif
        NUMETRON_ASSERT(mem.slots != nullptr);
        NUMETRON_ASSERT(off + cap <= rsz);
        auto& s = mem.slots[e.var];
        s.ptr = rb + off;
        s.cap = cap;
        s.len = cap;
        s.sign = 1;

        //if (off >= rsz) {
        //    s.ptr = rb + rsz;
        //    s.cap = 0;
        //    s.len = 0;
        //    s.sign = 0;
        //    return;
        //}

        //s.ptr = rb + off;
        //s.cap = (std::min)(rsz - off, 2 * size_ctx.chunk);
        //s.len = 0;
        //s.sign = 0;
    }
}

template <std::unsigned_integral LimbT, typename TraitsT, size_t... Is>
inline void init_slot_state_impl(
    stage_memory_state<LimbT>& mem,
    toom_size_eval_context const& size_ctx,
    LimbT* rb,
    size_t rsz,
    std::index_sequence<Is...>)
{
    (init_slot_state_entry<LimbT, TraitsT, Is>(mem, size_ctx, rb, rsz), ...);
}

template <std::unsigned_integral LimbT, typename TraitsT>
inline void init_slot_state(
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
void slot_add_signed(toom_slot<LimbT>& dst, toom_slot<LimbT> a, toom_slot<LimbT> b)
{
    slot_trim(a);
    if (!a.sign) { slot_copy(dst, b); return; }
    slot_trim(b);
    if (!b.sign) { slot_copy(dst, a); return; }

    if (a.sign == b.sign) {
        NUMETRON_ASSERT(dst.cap >= a.len);
        NUMETRON_ASSERT(dst.cap >= b.len);
        dst.sign = a.sign;
        size_t isz = (std::min)(dst.len, a.len);
        LimbT const* aptr = a.ptr;
        LimbT* dstptr = dst.ptr;
        LimbT carry = uadd_partial_unchecked(aptr, b.ptr, b.ptr + isz, dstptr);
        if (isz < a.len) {
            std::memcpy(dstptr, aptr, (a.len - isz) * sizeof(LimbT));
            dst.len = a.len;
        } else if (isz < b.len) {
            std::memcpy(dstptr, b.ptr + isz, (b.len - isz) * sizeof(LimbT));
            dst.len = b.len;
        } else {
            dst.len = isz;
        }
        if (carry) {
            carry = uadd_limb(dst.ptr + isz, dst.ptr + dst.len, carry);
            if (carry) {
                NUMETRON_ASSERT(dst.cap >= dst.len + 1);
                if (dst.cap >= dst.len + 1) {
                    dst.ptr[dst.len] = carry;
                    ++dst.len;
                }
            }
        }
        return;
    }

    NUMETRON_ASSERT(a.len && b.len);
    LimbT const* plast_a = a.ptr + a.len - 1;
    LimbT const* plast_b = b.ptr + b.len - 1;

    bool do_swap = false;
    if (a.len == b.len) {
        while (*plast_a == *plast_b) {
            if (plast_a == a.ptr) {
                dst.len = 0;
                dst.sign = 0;
                return;
            }
            --plast_a; --plast_b; --a.len;
        }
        b.len = a.len;
        do_swap = *plast_a < *plast_b;
    }
    LimbT* dstptr = dst.ptr;
    if ((a.len < b.len) ^ do_swap) {
        std::swap(a, b);
    }
    LimbT const* aptr = a.ptr;
    LimbT const* aptr_e = a.ptr + a.len;
    NUMETRON_ASSERT(dst.cap >= a.len);
    LimbT borrow = usub_partial_unchecked<LimbT>(aptr, b.ptr, b.ptr + b.len, dstptr);
    if (borrow) {
        borrow = usub_partial_limb(aptr, aptr_e, borrow, dstptr);
        NUMETRON_ASSERT(!borrow);
    }
    dstptr = std::copy(aptr, aptr_e, dstptr);
    dst.sign = a.sign;
    dst.len = static_cast<size_t>(dstptr - dst.ptr);
}

template <std::unsigned_integral LimbT>
void slot_inplace_add(toom_slot<LimbT>& dst, toom_slot<LimbT> const& a)
{
    if (!a.sign) { return; }
    
    NUMETRON_ASSERT(dst.cap >= a.len);

    if (dst.sign == a.sign) {
        size_t isz = (std::min)(dst.len, a.len);
        auto carry = uadd_inplace(dst.ptr, a.ptr, a.ptr + isz);

        if (isz < a.len) {
            std::memcpy(dst.ptr + isz, a.ptr + isz, (a.len - isz) * sizeof(LimbT));
            dst.len = a.len;
        }
        if (carry) {
            carry = uadd_limb(dst.ptr + isz, dst.ptr + dst.len, carry);
            if (carry) {
                // we can store the carry if there is a storage for it, but we don't require it
                if (dst.cap >= dst.len + 1) {
                    dst.ptr[dst.len] = static_cast<LimbT>(carry);
                    ++dst.len;
                }
            }
        }
        return;
    }
    NUMETRON_ASSERT(dst.len > a.len);
    auto borrow = usub_inplace(dst.ptr, a.ptr, a.ptr + a.len);
    if (borrow) {
        borrow = usub_limb(dst.ptr + a.len, dst.ptr + dst.len, borrow);
        // just ignore the borrow
        (void)borrow;
    }
}

template <std::unsigned_integral LimbT>
inline toom_slot<LimbT> make_positive_view(std::span<const LimbT> part) noexcept
{
    return toom_slot<LimbT>{ const_cast<LimbT*>(part.data()), part.size(), part.size(), part.empty() ? 0 : 1 };
}

template <toom_ref Ref, std::unsigned_integral LimbT, size_t N, size_t M>
inline toom_slot<LimbT> resolve_ref_read(
    stage_memory_state<LimbT>& mem,
    std::span<const LimbT> u,
    std::span<const LimbT> v,
    LimbT* rb,
    size_t rsz,
    size_t chunk)
{
    constexpr toom_mem_kind kind = ref_kind(Ref);
    constexpr unsigned short expr_id = ref_expr_id(Ref);
    if constexpr (kind == toom_mem_kind::u) {
        return make_positive_view<LimbT>(resolve_input_part<N>(u, chunk, expr_id));
    } else if constexpr (kind == toom_mem_kind::v) {
        return make_positive_view<LimbT>(resolve_input_part<M>(v, chunk, expr_id));
    } else {
        static_assert(kind == toom_mem_kind::tmp || kind == toom_mem_kind::rb);
        return mem.slots[expr_id];
    }
}

template <toom_ref Ref, std::unsigned_integral LimbT>
inline toom_slot<LimbT>& resolve_ref_write(
    stage_memory_state<LimbT>& mem)
{
    static_assert(ref_kind(Ref) == toom_mem_kind::tmp || ref_kind(Ref) == toom_mem_kind::rb, "Invalid dst ref for this op");
    return mem.slots[ref_expr_id(Ref)];
}

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
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_copy(dst, s0);
    } else if constexpr (op.op == toom_op::add) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_add_signed(dst, s0, s1);
    } else if constexpr (op.op == toom_op::inplace_add) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_inplace_add(dst, s0);
    } else if constexpr (op.op == toom_op::sub) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_add_signed(dst, s0, toom_slot<LimbT>{ s1.ptr, s1.len, s1.cap, -s1.sign });
    } else if constexpr (op.op == toom_op::inplace_sub) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_inplace_add(dst, toom_slot<LimbT>{ s0.ptr, s0.len, s0.cap, -s0.sign });
    } else if constexpr (op.op == toom_op::mul_small) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_mul_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::divexact_small) {
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_divexact_small(dst, s0, static_cast<LimbT>(op.imm));
    } else if constexpr (op.op == toom_op::mul_block) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto const s1 = resolve_ref_read<op.src1, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        auto& dst = resolve_ref_write<op.dst, LimbT>(mem);
        slot_mul_dispatch(dst, s0, s1, scratch_alloc);
    } else if constexpr (op.op == toom_op::compose_shifted) {
        auto const s0 = resolve_ref_read<op.src0, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        slot_add_shifted_to_result(rb, rsz, s0, static_cast<size_t>(op.imm) * size_ctx.chunk);
    } else if constexpr (op.op == toom_op::print) {
        auto const s0 = resolve_ref_read<op.dst, LimbT, TraitsT::N, TraitsT::M>(mem, u, v, rb, rsz, size_ctx.chunk);
        print_limbs(s0.ptr, s0.cap, "print"sv);
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
    constexpr size_t tmp_slot_count = required_slot_count<traits_t::slot_layout, toom_mem_kind::tmp>();
    constexpr size_t rb_slot_count = required_slot_count<traits_t::slot_layout, toom_mem_kind::rb>();
    constexpr size_t slot_count = tmp_slot_count + rb_slot_count;
    stage_memory_state<LimbT> mem;

    const size_t slab_len = eval_size_expr_ct<traits_t::size_exprs, traits_t::slab_size_expr_id>(size_ctx);
    const size_t slots_bytes = (tmp_slot_count + rb_slot_count) * sizeof(toom_slot<LimbT>);
    const size_t slots_extra_bytes = slots_bytes + alignof(toom_slot<LimbT>) - 1;
    const size_t slots_extra_limbs = (slots_extra_bytes + sizeof(LimbT) - 1) / sizeof(LimbT);
    const size_t slab_alloc_len = slab_len + slots_extra_limbs;

    mem.slab = std::allocator_traits<ScratchAllocatorT>::allocate(scratch_alloc, slab_alloc_len);
    mem.slab_len = slab_len;
    mem.slab_alloc_len = slab_alloc_len;
    mem.slab_owned = true;

    std::byte* slots_storage = reinterpret_cast<std::byte*>(mem.slab + slab_len);
    size_t slots_storage_space = slots_extra_limbs * sizeof(LimbT);
    void* slots_ptr = slots_storage;
    if constexpr (slot_count != 0) {
        slots_ptr = std::align(alignof(toom_slot<LimbT>), slots_bytes, slots_ptr, slots_storage_space);
        NUMETRON_ASSERT(slots_ptr != nullptr);
        static_assert(std::is_trivially_destructible_v<toom_slot<LimbT>>);
        mem.slots = new (slots_ptr) toom_slot<LimbT>[slot_count];
    }

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

template <size_t N, size_t M>
struct toom_engine
{
    template <std::unsigned_integral LimbT, typename AllocatorT>
    requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
    static std::tuple<LimbT*, size_t, size_t>
    umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
    {
        using namespace toom_runtime_detail;
        static_assert(M > 0, "toom_engine requires M > 0");

        const size_t un = u.size();
        const size_t vn = v.size();

        NUMETRON_ASSERT(un > 0 && vn > 0 && un >= vn);

        const size_t alloc_sz = un + vn;
        LimbT* rb = std::allocator_traits<AllocatorT>::allocate(alloc, alloc_sz);
        try {
            LimbT* re = rb;

            const size_t chunk = (vn + (M - 1)) / M;
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
