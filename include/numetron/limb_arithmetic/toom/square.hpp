// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <cstddef>
#include <array>

#include "core.hpp"

namespace numetron::limb_arithmetic::toom_runtime_detail {

// Squaring plans made from the balanced ones at compile time (u == v): A's evaluation is all
// there is to evaluate, so B's is dropped, and every pointwise product becomes the square of A's
// value there -- umul_fixed(W, EA, EA), which umul_dispatch sends down the squaring chain; the
// sign a minus point's product records is then +. Everything from the first product on
// (interpolation, composition) stays as it is.
//
// The balanced plans evaluate B with the same instructions as A, one for one (a slot of A's
// evaluation has its B counterpart at the same position; shared temporaries such as T map to
// themselves), so B's slots are mapped to A's by position. Before the first product an
// instruction belongs to B's evaluation if it reads v or a slot B's evaluation wrote; the two
// sequences must then match op for op, or the plan is not one this applies to (a compile error).

namespace square_detail {

[[nodiscard]] constexpr bool is_product(toom_op op) noexcept
{
    return op == toom_op::umul_fixed || op == toom_op::mul_block;
}

// the refs an instruction reads, whichever fields its op uses (unused ones are u(0): harmless)
[[nodiscard]] constexpr std::array<toom_ref, 8> sources(toom_instr const& in) noexcept
{
    return { in.src0, in.src1, in.src2, in.src3, in.srcb0, in.srcb1, in.srcb2, in.srcb3 };
}

[[nodiscard]] constexpr bool same(toom_ref a, toom_ref b) noexcept { return a.bits == b.bits; }

// Which instructions before the first product evaluate B: marks[i] true for them.
template <size_t K>
consteval std::array<bool, K> b_evaluation(std::array<toom_instr, K> const& plan)
{
    std::array<bool, K> marks{};
    std::array<toom_ref, 2 * K> bslots{};
    size_t nb = 0;
    auto is_b = [&](toom_ref r) {
        if (ref_kind(r) == toom_mem_kind::v) return true;
        if (ref_kind(r) != toom_mem_kind::tmp) return false;
        for (size_t j = 0; j < nb; ++j) if (same(bslots[j], r)) return true;
        return false;
    };
    for (size_t i = 0; i < K; ++i) {
        if (is_product(plan[i].op)) break;
        bool b = false;
        for (toom_ref r : sources(plan[i])) b = b || is_b(r);
        if (b) {
            marks[i] = true;
            bslots[nb++] = plan[i].dst;
            bslots[nb++] = plan[i].dst2;
        } else {
            // A temporary A's evaluation writes again after B's used it is A's from here on
            for (size_t j = 0; j < nb; ++j) {
                if (same(bslots[j], plan[i].dst) || same(bslots[j], plan[i].dst2)) bslots[j] = toom_ref{};
            }
        }
    }
    return marks;
}

template <size_t K>
consteval size_t b_count(std::array<toom_instr, K> const& plan)
{
    size_t n = 0;
    for (bool b : b_evaluation(plan)) n += b;
    return n;
}

} // namespace square_detail

// The squaring plan of a balanced plan (see above).
template <size_t KS, size_t K>
consteval std::array<toom_instr, KS> make_square_plan(std::array<toom_instr, K> const& plan)
{
    using namespace square_detail;
    const std::array<bool, K> bmark = b_evaluation(plan);

    // A's and B's evaluation, in order, paired: B's slots -> A's
    std::array<toom_ref, 2 * K> from{}, to{};
    size_t nmap = 0;
    {
        std::array<size_t, K> ai{}, bi{};
        size_t na = 0, nbi = 0;
        for (size_t i = 0; i < K && !is_product(plan[i].op); ++i) {
            if (bmark[i]) bi[nbi++] = i;
            else ai[na++] = i;
        }
        if (na != nbi) throw "make_square_plan: A's and B's evaluation differ in length";
        for (size_t j = 0; j < na; ++j) {
            toom_instr const& a = plan[ai[j]];
            toom_instr const& b = plan[bi[j]];
            if (a.op != b.op) throw "make_square_plan: A's and B's evaluation differ";
            from[nmap] = b.dst;  to[nmap++] = a.dst;
            from[nmap] = b.dst2; to[nmap++] = a.dst2;
        }
    }
    auto as_a = [&](toom_ref r) -> toom_ref {
        if (ref_kind(r) == toom_mem_kind::v) return make_ref(toom_mem_kind::u, ref_expr_id(r));
        for (size_t j = 0; j < nmap; ++j) if (same(from[j], r)) return to[j];
        return r;
    };

    std::array<toom_instr, KS> out{};
    size_t o = 0;
    for (size_t i = 0; i < K; ++i) {
        if (bmark[i]) continue;
        toom_instr in = plan[i];
        if (is_product(in.op)) {
            in.src0 = as_a(in.src0);
            in.src1 = as_a(in.src1);
            if (!same(in.src0, in.src1)) throw "make_square_plan: a product that isn't a square";
        }
        out[o++] = in;
    }
    if (o != KS) throw "make_square_plan: size mismatch";
    return out;
}

// Traits of the squaring variant of a balanced plan's traits.
template <typename BalancedTraitsT>
struct toom_square_traits
{
    static constexpr size_t plan_size = BalancedTraitsT::plan.size() - square_detail::b_count(BalancedTraitsT::plan);
    static constexpr std::array<toom_instr, plan_size> plan_storage = make_square_plan<plan_size>(BalancedTraitsT::plan);

    static constexpr auto const& plan              = plan_storage;
    static constexpr auto const& size_exprs        = BalancedTraitsT::size_exprs;
    static constexpr auto const& slot_layout       = BalancedTraitsT::slot_layout;
    static constexpr expr_handle slab_size_expr_id = BalancedTraitsT::slab_size_expr_id;
    static constexpr size_t N = BalancedTraitsT::N;
    static constexpr size_t M = BalancedTraitsT::M;
    static constexpr bool split_by_u = BalancedTraitsT::split_by_u;
    static constexpr bool zero_result = BalancedTraitsT::zero_result;
};

} // namespace numetron::limb_arithmetic::toom_runtime_detail
