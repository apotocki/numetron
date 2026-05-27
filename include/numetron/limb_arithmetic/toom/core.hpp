// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <array>

namespace numetron::limb_arithmetic::toom_runtime_detail {

enum class toom_op : unsigned char
{
    // dst <- 0
    clear,
    // dst <- src0
    copy,
    // dst <- src0 + src1
    add,
    // dst += src0
    inplace_add,
    // dst <- src0 - src1
    sub,
    // dst -= src0
    inplace_sub,
    // dst <- src0 * imm
    mul_small,
    // dst <- src0 / imm, exact division is required (remainder must be 0)
    divexact_small,
    // dst <- src0 * src1
    mul_block,
    // rb += src0 << (imm * chunk) limbs
    compose_shifted,
    // print dst for debugging (src0, src1 are ignored)
    print
};

enum class toom_mem_kind : unsigned char
{
    u = 0,
    v = 1,
    rb = 2,
    tmp = 3,
};

struct toom_ref
{
    // Packed ref: [kind:2 bits | expr/id:14 bits].
    unsigned short bits = 0;
};

[[nodiscard]] constexpr toom_ref make_ref(toom_mem_kind kind, unsigned short expr_id) noexcept
{
    return toom_ref{ static_cast<unsigned short>((static_cast<unsigned short>(kind) << 14) | (expr_id & 0x3FFF)) };
}

[[nodiscard]] constexpr toom_mem_kind ref_kind(toom_ref ref) noexcept
{
    return static_cast<toom_mem_kind>((ref.bits >> 14) & 0x3u);
}

[[nodiscard]] constexpr unsigned short ref_expr_id(toom_ref ref) noexcept
{
    return static_cast<unsigned short>(ref.bits & 0x3FFFu);
}

struct toom_instr
{
    toom_op op;
    // Destination ref (expected tmp for all ops except compose_shifted where dst is ignored)
    toom_ref dst;
    // Primary source ref
    toom_ref src0 = {};
    // Secondary source ref (used by binary ops)
    toom_ref src1 = {};
    // Immediate argument (small multiplier/divisor or compose shift index)
    unsigned short imm = 0;
    unsigned short imm2 = 0;
    unsigned short imm3 = 0;
};

enum class toom_size_var : unsigned short
{
    un,
    vn,
    chunk,
    u_hi,
    v_hi,
    d_buf_n,
};

enum class toom_size_expr_op : unsigned char
{
    constant,
    variable,
    add,
    sub,
    mul_const,
    max2,
};

using expr_id_t = uint64_t;

enum class expr_tag : uint64_t
{
    // Inline constant payload.
    constant = 0b00,
    // Inline toom_size_var payload.
    variable = 0b01,
    // Index into expr_pack::nodes.
    index = 0b10,
    // Reserved for future encoding variants.
    reserved = 0b11,
};

inline constexpr uint64_t expr_tag_shift = 62;
inline constexpr uint64_t expr_payload_mask = (uint64_t(1) << expr_tag_shift) - 1;

struct expr_handle
{
    // Tagged expression reference encoded in expr_id_t.
    expr_id_t raw{};

    [[nodiscard]] static consteval expr_handle constant(uint64_t v)
    {
        return { (uint64_t(expr_tag::constant) << expr_tag_shift) | (v & expr_payload_mask) };
    }

    [[nodiscard]] static consteval expr_handle variable(toom_size_var v)
    {
        return { (uint64_t(expr_tag::variable) << expr_tag_shift) | uint64_t(v) };
    }

    [[nodiscard]] static consteval expr_handle index(size_t i)
    {
        return { (uint64_t(expr_tag::index) << expr_tag_shift) | uint64_t(i) };
    }

    [[nodiscard]] consteval expr_tag tag() const
    {
        return static_cast<expr_tag>(raw >> expr_tag_shift);
    }

    [[nodiscard]] consteval uint64_t payload() const
    {
        return raw & expr_payload_mask;
    }
};

struct expr_node
{
    // For unary-style ops, 'b' can be ignored by evaluator.
    toom_size_expr_op op;
    expr_handle a;
    expr_handle b;
};

template <size_t N>
struct expr_pack
{
    // Fixed-capacity storage; only first 'count' nodes are valid.
    std::array<expr_node, N> nodes{};
    size_t count = 0;
    // Root expression handle for stage slab size.
    expr_handle root{};
};

template <size_t MaxNodes>
struct expr_builder
{
    std::array<expr_node, MaxNodes> nodes{};
    size_t count = 0;
    // Precondition: number of emitted nodes must not exceed MaxNodes.

    [[nodiscard]] consteval expr_handle c(uint64_t v) { return expr_handle::constant(v); }
    [[nodiscard]] consteval expr_handle v(toom_size_var x) { return expr_handle::variable(x); }

    [[nodiscard]] consteval expr_handle emit(toom_size_expr_op op, expr_handle a, expr_handle b = {})
    {
        if (count >= MaxNodes) {
            throw "expr_builder overflow";
        }
        nodes[count] = expr_node{ op, a, b };
        return expr_handle::index(count++);
    }

    [[nodiscard]] consteval expr_handle add(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::add, a, b); }
    [[nodiscard]] consteval expr_handle sub(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::sub, a, b); }
    [[nodiscard]] consteval expr_handle mul(expr_handle a, uint64_t k) { return emit(toom_size_expr_op::mul_const, a, c(k)); }
    [[nodiscard]] consteval expr_handle max2(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::max2, a, b); }

    [[nodiscard]] consteval auto finish(expr_handle root) const
    {
        expr_pack<MaxNodes> out{};
        out.nodes = nodes;
        out.count = count;
        out.root = root;
        return out;
    }
};

struct toom_slot_layout
{
    // Memory space for slot base address.
    toom_mem_kind kind;
    // Slot id inside the selected kind namespace.
    unsigned short var;
    // Offset and capacity expressions encoded as expression handles.
    expr_handle off_expr;
    expr_handle cap_expr;
};

template <size_t>
inline constexpr bool always_false_v = false;

template <size_t N, size_t M>
struct toom_stage_traits
{
    static_assert(always_false_v<N + M>, "Missing toom_stage_traits specialization for this (N,M)");
};

} // namespace numetron::limb_arithmetic::toom_runtime_detail
