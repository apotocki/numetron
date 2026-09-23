// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <array>
#include <vector>
#include <initializer_list>

#include "kernels.hpp"

namespace numetron::limb_arithmetic::toom_runtime_detail {

using detail::toom_kernels::lincomb_desc;

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
    print,

    // Fixed-width ops. They work on non-negative magnitudes and always write exactly dst.cap
    // limbs (dst.len = dst.cap afterwards); a source is read over its cap and zero-extended when
    // shorter than dst, so any memory view -- an input part, an rb window, a slot aliasing part
    // of another slot -- can be a source without having been written by an op first. There is
    // no trimming and no sign handling except where noted; the plan guarantees that results are
    // non-negative and fit (checked by debug asserts). src and dst may be the same slot.

    // dst <- src0 + src1
    uadd,
    // dst <- src0 - src1
    usub,
    // dst <- src0 - sign(src1) * src1   (src1's magnitude taken with the sign an abs_sub or
    // umul_fixed recorded in it)
    usub_signed,
    // dst <- |src0 - src1|, dst.sign <- sign of (src0 - src1): +1, -1, or 0 when equal
    abs_sub,
    // dst <- src0 << 1
    shl1,
    // dst <- src0 >> 1   (src0 even)
    shr1,
    // dst <- src0 / 3    (src0 a multiple of 3)
    divexact3,
    // dst <- src0 * src1, zero-padded to dst.cap (dst.cap >= src0.cap + src1.cap);
    // dst.sign <- src0.sign * src1.sign
    umul_fixed,
    // dst <- the low dst.cap limbs of src0 (the rest of src0 is deliberately ignored)
    copy_low,
    // dst += src0, the carry propagated through dst.cap
    uadd_into,
    // Evaluation at +-1 in one op: with E = src0 + src1 (the even parts) and O = src2 (the odd
    // part), dst <- E + O and dst2 <- |E - O|, dst2.sign <- sign of (E - O). dst and dst2 must be
    // distinct slots, neither aliasing a source; O must be no wider than dst2.
    eval_pm1,
    // Evaluation at +-x from ready even/odd values: dst <- src0 + src1, dst2 <- |src0 - src1|,
    // dst2.sign <- sign of (src0 - src1). dst must not alias a source; dst2 may be src0.
    addsub_abs,
    // dst <- src0 << imm   (0 < imm < limb bits)
    shl,
    // dst <- src0 >> imm   (0 < imm < limb bits; the shifted-out bits must be zero)
    shr,
    // dst <- src0 + (src1 << imm)   (0 < imm < limb bits)
    uaddlsh,
    // dst <- src0 - (src1 << imm)   (0 < imm < limb bits)
    usublsh,
    // dst <- src0 / imm   (src0 a multiple of imm; imm must divide 2^(limb bits) - 1: 3, 5, 15, ...)
    divexact,
    // dst <- ((sum over j of +-(src_j << k_j)) >> r) / d in one pass over up to four sources
    // src0..src3, the shape given by lc (see toom_kernels::lincomb_desc; built with lincomb()
    // below). A term may also take the sign its slot recorded (lc_sub_signed()). dst may be one
    // of the sources but must not overlap any other way.
    lincomb
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
    // Second destination and third source, for ops that need them (eval_pm1); appended last
    // so that positional initializers of the other ops are unaffected.
    toom_ref dst2 = {};
    toom_ref src2 = {};
    // Fourth source and the combination shape of toom_op::lincomb.
    toom_ref src3 = {};
    lincomb_desc lc = {};
};

// One term of a toom_op::lincomb: +-(src << shift).
struct toom_term
{
    toom_ref src;
    bool neg = false;
    bool slot_sign = false;
    unsigned char shift = 0;
};

// + (src << k)
[[nodiscard]] constexpr toom_term lc_add(toom_ref src, unsigned k = 0)
{
    return toom_term{ src, false, false, static_cast<unsigned char>(k) };
}

// - (src << k)
[[nodiscard]] constexpr toom_term lc_sub(toom_ref src, unsigned k = 0)
{
    return toom_term{ src, true, false, static_cast<unsigned char>(k) };
}

// - (sign(src) * src << k): subtracts the signed value a slot holds (its magnitude with the sign
// an abs_sub / addsub_abs / umul_fixed recorded), i.e. adds the magnitude when that is negative.
[[nodiscard]] constexpr toom_term lc_sub_signed(toom_ref src, unsigned k = 0)
{
    return toom_term{ src, true, true, static_cast<unsigned char>(k) };
}

// dst <- ((t0 + t1 + ...) >> rshift) / div, see toom_op::lincomb. The first term must be added.
[[nodiscard]] constexpr toom_instr lincomb(toom_ref dst, std::initializer_list<toom_term> terms, unsigned rshift = 0, unsigned div = 1)
{
    if (terms.size() < 1 || terms.size() > lincomb_desc::max_terms) throw "lincomb: 1 to 4 terms";
    if (terms.begin()->neg) throw "lincomb: the first term must be added";
    if (div < 1 || div > 0xFFFF) throw "lincomb: bad divisor";
    toom_instr in{ toom_op::lincomb, dst };
    in.lc.count = static_cast<unsigned char>(terms.size());
    in.lc.rshift = static_cast<unsigned char>(rshift);
    in.lc.div = static_cast<unsigned short>(div);
    toom_ref* srcs[lincomb_desc::max_terms] = { &in.src0, &in.src1, &in.src2, &in.src3 };
    unsigned j = 0;
    for (toom_term const& t : terms) {
        *srcs[j] = t.src;
        in.lc.neg[j] = t.neg;
        in.lc.slot_sign[j] = t.slot_sign;
        in.lc.shift[j] = t.shift;
        ++j;
    }
    return in;
}

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

// Maximum number of nodes an expr_pack can hold.
// Define NUMETRON_EXPR_PACK_MAX_NODES before including this header to override.
#ifndef NUMETRON_EXPR_PACK_MAX_NODES
#  define NUMETRON_EXPR_PACK_MAX_NODES 128
#endif
inline constexpr size_t expr_pack_max_nodes = NUMETRON_EXPR_PACK_MAX_NODES;

struct expr_pack
{
    // Fixed-capacity storage; only first 'count' nodes are valid.
    std::array<expr_node, expr_pack_max_nodes> nodes{};
    size_t count = 0;
    // Root expression handle for stage slab size.
    expr_handle root{};
};

struct expr_builder
{
    std::vector<expr_node> nodes{};

    [[nodiscard]] consteval expr_handle c(uint64_t v) { return expr_handle::constant(v); }
    [[nodiscard]] consteval expr_handle v(toom_size_var x) { return expr_handle::variable(x); }

    [[nodiscard]] consteval expr_handle emit(toom_size_expr_op op, expr_handle a, expr_handle b = {})
    {
        if (nodes.size() >= expr_pack_max_nodes) throw "expr_builder: exceeded expr_pack_max_nodes";
        nodes.push_back(expr_node{ op, a, b });
        return expr_handle::index(nodes.size() - 1);
    }

    [[nodiscard]] consteval expr_handle add(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::add, a, b); }
    [[nodiscard]] consteval expr_handle sub(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::sub, a, b); }
    [[nodiscard]] consteval expr_handle mul(expr_handle a, uint64_t k) { return emit(toom_size_expr_op::mul_const, a, c(k)); }
    [[nodiscard]] consteval expr_handle max2(expr_handle a, expr_handle b) { return emit(toom_size_expr_op::max2, a, b); }

    [[nodiscard]] consteval expr_pack finish(expr_handle root) const
    {
        expr_pack out{};
        for (size_t i = 0; i < nodes.size(); ++i) out.nodes[i] = nodes[i];
        out.count = nodes.size();
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

// Maximum number of slots a slot_pack can hold.
// Define NUMETRON_SLOT_PACK_MAX_SLOTS before including this header to override.
#ifndef NUMETRON_SLOT_PACK_MAX_SLOTS
#  define NUMETRON_SLOT_PACK_MAX_SLOTS 64
#endif
inline constexpr size_t slot_pack_max_slots = NUMETRON_SLOT_PACK_MAX_SLOTS;

struct slot_pack
{
    std::array<toom_slot_layout, slot_pack_max_slots> slots{};
    size_t count = 0;

    // size() and operator[] make slot_pack compatible with engine range loops
    // that use LayoutV.size() / LayoutV[i] on the NTTP value.
    [[nodiscard]] constexpr size_t size() const noexcept { return count; }
    [[nodiscard]] constexpr toom_slot_layout const& operator[](size_t i) const noexcept { return slots[i]; }
    [[nodiscard]] constexpr toom_slot_layout const* begin() const noexcept { return slots.data(); }
    [[nodiscard]] constexpr toom_slot_layout const* end()   const noexcept { return slots.data() + count; }
};

// slot_builder — consteval helper analogous to expr_builder.
// Each call to tmp()/rb() registers a toom_slot_layout and returns a ready-to-use
// toom_ref that can be embedded directly in toom_instr without repeating kind/id.
// var ids are assigned sequentially (globally unique within one builder instance).
// Static helpers u()/v() produce refs for u/v operand chunks.
struct slot_builder
{
    std::vector<toom_slot_layout> slots{};

    [[nodiscard]] consteval toom_ref add_slot(toom_mem_kind kind, expr_handle off, expr_handle cap)
    {
        if (slots.size() >= slot_pack_max_slots) throw "slot_builder: exceeded slot_pack_max_slots";
        auto var = static_cast<unsigned short>(slots.size());
        slots.push_back(toom_slot_layout{ kind, var, off, cap });
        return make_ref(kind, var);
    }

    [[nodiscard]] consteval toom_ref tmp(expr_handle off, expr_handle cap)
    {
        return add_slot(toom_mem_kind::tmp, off, cap);
    }

    [[nodiscard]] consteval toom_ref rb(expr_handle off, expr_handle cap)
    {
        return add_slot(toom_mem_kind::rb, off, cap);
    }

    [[nodiscard]] static consteval toom_ref u(unsigned short idx)
    {
        return make_ref(toom_mem_kind::u, idx);
    }

    [[nodiscard]] static consteval toom_ref v(unsigned short idx)
    {
        return make_ref(toom_mem_kind::v, idx);
    }

    [[nodiscard]] consteval slot_pack finish() const
    {
        slot_pack out{};
        for (size_t i = 0; i < slots.size(); ++i) out.slots[i] = slots[i];
        out.count = slots.size();
        return out;
    }
};

template <size_t>
inline constexpr bool always_false_v = false;

template <size_t N, size_t M>
struct toom_stage_traits
{
    static_assert(always_false_v<N + M>, "Missing toom_stage_traits specialization for this (N,M)");
};

template <size_t PlanSize>
struct toom_full_spec
{
    expr_pack exprs{};
    slot_pack slot_layout{};
    expr_handle slab_expr{};
    std::array<toom_instr, PlanSize> plan{};
};

// Deduction guide: toom_full_spec{ exprs, slots, slab, plan } deduces PlanSize from the array.
template <size_t PlanSize>
toom_full_spec(expr_pack, slot_pack, expr_handle, std::array<toom_instr, PlanSize>) -> toom_full_spec<PlanSize>;

} // namespace numetron::limb_arithmetic::toom_runtime_detail
