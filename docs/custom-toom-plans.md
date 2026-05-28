# Writing Custom Toom Plans

This guide explains how to add a new Toom multiplication variant to Numetron
by specialising `toom_stage_traits<N, M>` with a custom plan.

---

## Background

The Toom engine (`include/numetron/limb_arithmetic/toom/engine.hpp`) is fully
table-driven. All algorithm-specific knowledge lives in a plan — a
`constexpr` array of `toom_instr` instructions — and a matching slot layout
that describes the scratch and result-buffer windows those instructions
operate on. The engine reads both through `toom_stage_traits<N, M>` at
compile time and executes them at runtime.

You never need to touch the engine itself; you only need to provide:

1. A `consteval` factory function that builds the plan with `expr_builder` and `slot_builder`.
2. A `toom_stage_traits<N, M>` specialisation that exposes the result.
3. (Optional) A dispatcher entry that tells `umul_dispatch` when to use the new variant.

---

## Key types (all in `toom_runtime_detail`)

### `expr_builder` / `expr_pack`

`expr_builder` is a `consteval` accumulator for **size expressions** — symbolic
arithmetic over the runtime parameters of a multiplication call.

Available variables (`toom_size_var`):

| Variable   | Meaning                                    |
|------------|--------------------------------------------|
| `un`       | total limb count of operand u              |
| `vn`       | total limb count of operand v              |
| `chunk`    | limb count of one piece (`= ceil(vn / M)`) |
| `u_hi`     | limb count of the last piece of u          |
| `v_hi`     | limb count of the last piece of v          |
| `d_buf_n`  | capacity needed for an absolute-difference buffer (Karatsuba only) |

Builder methods:

```cpp
expr_builder b;

auto x = b.v(toom_size_var::chunk);   // variable reference
auto k = b.c(3);                       // compile-time constant
auto e = b.add(x, k);                 // x + 3
auto f = b.sub(e, b.c(1));            // x + 2
auto g = b.mul(x, 4);                 // 4*x
auto h = b.max2(x, b.c(1));           // max(x, 1)

expr_pack pack = b.finish(slab_expr); // materialise; slab_expr is the
									   // total scratch size expression
```

`b.finish()` returns an `expr_pack` (fixed-capacity NTTP-safe struct).

### `slot_builder` / `slot_pack`

`slot_builder` registers named memory windows and returns a `toom_ref` handle
for each one. The handle encodes both the memory space (`tmp` or `rb`) and the
slot index, so it can be used directly in instruction fields without repeating
the kind.

```cpp
slot_builder sb;

// Scratch (tmp) window: sb.tmp(offset_expr, capacity_expr)
auto W1 = sb.tmp(off_w1, cap_w1);   // returns toom_ref for tmp slot

// Result-buffer (rb) window: sb.rb(offset_expr, capacity_expr)
auto R0 = sb.rb(b.c(0), two_c);     // returns toom_ref for rb slot

// u/v operand chunk refs (no registration needed)
auto u = slot_builder::u;  // shorthand for make_ref(toom_mem_kind::u, idx)
auto v = slot_builder::v;

slot_pack layout = sb.finish();      // materialise
```

Slot indices are assigned automatically in registration order; you never
assign or track numbers manually.

### `toom_instr`

Each instruction has the form:

```cpp
toom_instr{ op, dst, src0, src1, imm }
```

Available operations (`toom_op`):

| Op                | Semantics                           | Fields used           |
|-------------------|-------------------------------------|-----------------------|
| `copy`            | `dst ← src0`                        | dst, src0             |
| `clear`           | `dst ← 0`                           | dst                   |
| `add`             | `dst ← src0 + src1`                 | dst, src0, src1       |
| `inplace_add`     | `dst += src0`                       | dst, src0             |
| `sub`             | `dst ← src0 - src1` (signed result) | dst, src0, src1       |
| `inplace_sub`     | `dst -= src0`                       | dst, src0             |
| `mul_small`       | `dst ← src0 × imm`                  | dst, src0, imm        |
| `divexact_small`  | `dst ← src0 / imm` (exact)          | dst, src0, imm        |
| `mul_block`       | `dst ← src0 × src1` (full multiply) | dst, src0, src1       |
| `compose_shifted` | `rb += src0 << (imm × chunk)` limbs | dst=rb slot, src0, imm|
| `print`           | debug print of dst                  | dst                   |

`imm` is an `unsigned short`. `src1` can be left default-constructed `{}` for
unary ops.

### `toom_full_spec` and `toom_stage_traits`

`toom_full_spec<PlanSize>` is the aggregate returned by your factory function:

```cpp
struct toom_full_spec<PlanSize> {
	expr_pack                    exprs;
	slot_pack                    slot_layout;
	expr_handle                  slab_expr;
	std::array<toom_instr, N>    plan;
};
```

`toom_stage_traits<N, M>` is the interface the engine reads:

```cpp
template <>
struct toom_stage_traits<N, M> {
	static constexpr auto const& plan              = mySpec.plan;
	static constexpr auto const& size_exprs        = mySpec.exprs;
	static constexpr auto const& slot_layout       = mySpec.slot_layout;
	static constexpr expr_handle slab_size_expr_id = mySpec.slab_expr;
	static constexpr size_t N = ...;
	static constexpr size_t M = ...;
};
```

---

## Step-by-step: adding Toom-2×3

As a concrete example, here is a skeleton for an asymmetric Toom-2×3 variant
(u split into 2 pieces, v split into 3 pieces).

### 1. Create the file

`include/numetron/limb_arithmetic/toom/toom_2x3.hpp`

```cpp
#pragma once
#include "core.hpp"

namespace numetron::limb_arithmetic::toom_runtime_detail {

consteval auto make_toom2x3()
{
	expr_builder b;

	// ── size variables ──────────────────────────────────────────────────────
	auto c   = b.v(toom_size_var::chunk);   // ceil(vn / 3)
	auto h   = b.v(toom_size_var::u_hi);    // un - 1*c  (last piece of u)
	auto g   = b.v(toom_size_var::v_hi);    // vn - 2*c  (last piece of v)

	// ── capacity expressions ────────────────────────────────────────────────
	// (fill in based on your algorithm analysis)
	auto cap_ea  = b.add(b.add(c, h), b.c(1));   // u0 + u1
	auto cap_eb1 = b.add(b.mul(c, 3), b.c(1));   // example
	// ... more capacities ...
	auto slab = /* total scratch needed */  b.c(0); // replace with real expr

	// ── slot layout ─────────────────────────────────────────────────────────
	slot_builder sb;

	// tmp slots (scratch buffer windows)
	auto EA   = sb.tmp(b.c(0),  cap_ea);
	// ... more tmp slots ...

	// rb slots (result-buffer windows)
	auto R0   = sb.rb(b.c(0), b.mul(c, 2));

	auto u = slot_builder::u;
	auto v = slot_builder::v;

	// ── instruction plan ────────────────────────────────────────────────────
	std::array plan = {
		// evaluate, multiply, interpolate, compose ...
		toom_instr{ toom_op::mul_block, R0, u(0), v(0) },
		// ...
	};

	return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom2x3 = make_toom2x3();

template <>
struct toom_stage_traits<2, 3>
{
	static constexpr auto const& plan              = toom2x3.plan;
	static constexpr auto const& size_exprs        = toom2x3.exprs;
	static constexpr auto const& slot_layout       = toom2x3.slot_layout;
	static constexpr expr_handle slab_size_expr_id = toom2x3.slab_expr;
	static constexpr size_t N = 2;
	static constexpr size_t M = 3;
};

} // namespace numetron::limb_arithmetic::toom_runtime_detail
```

### 2. Include it in `engine.hpp`

```cpp
// engine.hpp
#include "toom_2x2.hpp"
#include "toom_3x3.hpp"
#include "toom_2x3.hpp"   // ← add this line
```

### 3. Add a dispatch condition

In `limb_arithmetic.hpp` (or wherever `umul_dispatch` selects the algorithm),
add a size guard before the existing Toom-3×3 fallback:

```cpp
if (un >= 2 * chunk && vn >= 3 * chunk) {
	auto [ptr, len, cap] = toom_engine<2, 3>::umul(...);
	...
}
```

Adjust thresholds to match what is actually profitable for your variant.

---

## Capacity budget rules

Each slot must be large enough to hold the result of any operation that writes
into it. A practical approach:

- **Evaluation slots** (EA, EB): bound by the sum of input pieces plus a
  carry limb: `sum_of_pieces + 1`.
- **Product slots** (W): bound by the sum of the capacities of the two
  evaluation slots that feed them.
- **Interpolation slots** (R, T): bound by `un + vn + 1` in the worst case.
- **rb slots**: bound by the corresponding coefficient window in the result
  buffer.

Always add `+1` guard limbs for carry propagation in addition and
multiplication-by-small operations.

---

## Overriding capacity limits

If your plan needs more than 128 size-expression nodes or more than 64 slots,
define the corresponding macros **before** including any Numetron header:

```cpp
#define NUMETRON_EXPR_PACK_MAX_NODES 256
#define NUMETRON_SLOT_PACK_MAX_SLOTS 128
#include <numetron/limb_arithmetic.hpp>
```

---

## Debugging

Add `toom_instr{ toom_op::print, mySlot }` anywhere in your plan to dump the
current value of a slot to stderr at runtime (debug builds only; the `print`
op is a no-op in release).

---

## Checklist

- [ ] Factory function is `consteval` and returns `toom_full_spec{...}`.
- [ ] `slab` expression is the **total** scratch size; it is passed to both
	  `b.finish(slab)` and stored as `slab_expr` in the spec.
- [ ] Every `rb` slot offset + capacity stays within `[0, un+vn)`.
- [ ] Every `tmp` slot offset + capacity stays within `[0, slab)` and no two
	  slots overlap (checked by `has_unique_slot_vars` at compile time).
- [ ] `toom_stage_traits<N, M>` is specialised inside
	  `numetron::limb_arithmetic::toom_runtime_detail`.
- [ ] The new header is `#include`d in `engine.hpp`.
- [ ] A dispatch condition is added so the engine is actually called.
