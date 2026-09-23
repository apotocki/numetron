# Writing Custom Toom Plans

How to add a Toom multiplication variant to Numetron as a *plan* for the Toom engine. For the
design background, measurements and the existing plans' math, see `docs/multiplication.md`.

---

## Background

The Toom engine (`include/numetron/limb_arithmetic/toom/engine.hpp`) is table-driven. All
algorithm-specific knowledge lives in a **plan**: a `consteval`-built array of `toom_instr`,
plus a **slot layout** describing the scratch and result-buffer windows the instructions work
on, plus the **size expressions** those windows are sized by. The engine reads them through a
*traits* struct at compile time. It runs the plan as a fold over the instructions, so every
instruction is a compile-time constant and dispatches statically.

To add a variant you provide:

1. A `consteval` factory function that builds the plan with `expr_builder` and `slot_builder`
   and returns a `toom_full_spec`.
2. A traits struct exposing it, and an engine alias `toom_engine_t<YourTraits>`.
3. A split condition and a dispatch entry (plus a threshold) so `umul` actually uses it.

The engine itself needs no changes.

The existing plans are the best reference:
- `toom_3x3.hpp` — `toom3_balanced` (short, a good first read); the old generic
  `toom_stage_traits<3,3>` plan is in the same file.
- `toom_4x4.hpp`, `toom_6x6.hpp`, `toom_8x8.hpp` — the balanced Toom-4, Toom-6.5 and Toom-8.5.

---

## Key types (namespace `toom_runtime_detail`, `toom/core.hpp`)

### `expr_builder` — size expressions

Symbolic arithmetic over the runtime parameters of one multiplication node:

| Variable (`toom_size_var`) | Meaning |
|---|---|
| `un`, `vn` | limb counts of the operands (un >= vn) |
| `chunk` | limbs per piece: `ceil(un/N)` with `split_by_u`, else `ceil(vn/M)` |
| `u_hi`, `v_hi` | limbs of the top piece of u / v (`un - (N-1)*chunk`, `vn - (M-1)*chunk`) |
| `d_buf_n` | `max(chunk, u_hi)` (used by the Karatsuba plan) |

```cpp
expr_builder b;
auto c  = b.v(toom_size_var::chunk);   // variable
auto e  = b.add(c, b.c(1));            // c + 1
auto p  = b.add(b.mul(c, 2), b.c(2));  // 2c + 2
auto x  = b.sub(p, c);                 // also: b.max2(a, b)
```

### `slot_builder` — memory windows

`sb.tmp(offset, capacity)` registers a window in the node's scratch slab, `sb.rb(offset,
capacity)` a window in the result buffer; both return a `toom_ref` to use in instructions.
Slots may alias each other freely: a slot over part of another (e.g. the top 2 limbs of a
product slot), or rb windows from different offsets to the end. Slot ids are assigned in
registration order.

The operand pieces need no registration. Take them as plain refs:

```cpp
// Not `auto u = slot_builder::u;` + u(i): GCC 13 rejects calls through that pointer inside the
// arguments of lincomb() & co.
constexpr std::array<toom_ref, 3> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2) };
constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };
```

Also avoid helper lambdas in the factory that call `sb.tmp()`/`b.add()`: GCC 13 doesn't accept
consteval calls inside them. Write the calls out.

### `toom_instr`

```cpp
struct toom_instr {
    toom_op op;
    toom_ref dst, src0, src1;
    unsigned short imm, imm2, imm3;
    toom_ref dst2, src2, src3;           // ops with a second output / 3rd-4th source
    lincomb_desc lc;                     // toom_op::lincomb / lincomb_dual shape
    toom_ref srcb0, srcb1, srcb2, srcb3; // toom_op::lincomb_dual second combination
};
```

Short ops are written positionally, `toom_instr{ toom_op::uadd, dst, a, b }`. Ops that use the
later fields need designated initializers:
`toom_instr{ .op = toom_op::addsub_abs, .dst = EA2, .src0 = EAM2, .src1 = T, .dst2 = EAM2 }`.
`lincomb` instructions come from their builder functions (below).

---

## Operations

There are two families. **New plans should use the fixed-width ops.** The generic ones are
what the old plans (`toom_stage_traits<2,2>`, `<3,3>`) are made of; they trim, track signs and
are much slower.

### Fixed-width ops (recommended)

They work on non-negative magnitudes and always write exactly `dst.cap` limbs. Each source is
read over its cap, zero-extended, and whatever lies beyond `dst.cap` must be zero. So any window
can be a source without having been written by an op first. `dst` may be a source unless noted.
Debug builds check that results are non-negative and fit.

| Op | Semantics |
|---|---|
| `uadd`, `usub` | `dst = src0 ± src1` |
| `usub_signed` | `dst = src0 − sign(src1)·src1` (the sign recorded in the slot) |
| `abs_sub` | `dst = |src0 − src1|`, `dst.sign` = sign of the difference |
| `shl1`, `shr1`, `shl`, `shr` | shifts by 1 / by `imm` bits (shifted-out bits must be zero for `shr`) |
| `uaddlsh`, `usublsh` | `dst = src0 ± (src1 << imm)` |
| `divexact3`, `divexact` | exact division by 3 / by `imm` (a divisor of 2^64 − 1) |
| `umul_fixed` | `dst = src0 · src1`, zero-padded to `dst.cap` (>= sum of the caps); `dst.sign = src0.sign · src1.sign` |
| `copy_low` | `dst` = the low `dst.cap` limbs of `src0` (the rest deliberately ignored) |
| `uadd_into` | `dst += src0`, carry propagated through `dst.cap` |
| `eval_pm1` | `E = src0 + src1`; `dst = E + src2`, `dst2 = |E − src2|` with sign — evaluation at ±1 |
| `addsub_abs` | `dst = src0 + src1`, `dst2 = |src0 − src1|` with sign (`dst2` may be `src0`) |
| `lincomb` | fused linear combination, see below |
| `lincomb_dual` | two same-shaped lincombs in one interleaved pass, see below |

`lincomb` covers everything `uadd`…`divexact` do, usually in fewer passes. The older
single-purpose ops remain for existing plans.

### `lincomb`

`dst = ((±(s0<<k0) ± (s1<<k1) ± (s2<<k2) ± (s3<<k3)) >> rshift) / div` in one streaming pass:

```cpp
lincomb(W2, { lc_add(W2), lc_sub(WM2, 1), lc_sub(C0), lc_sub(C6, 6) }, /*rshift*/ 2)   // (W2 - 2 WM2 - C0 - 64 C6) / 4
lincomb(W2, { lc_add(W2), lc_sub(W1) }, 0, /*div*/ 3)                                    // (W2 - W1) / 3
lincomb(EAH, { lc_add(u[0], 3), lc_add(u[1], 2), lc_add(u[2], 1), lc_add(u[3]) })       // 8 a0 + 4 a1 + 2 a2 + a3
```

- 1 to 4 terms. The first must be added (`lc_add` / `lc_add_tc`) and may be shifted.
- `lc_add(s, k)` / `lc_sub(s, k)`: ±(s << k), k < 64.
- `lc_sub_signed(s, k)`: subtract the *signed* value a slot holds (magnitude plus the sign
  `umul_fixed` / `addsub_abs` / `eval_pm1` recorded). Used to split r(x) / r(−x) pairs.
- `rshift`: exact right shift (the shifted-out bits must be zero). Put powers of two here,
  not in `div`.
- `div`: any **odd** divisor, 32-bit. The kernel picks the stages at compile time:
  - one 2-adic *dbm1* stage if div | 2^64 − 1 (3, 5, 15, 17, 51, 85, 255, 257, …);
  - two such stages if div factors that way (9, 225, 3825);
  - otherwise a Hensel stage, about 4x slower per limb (7, 189, 3069, …). Keep those few:
    divisors that are only needed in later steps can be folded into one division, see
    `toom_8x8.hpp`.
- **Negative intermediates**: `lincomb_tc(...)` allows a negative result, written in two's
  complement over `dst.cap` limbs. Read such a slot back with `lc_add_tc` / `lc_sub_tc`
  (sign-extended), never with the plain `lc_*`. Right shifts and divisions of two's complement
  values are exact as well.
- `dst` may be one of the sources (same slot), but must not partially overlap one.

### `lincomb_dual`

```cpp
lincomb_dual(W4,  { lc_add(W4), lc_sub(W2, 2) },
             WM4, { lc_add(WM4), lc_sub(WM2, 2) }, 0, 189)      // lincomb_dual_tc for signed results
```

Two independent combinations of the same shape: same term count, shifts, signs and `tc` flags;
only the slots differ; no `lc_sub_signed`. They run in one loop, so their dependency chains
overlap. This matters most for Hensel divisions, which are latency-bound. The even and odd
halves of a symmetric-point interpolation are exactly this. Neither destination may touch the
other combination's slots. Whether the two streams are actually interleaved is decided per shape
by the kernel.

---

## The traits struct

```cpp
struct toom6h_balanced_traits
{
    static constexpr auto const& plan              = toom6h_balanced.plan;
    static constexpr auto const& size_exprs        = toom6h_balanced.exprs;
    static constexpr auto const& slot_layout       = toom6h_balanced.slot_layout;
    static constexpr expr_handle slab_size_expr_id = toom6h_balanced.slab_expr;
    static constexpr size_t N = 6;                 // pieces of u
    static constexpr size_t M = 6;                 // pieces of v
    static constexpr bool split_by_u = true;       // optional, default false
    static constexpr bool zero_result = false;     // optional, default true
};
```

- `split_by_u`: chunk = ceil(un/N) instead of ceil(vn/M). Every piece of both operands is then
  <= chunk, and the top pieces are the remainders. The dispatch must make sure v still reaches
  its top piece (a `*_split_fits(un, vn)` check).
- `zero_result`: whether rb is zeroed before the plan runs. A plan that writes every limb of rb
  itself sets it to false: its rb windows (C0, R2, ..., the top one) cover rb, they are written
  with `umul_fixed` / `copy_low`, and only then accumulated into with `uadd_into`.

The engine is `toom_engine_t<YourTraits>`. The legacy route — specializing
`toom_stage_traits<N, M>` and using `toom_engine<N, M>` — still works, and the generic Toom-2
and Toom-3 plans use it.

---

## Worked example: the balanced Toom-3 plan

`toom_3x3.hpp`, `make_toom3_balanced()` (abridged). u = a0 + a1 W + a2 W², same for v with b's,
W = B^c; points 0, ±1, 2, ∞.

```cpp
consteval auto make_toom3_balanced()
{
    expr_builder b;
    auto c  = b.v(toom_size_var::chunk);
    auto s  = b.v(toom_size_var::u_hi);
    auto t  = b.v(toom_size_var::v_hi);
    auto rn = b.add(b.v(toom_size_var::un), b.v(toom_size_var::vn));
    auto e  = b.add(c, b.c(1));             // an evaluated operand: c limbs + carry
    auto p  = b.add(b.mul(c, 2), b.c(2));   // a product of two of them
    auto st = b.add(s, t);
    auto two_c = b.mul(c, 2);

    auto off_w1 = b.mul(e, 6);
    auto off_wm1 = b.add(off_w1, p), off_w2 = b.add(off_wm1, p);
    auto slab = b.add(off_w2, p);           // total scratch of the node

    slot_builder sb;
    auto EA1 = sb.tmp(b.c(0), e);  auto EAM1 = sb.tmp(e, e);  auto EA2 = sb.tmp(b.mul(e, 2), e);
    auto EB1 = sb.tmp(b.mul(e, 3), e);  auto EBM1 = sb.tmp(b.mul(e, 4), e);  auto EB2 = sb.tmp(b.mul(e, 5), e);
    auto W1 = sb.tmp(off_w1, p);  auto WM1 = sb.tmp(off_wm1, p);  auto W2 = sb.tmp(off_w2, p);
    auto W1HI = sb.tmp(b.add(off_w1, two_c), b.c(2));   // alias: c2's limbs above 2c

    auto C0 = sb.rb(b.c(0), two_c);                     // c0 = a0 b0
    auto R2 = sb.rb(two_c, two_c);                      // c2's low 2c limbs
    auto C4 = sb.rb(b.mul(c, 4), st);                   // c4 = a2 b2
    auto R1 = sb.rb(c, b.sub(rn, c));                   // accumulation windows for c1, c3
    auto R3 = sb.rb(b.mul(c, 3), b.sub(rn, b.mul(c, 3)));

    constexpr std::array<toom_ref, 3> u = { slot_builder::u(0), slot_builder::u(1), slot_builder::u(2) };
    constexpr std::array<toom_ref, 3> v = { slot_builder::v(0), slot_builder::v(1), slot_builder::v(2) };

    std::array plan = {
        // evaluate
        toom_instr{ .op = toom_op::eval_pm1, .dst = EA1, .src0 = u[0], .src1 = u[2], .dst2 = EAM1, .src2 = u[1] },
        lincomb(EA2, { lc_add(u[0]), lc_add(u[1], 1), lc_add(u[2], 2) }),
        toom_instr{ .op = toom_op::eval_pm1, .dst = EB1, .src0 = v[0], .src1 = v[2], .dst2 = EBM1, .src2 = v[1] },
        lincomb(EB2, { lc_add(v[0]), lc_add(v[1], 1), lc_add(v[2], 2) }),
        // multiply (the sign of r(-1) is kept in WM1)
        toom_instr{ toom_op::umul_fixed, W1,  EA1,  EB1  },
        toom_instr{ toom_op::umul_fixed, WM1, EAM1, EBM1 },
        toom_instr{ toom_op::umul_fixed, W2,  EA2,  EB2  },
        toom_instr{ toom_op::umul_fixed, C0,  u[0], v[0] },
        toom_instr{ toom_op::umul_fixed, C4,  u[2], v[2] },
        // interpolate
        lincomb(W2,  { lc_add(W2), lc_sub_signed(WM1) }, 0, 3),                 // c1 + c2 + 3c3 + 5c4
        lincomb(WM1, { lc_add(W1), lc_sub_signed(WM1) }, 1),                    // c1 + c3
        lincomb(W2,  { lc_add(W2), lc_sub(W1), lc_add(C0), lc_sub(C4, 2) }, 1), // c3
        lincomb(W1,  { lc_add(W1), lc_sub(C0), lc_sub(WM1), lc_sub(C4) }),      // c2
        toom_instr{ toom_op::usub, WM1, WM1, W2 },                              // c1
        // compose
        toom_instr{ toom_op::copy_low,  R2, W1   },
        toom_instr{ toom_op::uadd_into, C4, W1HI },
        toom_instr{ toom_op::uadd_into, R1, WM1  },
        toom_instr{ toom_op::uadd_into, R3, W2   },
    };
    return toom_full_spec{ b.finish(slab), sb.finish(), slab, plan };
}

static constexpr auto toom3_balanced = make_toom3_balanced();
// + toom3_balanced_traits as above (N = M = 3, split_by_u = true, zero_result = false)
```

### Wiring a new plan in

1. **Header** `toom/toom_KxK.hpp` with the factory, the `static constexpr` spec, the traits and
   a `detail::toomK_split_fits(un, vn)`. The balanced plans need v to reach u's top piece:
   `vn > (K-1)·ceil(un/K)`.
2. **`engine.hpp`**: `#include` it and add `using toomK_..._engine = toom_engine_t<...traits>;`.
3. **Threshold** (`toom/thresholds.hpp`): a `NUMETRON_DEFAULT_TOOMK_THRESHOLD` per compiler, an
   overridable `NUMETRON_TOOMK_THRESHOLD`, a `min_toomK_threshold`, the atomic, and the
   getter/setter.
4. **Dispatch** (`umul.hpp`): an `is_toomK_applicable(un, vn)`, checked in **both**
   `umul_dispatch()` (recursion) and `umul()` (top level, where the scratch allocator is
   created), in order from the highest algorithm down.
5. **Tuner** (`mul_tuning.hpp`): a `toomK_max` option, a result field, a tuning phase after the
   algorithm below it; print it in `bench/mul_bench.cpp` `--tune`.
6. **Project files**: add the header to `msvc/numetron.vcxproj` and `.vcxproj.filters` (CMake
   doesn't list headers).

---

## Capacity rules

- An **evaluated operand** is a sum of pieces times small powers of two: size it
  `c + 1`. That holds as long as the coefficient sum (e.g. 1365 for 4^0..4^5, ~2^21 for 8^0..8^7)
  fits in the extra limb.
- A **product slot** needs the sum of its factors' caps (`umul_fixed` asserts it): `2c + 2`.
- **Interpolation values** live in the product slots, and every intermediate must fit there.
  Leave headroom for the largest multiplier you apply (shifted terms) and, with `lincomb_tc`,
  for the sign bit. `lincomb` forms the sum over `w + 1` limbs, so a transient overflow by one
  limb before the shift/division is fine; the result must fit in `w`.
- **rb windows**: the coefficient pieces at k·c, each `2c` limbs (the top one `s + t`), plus
  accumulation windows from k·c to the end. A coefficient's limbs above `2c` go through a 2-limb
  alias slot (`W1HI`) added into the next window.
- `tmp` windows must stay within the slab; `rb` windows within `[0, un + vn)`.

---

## Limits

A plan may have up to `NUMETRON_EXPR_PACK_MAX_NODES` (default 256) size-expression nodes and
`NUMETRON_SLOT_PACK_MAX_SLOTS` (default 96) slots. Toom-8.5 needs ~100 nodes and 69 slots. To
raise the limits, define them before including any Numetron header.

---

## Debugging and verification

- `toom_instr{ toom_op::print, slot }` dumps a slot's limbs to **stdout**, in every build type.
  Remove it before committing.
- Debug builds check a lot, so run new plans in Debug first:
  - clipped source limbs are zero;
  - non-negative lincomb results don't carry out;
  - product slots are big enough;
  - slot ids are unique and dense (`has_unique_slot_vars`, at compile time).
- Verify against GMP **before any timing**:
  - random and adversarial operands (all-ones limbs for maximal carries, sparse extremes);
  - sizes on both sides of the split boundary;
  - thresholds set so the new plan recurses into itself with minimal lower thresholds (deep
    recursion), and also sits directly on basecase leaves.

  See `docs/multiplication.md` § 7.
- Compile with GCC too (see the consteval notes above), and keep in mind that the kernels are
  tuned per compiler.

---

## Checklist

- [ ] The factory is `consteval` and returns `toom_full_spec{ b.finish(slab), sb.finish(), slab, plan }`.
- [ ] Operand refs are `constexpr std::array<toom_ref, N>`; the factory has no helper lambdas.
- [ ] Only fixed-width ops / `lincomb` are used; negative intermediates go through
      `lincomb_tc` and are read with `lc_*_tc`.
- [ ] Hensel divisors (odd, not dividing 2^64 − 1) are as few as possible; independent halves
      use `lincomb_dual`.
- [ ] Every slot's capacity holds every value written into it; the rb windows cover rb if
      `zero_result = false`.
- [ ] Traits declare `split_by_u` / `zero_result` as intended; the engine alias exists.
- [ ] The header is included in `engine.hpp` and listed in the `.vcxproj` + `.filters`.
- [ ] Threshold, `*_split_fits`, `is_*_applicable` in both `umul_dispatch()` and `umul()`,
      tuner phase, bench print.
- [ ] Verified against GMP in Debug (deep recursion, boundaries), then timed.
