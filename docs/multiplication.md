# Multiplication: design, findings, results

Working notes on the runtime multiplication stack (`umul` and below): what is implemented, why
it looks the way it does, what was measured, and what was tried and rejected. Read this before
changing anything under `include/numetron/limb_arithmetic/` that sits on the multiplication path.

All timings are from one x86-64 machine (Alder Lake class), Release builds: MSVC with
`/arch:AVX2` (as `msvc/numetron_bench_mul.vcxproj`) and GCC 13.3 with `-O3 -march=native` (as
CMake). They are good for comparing variants, not as absolute numbers.

---

## 1. Where things stand

`numetron_bench_mul --tune` (n x n limbs, `gmp/reuse` = GMP time / numetron time with a reused
result; > 1 means numetron is faster):

| limbs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCC   | 0.69 | 0.65 | 0.68 | 0.93 | 1.20 | 1.09 | 1.08 | 1.20 | 1.14 | 1.13 | 1.17 | 1.10 | 1.11 | 1.12 | 0.86 | 0.76 |
| MSVC  | 0.60–0.73 | 0.65–0.70 | 0.54–0.59 | 0.82–0.87 | 1.10–1.13 | 1.04–1.10 | 1.06 | 1.20 | 1.11 | 1.12 | 1.10 | 1.06–1.10 | 1.06–1.09 | 1.05–1.09 | 0.80 | 0.75 |

For reference, at the start of this work numetron was at 0.65 at 4096 limbs.

- **16 .. 8192 limbs: faster than GMP** with both compilers.
- **1 .. 8 limbs: slower** (0.6–0.9). The work there is a handful of `mul` instructions; the
  cost is the call/dispatch/allocation overhead around it (see § 9).
- **12288+ limbs: slower and falling** — GMP switches to FFT (its time grows ~1.36x from 8192
  to 12288 limbs, 1.5x the size). No Toom variant closes that; it needs FFT (see § 9).

Default thresholds (limbs, `toom/thresholds.hpp`, chosen per compiler — see § 6):

| | karatsuba | toom3 | toom4 | toom6h | toom8h |
|---|---|---|---|---|---|
| MSVC | 38 | 115 | 330 | 717 | 967 |
| GCC (and Clang, untuned) | 38 | 57 | 500 | 717 | 967 |

---

## 2. The dispatch chain

`umul()` / `umul_dispatch()` in `limb_arithmetic/umul.hpp`, operands normalized to un >= vn:

1. **Toom-8.5** (balanced 8 x 8, engine plan) — `vn >= toom8h_threshold()` and
   `toom8h_split_fits(un, vn)` (v reaches u's top eighth: `vn > 7*ceil(un/8)`).
2. **Toom-6.5** (balanced 6 x 6, engine plan) — same pattern with sixths.
3. **Toom-4** (balanced 4 x 4, engine plan) — same pattern with quarters.
4. **Toom-3** — balanced (`toom3_split_fits`): the hand-written `detail::umul_toom3_impl`
   (default) or the engine's `toom3_balanced` plan (`NUMETRON_TOOM3_USE_ENGINE`); unbalanced:
   the old generic engine plan `toom_engine<3,3>`.
5. **Karatsuba** — hand-written `umul_karatsuba_impl` (`NUMETRON_EXPLICIT_KARATSUBA`, always on).
6. **Basecase** — `umul_basecase`, the GMP-derived asm `mul_basecase` (`src/arch`, LGPL,
   runtime-selected alderlake/core2/k8), with `umul_basecase_2x` as a loop-free special case for
   operands of at most 2 limbs.

Only *balanced* products get Toom-4/6.5/8.5. Anything more lopsided falls through to Toom-3
or Karatsuba (§ 9: unbalanced variants are an open item).

**Scratch memory**: `umul()` is the single place that creates the scratch allocator — a
thread-local LIFO `numetron::detail::stack_allocator` — and passes it down. Everything below
(Toom slabs, Karatsuba temporaries, all recursion levels) allocates from it; only the result
comes from the caller's allocator. Nested products write into buffers their parent provides.

**The short paths in `mul()`** (`limb_arithmetic.hpp`) handle 1 x 1 and <= 2-limb operands
without going through `umul`. Past bug worth remembering: the trimming loop after
`umul_basecase_2x` read `*pe` (one past the end) instead of `*(pe - 1)`, dropping the top limb
whenever that uninitialized limb happened to be 0 — a *flaky* mismatch against GMP.

---

## 3. The Toom engine

`toom/engine.hpp` + `core.hpp` + `slot.hpp` + `kernels.hpp`. A plan is a `consteval`-built
array of `toom_instr`, executed by a fold over `run_toom_op<..., I>` — every instruction is a
compile-time constant, so each op dispatches statically. `docs/custom-toom-plans.md` is the
how-to for writing and wiring in a plan.

### 3.1 Fixed-width ops

The fast plans use only the *fixed-width* ops (`toom_op::uadd` and later in `core.hpp`): they
work on non-negative magnitudes, always write exactly `dst.cap` limbs, and read each source over
its cap, zero-extended. So any memory view — an input part, an rb window, a slot aliasing part of
another slot — can be a source. There's no trimming and no sign logic except where noted:

- `umul_fixed` records the sign of a product of signed evaluations in `dst.sign`.
- `eval_pm1` (E = e0 + e1, E ± O) and `addsub_abs` (a + b, |a − b|) produce the ± point pairs.
- `lincomb` / `lincomb_dual` (§ 3.2) do everything linear.
- `copy_low` + `uadd_into` compose the coefficients into the result.

The balanced plans set `split_by_u` (chunk = ceil(un/N), every part <= chunk) and
`zero_result = false`: their rb windows cover the whole result, so rb isn't zeroed first.

### 3.2 `lincomb`: the fused linear-combination op

`dst = ((±(s0<<k0) ± (s1<<k1) ± (s2<<k2) ± (s3<<k3)) >> r) / d` in one streaming pass. The shape
(`lincomb_desc`) is a compile-time NTTP; built in plans with
`lincomb(dst, { lc_add(A), lc_sub(B, 4), ... }, rshift, div)`.

- Up to 4 terms, each with its own shift and its own carry chain (independent chains overlap).
  The first term must be added.
- `lc_sub_signed(S)`: subtract the *signed* value a slot holds (magnitude + sign that
  `umul_fixed`/`addsub_abs` recorded). The engine picks the kernel instantiation by the runtime
  sign.
- **Two's complement values**: `lincomb_tc` may produce a negative result (written in two's
  complement over the slot width); `lc_add_tc`/`lc_sub_tc` read such a slot sign-extended. The
  big interpolations need signed intermediates. In debug builds the kernel checks the net carry,
  accounting ±1 for every negative two's complement term.
- Right shift: arithmetic for two's complement sums (the sum is formed over w + 1 limbs, so the
  top limb supplies the sign bits).
- **Division by any odd d**, split at compile time (`plan_div`):
  - d | 2^64 − 1 (3, 5, 15, 17, 51, 85, 255, 257, …): one **dbm1** stage — 2-adic division
    with m = (B−1)/d, the multiply off the critical path, sub + sbb per limb;
  - d = a·b with both | 2^64 − 1 (9 = 3·3, 225 = 15·15, 3825 = 15·255): two dbm1 stages;
  - anything else (7, 189, 3069, 3969, 4095, …): a **Hensel** stage, q = (y − borrow)·d⁻¹ mod B,
    borrow' = hi(q·d) + borrow-out. A multiply latency per limb on the critical path, about
    4x the cost of a dbm1 pass.
  - All stages are 2-adic, so they are exact for negative (two's complement) dividends too.
- `dst` may be one of the sources (same pointer), but must not partially overlap one.

**`lincomb_dual`**: two independent lincombs of the same shape in one interleaved loop, so their
dependency chains overlap. The even and odd halves of the Toom-6.5/8.5 interpolations take
identical steps, so every step is one dual op. Whether to actually interleave is decided at
compile time (`lincomb_dual_interleaves`), from measurements:

- interleave when there is a Hensel stage (about −40% per pair);
- interleave the plain and shifted shapes (−15..25%);
- run the two streams one after the other for four unshifted terms, or for two dbm1 stages
  without Hensel — both streams' state no longer fits in registers (+3..9% if interleaved).

### 3.3 Per-op code layout (VTune findings)

- **Each op type is one `NUMETRON_NOINLINE` function** (`slot_fx_*`), not inlined per
  instruction. Inlining everything made the engine code big enough to evict the asm basecase
  from the DSB (uop cache) into the legacy decoder (MITE), which was measurable.
- **Sources are passed by const reference** (`resolve_ref_cref`), not copied. Copying the
  32-byte slot struct right after its fields were stored one by one caused a blocked store
  forward on almost every op.
- `same_tmp_width<>()` detects, at compile time, tmp slots of the same capacity expression;
  ops between them call the raw kernel with no clipping, zero-extension or tail fill.
- The slot table lives on the stack, uninitialized (`std::byte` storage + `std::launder`);
  `init_slot_state` writes every field.

---

## 4. The algorithms

Points are symmetric pairs wherever possible, so each pair splits into an even and an odd half
with one lincomb each, and the interpolation separates into two independent halves.

### Karatsuba (`umul_karatsuba.hpp`)
GMP-toom22-like split by un, hand-written. The engine's 2 x 2 plan exists but isn't used.

### Toom-3 (`umul_toom3.hpp` explicit; `toom_3x3.hpp` `toom3_balanced` engine plan)
Points 0, ±1, 2, ∞. The engine plan has 1 `eval_pm1` + 1 lincomb per operand, 5 products,
5 lincomb interpolation passes, and 4 compose ops. The two versions are now equally fast (engine
−0.6% on average, within ±3%). **Whether to switch the default to the engine is still an open
decision** — keep `NUMETRON_EXPLICIT_TOOM3` until it is made.

### Toom-4 (`toom_4x4.hpp`)
Points 0, ±1, ±2, ½, ∞. Evaluation: 6 ops per operand. Interpolation: 12 lincomb passes
(formulas in the file header). One level of Toom-4 over Toom-3 is worth at most about 8% on the
multiplication part (7·M(n/4) vs 5·M(n/3)), and measured it is 2–9% — the threshold matters
little.

### Toom-6.5 (`toom_6x6.hpp`, balanced 6 x 6 case of GMP's toom6h)
11 points: 0, ±1, ±2, ±4, ±½, ±¼ (fractional points scaled to integers). Each half is a quartic
known at y = 1, 4, 16 plus its reverse at 4, 16. It is solved through sums and differences of
mirrored coefficients: 13 `lincomb_dual` steps. The division by **189 = 27·7 cannot be avoided**:
for 5 points of the form 4^k, the Vandermonde differences always include 4^3 − 1 = 63, and 7
doesn't divide 2^64 − 1. The 12-point 6 x 7 variant (the ".5", for mildly unbalanced operands)
is not implemented.

### Toom-8.5 (`toom_8x8.hpp`, balanced 8 x 8 case of GMP's toom8h)
15 points: 0, ±1, ±2, ±4, ±8, ±½, ±¼, ±⅛. Each half is a sextic at y = 1, 4, 16, 64 plus its
reverse at 4, 16, 64, solved through U_y (antisymmetric) and G_y (symmetric) forms:
28 `lincomb_dual` steps. Divisions 189, 3069, 3969, 4095 need Hensel. **U64 and G64 are kept
undivided** (V64 = 4095·U64, H64 = 3969·G64), and their divisors are folded into Y and Y'
(divide by 4095·3069 and 3969·3069, which is why `lincomb_desc::div` is 32-bit). That leaves
4 Hensel passes per half instead of 6. Plans have 147 instructions and 69 slots; the pack limits
were raised to 256 expression nodes and 96 slots.

The derivations are in the file headers. Every new plan was verified against GMP before any
timing (§ 7).

---

## 5. Kernel costs (ns per limb, n = 1024)

| pass | MSVC | GCC |
|---|---|---|
| asm `sub_n` / `add_n` | 0.18 | 0.18 |
| lincomb `a−b` | 0.28 | 0.31 |
| `(a−b)>>1` | 0.28 | 0.33 |
| `(a−b)/3` (dbm1) | 0.43 | 0.47 |
| `(a−b)/225` (2 dbm1) | 0.54 | 0.74 |
| `(a−b)/189` (Hensel) | 1.62 | 1.66 |
| 2 x `(x−y)/189`, `lincomb_dual` (per limb of both) | 0.99 | 0.83 |
| `a−b−c−d` | 0.37 | 0.43 |
| `(a−b+c−(d<<2))>>1` | 0.62 | 0.57 |
| `a+(b<<1)+(c<<2)` | 0.64 | 0.49 |
| `(a−(b<<1)−c−(d<<6))>>2` | 0.64 | 0.61 |
| `a−(b<<6)−(c<<4)−d` | 0.69 | 0.55 |
| `(a<<3)+(b<<2)+(c<<1)+d` | 0.79 | 0.67 |

The whole linear part of a Toom-3 node is about 3–4% of its time. From here, faster kernels
buy only small overall gains; Hensel passes in the big Tooms are the most expensive linear work.

### What made the kernels fast — and what didn't

- **Blocks of 4 limbs per term** (`combine4`/`apply4`): shift the 4 source limbs, then 4
  back-to-back adc/sbb of that term's chain. The carry stays in the flags within the block and
  is saved/restored once per block. This halved `a−b` (0.55 → 0.28 on MSVC).
- **Double-width shifts**: a shifted source limb is `shld(prev, cur, K)` — `__shiftleft128` /
  `__shiftright128` on MSVC — instead of shl + shr + or + copy. The output right shift likewise
  uses shrd. GCC gets the plain `(l<<K)|(prev>>(64−K))`: an `__int128` shift made GCC *slower*.
- **dbm1 step as `sub` + `sbb`** (two dependent instructions) instead of sub/setb/sub/sub:
  `divexact_by3` 0.70 → 0.40.
- **GCC needs inline asm for the carry chains** (`chain4`, `accumulate`, the dbm1 step, all
  under `__GNUC__`/`__clang__` + x86-64). Left to itself GCC routes `_subborrow_u64` results
  through a stack slot (store + reload on every step), keeps the chains' carries in memory, and
  restores/saves the flag around every limb. The asm pins the carry to a register and spells out
  the 4-step chain. That took the GCC build from 1.00–1.09 to 1.10–1.17 against GMP.
- Tried and rejected:
  - **2-limb blocks**: fewer live values, but more flag save/restore; not better.
  - **`+` instead of `|` in shifts** (hoping for flag-free lea): MSVC turns it back into `or`.
  - **Putting the lincomb family in `.asm`/`.s` files**: lincomb is a compile-time family of
    shapes (terms, shifts, signs, divisions). External asm would mean dozens of routines, or a
    slower runtime-parameterized loop, in two assembler dialects. External asm stays for fixed
    primitives (`add_n`/`sub_n`, future FFT kernels).
- **Hensel latency**: sub → imul → mul-hi → add is inherent; tables, 32-bit splitting and
  2-limb inverses don't shorten it. The two levers that work are fewer Hensel passes (folding
  divisors) and overlapping two chains (`lincomb_dual`).

---

## 6. Thresholds and the tuner

Thresholds are runtime atomics (`set_*_threshold()`, relaxed loads) with per-compiler
compile-time defaults (`NUMETRON_DEFAULT_*` in `toom/thresholds.hpp`, overridable with
`NUMETRON_*_THRESHOLD`). They differ per compiler because everything above the asm basecase is
compiled C++.

`tune_mul_thresholds()` (`mul_tuning.hpp`; `numetron_bench_mul --tune[=samples] [--trace]`)
tunes Karatsuba, Toom-3, Toom-4, Toom-6.5 and Toom-8.5 in that order, each against the tuned
ones below. **Why it works the way it does**:

- A one-level comparison (threshold n vs n+1: only the top node changes) is **not a smooth
  curve**. The higher/lower time ratio swings in bands about an octave wide as the two
  algorithms' sub-products cross the lower thresholds at different sizes. For Toom-3 vs
  Karatsuba it wins at 60–78, 115–154, 231–330, 471+ and loses in between. Any rule that picks
  "the first win" (or "first win that lasts an octave") lands on a random band. That is why
  earlier runs gave 57, 115, 122, 231, 418.
- So the tuner works in **two phases**:
  1. The one-level scan (steps of n/16) only *proposes candidates*: the start of each winning
     band, plus the minimum of the suffix sum of log(ratio / (1 − min_gain)).
  2. Each candidate, and "higher algorithm off", is then **validated with full products** (all
     recursion levels) at sizes from the smallest candidate to the end of the range (steps of
     n/8). Measurements are interleaved per size. The least total log time wins.
- Measurements are interleaved in ABBA order, so clock drift hits all settings alike.
- **Ties**: candidates within `tie_tolerance` (0.3%) of the best count as equal, and the largest
  threshold is chosen. Toom-4's and Toom-8.5's candidates are often within 0.5% of each other,
  so their thresholds may move between runs — harmless, since that is exactly when the choice
  doesn't matter.
- The whole `--tune` takes about 25 s.

What each level buys (full product time vs. the higher algorithm off, MSVC): Karatsuba ~29%,
Toom-3 ~8–9%, Toom-4 ~5%, Toom-6.5 ~5%, Toom-8.5 ~2%.

---

## 7. How changes were verified

Rule of the project: nothing is built or run without the maintainer asking (see CLAUDE.md).
When a run is asked for, this was the procedure:

- **Correctness against GMP, debug build (all asserts on)**, before any timing. The scratch
  driver multiplied random and adversarial operands: all-ones limbs (longest carries), sparse
  extremes, small limbs. Sizes were random un up to 900–4000 and vn from 2/3·un to un, plus the
  exact split boundaries (vn = k·ceil(un/(k+1)) and +1, for k = 2, 3, 5, 7). Each algorithm was
  run under several threshold sets: its default neighbours, **minimal thresholds (deep
  recursion)**, and directly over basecase leaves. It was run in both Toom-3 modes (explicit /
  engine) and under both compilers. The last run was 9944 products, 0 mismatches.
- `numetron_tests` (gtest) in Release, both builds.
- GCC build: CMake in the `ubuntu.gcc.devel` Docker image (`-Wall -Wextra`, no warnings).
- **Performance**: prefer *node* benchmarks — one top-level node of algorithm A vs B on the same
  operands, with the children through fixed thresholds, interleaved rounds, best-of. Compare
  plan variants on the same kernels by keeping a copy of the old plan under another name. Then
  `numetron_bench_mul --tune` for the end-to-end picture. Run-to-run noise is about ±3%; never
  decide on a single point.
- VTune (run as administrator) was used for the uop-cache and store-forwarding findings in
  § 3.3.

The scratch drivers are not in the repository. What they did is described above: a GMP
comparison with `mpz_import`/`mpz_mul`, and a timing loop over `umul_dispatch` / `*_engine::umul`.

---

## 8. Pitfalls met along the way

- **GCC 13 and consteval**:
  - calling a `consteval` function through a pointer (`auto u = slot_builder::u; u(0)`) inside
    the arguments of another consteval call fails ("does not designate a constexpr function");
  - consteval helpers called in a `std::array` initializer are flagged as "taking address of an
    immediate function";
  - lambdas calling consteval members inside a consteval builder don't compile.

  Fixes: `constexpr std::array<toom_ref, N> u = {...}` with `u[i]`; plan helpers (`lincomb`,
  `lc_*`) are `constexpr`, not `consteval`; no helper lambdas in plan builders.
- **`NUMETRON_ASSERT` evaluates its argument in Release** (`((void)(expr))`) — guard expensive
  checks with `#ifndef NDEBUG`. In debug it expands to an unbraced `if` — always brace it inside
  `if constexpr ... else`.
- **IntelliSense** reports cascades of bogus errors in `kernels.hpp`/`umul.hpp`; trust the
  compiler.
- **Timing under Docker on Windows** (WSL2 VM) is noisier than native.
- Scratch builds must use the same flags as the real ones (`/arch:AVX2` for MSVC), or kernel
  timings don't match the bench.

---

## 9. Open items

1. **FFT (Schönhage–Strassen)** for 10k+ limbs — the only way to close the gap above 8192
   limbs. A separate subsystem, not an engine plan; fixed asm kernels are fine there.
2. **Small operands (1–8 limbs, 0.6–0.9 vs GMP)**: overhead around the basecase in
   `mul()`/`umul()` (dispatch, allocation, normalization).
3. **Unbalanced products**: the generic `toom_engine<3,3>` plan still uses the old slow ops.
   Candidates: a Toom-3 2 x 3-style plan (v split into 2) for un/vn ≈ 1.5–2.5, a lincomb-based
   rewrite of the generic plan, and the 6 x 7 / 8 x 9 ("half") Toom-6.5/8.5 variants. Needs a
   benchmark on unequal sizes first.
4. **Decide Toom-3 explicit vs engine** default (now equal speed).
5. Retune thresholds after any kernel change, on both compilers, and update the defaults.
