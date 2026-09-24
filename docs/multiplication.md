# Multiplication: design, findings, results

Working notes on the runtime multiplication stack (`umul` and below): what is implemented, why
it looks the way it does, what was measured, and what was tried and rejected. Read this before
changing anything under `include/numetron/limb_arithmetic/` that sits on the multiplication path.

All timings are from one x86-64 machine (Alder Lake class), Release builds: MSVC with
`/arch:AVX2` (as `msvc/numetron_bench_mul.vcxproj`) and GCC 13.3 with `-O3 -march=native` (as
CMake). They are good for comparing variants, not as absolute numbers.

---

## 1. Where things stand

`numetron_bench_mul` with the default configuration (`NUMETRON_USE_ASM`: asm basecase and asm
Karatsuba; AVX2 FFT; the default thresholds below), n x n limbs, `gmp/reuse` = GMP time /
numetron time with a reused result; > 1 means numetron is faster (2026-09-24):

| limbs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 3072 | 4096 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCC  | 0.70 | 0.57 | 0.68 | 0.95 | 1.16 | 1.15 | 1.18 | 1.26 | 1.20 | 1.17 | 1.16 | 1.15 | 1.39 | 1.59 | 1.98 | 1.57 | 1.61 |
| MSVC | 0.63 | 0.71 | 0.53 | 0.82 | 1.12 | 1.16 | 1.18 | 1.26 | 1.19 | 1.16 | 1.15 | 1.16 | 1.33 | 1.51 | 1.79 | 1.58 | 1.54 |

At the start of this work numetron was at 0.65 at 4096 limbs; before the FFT, 0.75–0.86 at
12288–16384.

- **16 limbs and up: faster than GMP** with both compilers — 1.15–1.26 in the Karatsuba / Toom
  range, 1.33–1.98 from the FFT threshold (~2.5–2.7k limbs) up; `docs/fft.md` has the FFT up to
  524288 limbs (0.45–0.86 of GMP's time there).
- **1 .. 8 limbs: slower** (0.5–0.95). The work there is a handful of `mul` instructions; the
  cost is the call/dispatch/allocation overhead around it (see § 9).

Default thresholds (limbs, `toom/thresholds.hpp`, chosen per compiler and implementation — § 6):

| | karatsuba | toom3 | toom4 | toom6h | toom8h | fft |
|---|---|---|---|---|---|---|
| asm Karatsuba (default with `NUMETRON_USE_ASM`), MSVC | 29 | 115 | 444 | 717* | 967* | |
| asm Karatsuba, GCC (and Clang, untuned) | 29 | 115 | 444 | 675 | 967 | |
| C++ Karatsuba, MSVC | 38 | 115 | 330 | 717 | 967 | |
| C++ Karatsuba, GCC (and Clang, untuned) | 38 | 57 | 500 | 717 | 967 | |
| FFT, AVX2 kernel (default with AVX2), MSVC / GCC | | | | | | 2696 / 2538 |
| FFT, scalar kernel, MSVC / GCC | | | | | | 11530 / 13828 |

\* `--tune` gave 2249 / 2249 on MSVC once (and 1231 / 1766, 761 / 1766 in other runs), but the
measured node curves are the same as GCC's (§ 4, Toom-6.5), so the thresholds are set by the
curves, not by single runs. The FFT thresholds were the same in two runs each.

---

## 2. The dispatch chain

`umul()` / `umul_dispatch()` in `limb_arithmetic/umul.hpp`, operands normalized to un >= vn:

0. **FFT** (`umul_fft.hpp`, 64-bit limbs) — `vn >= fft_threshold()`, any shape: a multi-prime
   number-theoretic transform, 2 limbs per coefficient; `NUMETRON_FFT_IMPL` picks the AVX2 + FMA
   kernel (default when the compiler targets AVX2) or the portable scalar one. Design,
   measurements and history: `docs/fft.md`.
1. **Toom-8.5** (balanced 8 x 8, engine plan) — `vn >= toom8h_threshold()` and
   `toom8h_split_fits(un, vn)` (v reaches u's top eighth: `vn > 7*ceil(un/8)`).
2. **Toom-6.5** (balanced 6 x 6, engine plan) — same pattern with sixths.
3. **Toom-4** (balanced 4 x 4, engine plan) — same pattern with quarters.
4. **Toom-3** — balanced (`toom3_split_fits`): `NUMETRON_TOOM3_IMPL` picks the hand-written
   `detail::umul_toom3_impl` (`_CXX`, default) or the engine's `toom3_balanced` plan (`_ENGINE`);
   unbalanced: the old generic engine plan `toom_engine<3,3>`.
5. **Karatsuba** — `NUMETRON_KARATSUBA_IMPL` picks the all-assembly `umul_karatsuba_asm_impl`
   (`_ASM`, x86-64 only; the default with `NUMETRON_USE_ASM`) (§ 4), the hand-written
   `umul_karatsuba_impl` (`_CXX`, the default without the assembly), `umul_karatsuba_fused_impl`
   (`_FUSED`), or the engine's 2 x 2 plan (`_ENGINE`).
6. **Basecase** — `umul_basecase`, the GMP-derived asm `mul_basecase` (`src/arch`, LGPL,
   runtime-selected alderlake/core2/k8), with `umul_basecase_2x` as a loop-free special case for
   operands of at most 2 limbs.

The implementation choices (and their defaults) are all in `numetron/config/implementation.hpp`;
override one per build with a compiler flag, e.g.
`-DNUMETRON_KARATSUBA_IMPL=NUMETRON_KARATSUBA_IMPL_FUSED` (shorthands: `NUMETRON_KARATSUBA_FUSED`,
`NUMETRON_KARATSUBA_ASM`, `NUMETRON_TOOM3_USE_ENGINE`). `numetron_bench_mul` prints the ones its
build uses.

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

### Karatsuba (`umul_karatsuba.hpp`, `umul_karatsuba_fused.hpp`, `umul_karatsuba_asm.hpp`)
GMP-toom22-like split by un. Three implementations of the same algorithm: the hand-written C++
one, the fused variant and the all-assembly one — the default wherever the assembly is in use
(`NUMETRON_USE_ASM` on x86-64). The engine's 2 x 2 plan exists but isn't used.

**Where a Karatsuba node's time goes** (one level, both halves basecase, both compilers alike;
scratch `karatsuba_cost.cpp`):

| n | \|a0−a1\|, \|b0−b1\| | interpolation | rest (dispatch, allocator, memsets; noisy) |
|---|---|---|---|
| 38–40 | 3.3–3.6% | 5.7–6.5% | 2–8% |
| 64 | 1.6–1.8% | 3.4–3.9% | 2–7% |
| 114 | 0.9% | 2.0% | 1–3% |

The rest is the three basecase products. The interpolation is 4 passes (3 add_n over the middle
columns, 1 add/sub_n of vm1 over 2n), as in GMP; most of them already run the asm
`numetron_add_n`/`numetron_sub_n`. So no rewrite of the node can win much on its own; the
all-assembly variant below measures what removing all of it (plus the glue) gives.

**Fused variant** (`umul_karatsuba_fused.hpp`, `NUMETRON_KARATSUBA_IMPL_FUSED`): the same algorithm,
but the three middle-column add passes are one pass, `detail::karatsuba_interp` — the MIT asm
`numetron_karatsuba_interp` (`src/arch/x86_64/karatsuba_interp.{s,asm}`) with `NUMETRON_USE_ASM`,
a C++ loop otherwise. It keeps three carry chains (X = H0 + Li, X + L0, X + Hi) in byte registers
between 4-limb blocks and reads 4 / writes 2 limbs per index instead of 6 / 3. The vm1 pass stays
`numetron_sub_n`/`numetron_add_n`. Measured (2026-09-23; correctness: 11148 products and
kernel cases vs GMP / a C++ reference, and gtest with the flag, both compilers — all pass):

- Kernel alone vs the three passes (n = 19..64): MSVC 25–35% faster (9.0 vs 13.6 ns at n = 20),
  GCC 0–18% (its three passes were already fast; 9.3 vs 9.9 ns at n = 20).
- One node (halves basecase), interleaved A/B: +0.2–0.5% on both compilers for n = 44..128. n = 38–40
  shows +5–10% on both, more than the kernel saves (1–4 ns); likely code placement, not the pass.
- `numetron_bench_mul`, 2 runs each, ABBA: MSVC −1.6..+3.5% (mean ~+0.5%), GCC −0.9..+4.8% (mean
  ~+1.9%) over 96–8192 limbs. The bench noise is ±3–4% (compare the sizes below the Karatsuba
  threshold, where both builds run identical code), so the gain is within noise.

Superseded by the all-assembly variant below; kept selectable for comparison.

**All-assembly variant** (`umul_karatsuba_asm.hpp`, `NUMETRON_KARATSUBA_IMPL_ASM`, x86-64, the
default with `NUMETRON_USE_ASM`; MIT `src/arch/x86_64/karatsuba_mul.s` for SysV,
`karatsuba_mul.asm` for Microsoft x64 — same bodies, Win64 calls out with home space and the fifth
argument on the stack, no red zone, unwind info): the whole recursion in asm. The C++
wrapper allocates the scratch for all levels once (a node takes 2n limbs, children get the rest;
`karatsuba_asm_scratch`) and passes it, the threshold and the CPU's `mul_basecase` pointer in a
context struct. Below the threshold the asm tail-jumps into `mul_basecase` (a linked call, not
inlined: the call is ~1% of a 20 x 20 basecase, and inlining GMP code would make the file LGPL).
Beyond the fused variant: every product writes exactly un+vn limbs (no memsets, no zero
stripping), |a0−a1| subtracts only up to the highest differing limb, and vm1 is folded into the
middle-column pass as two more carry chains (5 in all, al/ah/bl/bh/cl), so the interpolation is
one pass instead of four. Measured with GCC 13 (2026-09-24, scratch `karatsuba_asm_check.cpp`;
37884 products vs GMP with guard limbs around result and scratch, thresholds 4..16 and single
node, and gtest `mul`/`mpn_mul` with the flag — all pass), two runs, n x n, Toom-3 off:

- One node (halves basecase), vs the original: +5–8% at n = 24–40, +1.5–3% at 48–80,
  ±1–3% (noise) above.
- Whole recursion, default threshold 38: +2–8%, typically ~3% (n = 48..384); fused alone is
  within ±3% of the original there.
- The MSVC version runs the same checks (gtest `mul`/`mpn_mul`, the Toom-3 checks vs GMP); `--tune`
  moves the Karatsuba threshold 38 → 29 on both compilers, and every threshold above it with it
  (§ 1 table, § 6).

### Toom-3 (`umul_toom3.hpp` explicit; `toom_3x3.hpp` `toom3_balanced` engine plan)
Points 0, ±1, 2, ∞. The engine plan has 1 `eval_pm1` + 1 lincomb per operand, 5 products,
5 lincomb interpolation passes, and 4 compose ops. The two versions are now equally fast (engine
−0.6% on average, within ±3%). **Whether to switch the default to the engine is still an open
decision** — keep `NUMETRON_TOOM3_IMPL_CXX` the default until it is made.

**MSVC vs GCC, one node** (2026-09-24, scratch `toom3_cost.cpp`; hand-written Toom-3 at the top,
products on the asm Karatsuba, threshold 29, and asm basecase — identical on both compilers; two
runs each, stable). With `NUMETRON_KARATSUBA_IMPL_ASM`, `--tune` gave toom3 206 (MSVC) vs 109
(GCC); this is why:

| N | full MSVC / GCC (ns) | interp MSVC / GCC | eval | overhead over the 5 products |
|---|---|---|---|---|
| 90 | 1290 / 1244 | 150 / 125 | 71 / 65 | 17–20% / 15% |
| 150 | 2880 / 2810 | 253 / 205 | 110 / 110–124 | 14.5% / 12% |
| 300 | 8410 / 7870–8285 | 512 / 414–452 | 219 / 206 | 10% / 4.5–9% |
| 600 | 25040 / 23525–24817 | 1029 / 930 | 443 / 393–415 | 6.5% / 6% |

- The node costs MSVC only ~2–4% more. Interpolation is 15–25% slower (1–2.5% of the node), eval3
  5–15%; compose and the rest (split, scratch, dispatch) are equal.
- Per kernel (ns/limb, L = 61..201): add_n / sub_n (asm) and divexact_by3 are equal; **rshift1 is
  2.5–3x slower on MSVC** (0.27–0.37 vs 0.09–0.14 — GCC vectorizes it with -march=native; MSVC
  runs `__shiftright128` per limb) and **lshift1 1.6x** (0.22–0.28 vs 0.14–0.15).
- **The threshold gap is mostly the shape of the curve, not the code.** Toom-3 node / asm
  Karatsuba for N = 60..330 is 1.04 → 0.92 on MSVC and 1.00 → 0.93 on GCC, flat and
  non-monotonic (split parities: MSVC 180 → 0.997, GCC 150 → 1.000, 330 → 1.01). A 2–4% offset on
  such a curve moves the first sustained win by ~100 limbs. Running MSVC at 206 instead of ~110
  costs at most ~5% for N in between.
- **Tried: fusing the shifts into the neighbouring passes in C++ — slower on both compilers.**
  `wm = (w1 ± wm) >> 1`, `w2 = (w2 − w1) >> 1` as one pass each (C++ `rsh1add_n` / `rsh1sub_n`
  in `toom/kernels.hpp`, like GMP's) and `w2 −= 2·c4` via `sublsh_n(…, 1, …)` remove three passes,
  but the fused C++ loops (carry chain + shrd per limb) cost more than the asm `add_n`/`sub_n`
  (0.17 ns/limb) plus a separate shift. Interpolation, before → after (ns): GCC 125 → 157 (N=90),
  205 → 276 (150), 414–452 → 598–605 (300), 930 → 1150–1247 (600), i.e. +25–45%; MSVC 150 → 148,
  251 → 271, 512 → 573, 1029 → 1170, i.e. −1..+14%. Correct (10912 kernel and Toom-3-vs-GMP
  checks on both compilers, gtest mul/mpn_mul on GCC). Reverted. Only asm versions of these
  kernels could pay off (GMP has them in asm), and would gain ~1% of the node on MSVC, ~0 on GCC.
- Instead, the MSVC-specific loss is addressed in the shift kernels themselves: under MSVC with
  `/arch:AVX2`, `lshift1` / `rshift1` in `toom/kernels.hpp` run explicit AVX2 loops (4 limbs per
  step from two overlapping loads), i.e. what GCC's auto-vectorization does. This also reaches
  eval3's lshift1 and the engine's shift ops (Toom-4/6.5/8.5). Measured (MSVC, two runs):
  rshift1 0.27–0.37 → 0.12–0.15 ns/limb, lshift1 0.22–0.28 → 0.10–0.12 (GCC: 0.09–0.14 /
  0.14–0.15); interpolation 150 → 119–124 ns (N=90), 251 → 197 (150), 512 → 413–428 (300),
  1029 → 862–870 (600), −15..−22%, now level with GCC; eval3 −5%; the whole node −1.5..−2%.
  Correct: 8672 kernel (vs a scalar reference, in and out of place) and Toom-3-vs-GMP checks on
  both compilers, gtest mul/mpn_mul on MSVC.

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

**Toom-6.5 / 8.5 thresholds, MSVC vs GCC** (2026-09-24, scratch `toom68_cost.cpp`, two runs each;
one top node of Toom-4 / 6.5 / 8.5, children through karatsuba 29 (asm) / toom3 115 / toom4 444
on both). With the asm Karatsuba, `--tune` gave toom6h / toom8h = 2249 / 2249 on MSVC and 675 / 967
on GCC. The node curves don't show that difference:

| N | t6/t4 MSVC | t6/t4 GCC | t8/t6 MSVC | t8/t6 GCC |
|---|---|---|---|---|
| 500 | 1.01 | 1.02 | 1.07 | 1.05 |
| 700 | 0.93–0.96 | 0.94–0.99 | 1.00–1.01 | 1.01–1.02 |
| 1000 | 0.96 | 0.97–0.98 | 0.98 | 0.98 |
| 1400 | 0.93–0.95 | 0.96 | 1.00–1.02 | 0.99–1.00 |
| 1800 | 0.96 | 0.95 | 0.97 | 0.97 |
| 2500 | 0.92–0.93 | 0.92–0.93 | 0.98–0.99 | 0.98–0.99 |
| 4000 | 0.93 | 0.93–0.94 | 0.94–0.95 | 0.94–0.95 |

- Toom-6.5 beats Toom-4 from ~650–700 limbs on both compilers (by 3–8%; MSVC slightly more);
  Toom-8.5 is level with Toom-6.5 from ~700 to ~1600 and ahead (2–6%) from ~1800.
- The linear part (node minus its 7 / 11 / 15 pointwise products) is the same or a bit smaller on
  MSVC: ~5–7 / 10–11 / 15–17 ns per limb of N for Toom-4 / 6.5 / 8.5 (GCC 5–7 / 11–12 / 15–17).
  No codegen gap here (the AVX2 shift kernels included).
- So the MSVC 2249 is the tuner, not the code: Toom-6.5's gain is a few % and the validation
  phase takes the *largest* candidate whose full-product score is within `tie_tolerance` (0.3%)
  of the best; with gains this small, which candidate falls inside that band depends on the run.
  The defaults (717 / 967 on MSVC, 675 / 967 on GCC) follow the curves.

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

Thresholds are runtime atomics (`set_*_threshold()`, relaxed loads) with compile-time defaults
(`NUMETRON_DEFAULT_*` in `toom/thresholds.hpp`, overridable with `NUMETRON_*_THRESHOLD`), one
set per Karatsuba implementation (asm vs C++: a faster level moves every crossover above it)
and per compiler (the Toom kernels are compiled C++). The header-only build (no
`NUMETRON_USE_ASM`) takes the C++ set; it has never been tuned.

`tune_mul_thresholds()` (`mul_tuning.hpp`; `numetron_bench_mul --tune[=samples] [--trace]`)
tunes Karatsuba, Toom-3, Toom-4, Toom-6.5, Toom-8.5 and the FFT in that order, each against the
tuned ones below (the FFT from the Toom-4 threshold up to `fft_max` = 16384, against the whole
Toom chain; everything above the stage being tuned is off, the FFT included). **Why it works the way it does**:

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
  doesn't matter. The same rule gave toom6h = toom8h = 2249 in one MSVC run although the node
  curves match GCC's (§ 4, Toom-6.5): on a plateau, read the curves, not a single `--tune`.
- The whole `--tune` takes about 25 s, plus ~10–20 s for the FFT stage.

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
- **Passing a define to an MSBuild build**: the `CL` / `_CL_` environment variables did not
  reach the compiler (the A/B builds came out identical). What works without editing the project:
  `/p:ForceImportBeforeCppTargets=<file>.props` with an `ItemDefinitionGroup` that adds to
  `PreprocessorDefinitions`.
- **Fusing a shift into an add/sub pass in C++ can lose**: the asm `add_n`/`sub_n` plus a
  separate (vectorized) shift beat a fused C++ carry-chain loop (§ 4, Toom-3).

---

## 9. Open items

1. **FFT**: done — a multi-prime NTT (not Schönhage–Strassen), on by default from ~2.5–2.7k
   limbs with the AVX2 kernel; `docs/fft.md` (its § 6 lists what is left: the scalar Horner step of
   the CRT, AVX-512, finer lengths).
2. **Small operands (1–8 limbs, 0.6–0.9 vs GMP)**: overhead around the basecase in
   `mul()`/`umul()` (dispatch, allocation, normalization).
3. **Unbalanced products**: the generic `toom_engine<3,3>` plan still uses the old slow ops.
   Candidates: a Toom-3 2 x 3-style plan (v split into 2) for un/vn ≈ 1.5–2.5, a lincomb-based
   rewrite of the generic plan, and the 6 x 7 / 8 x 9 ("half") Toom-6.5/8.5 variants. Needs a
   benchmark on unequal sizes first.
4. **Decide Toom-3 explicit vs engine** default (now equal speed).
5. Retune thresholds after any kernel change, on both compilers, and update the defaults.
6. The § 1 table is current (2026-09-24, FFT included); refresh it after the next change.
7. **Header-only build** (no `NUMETRON_USE_ASM`, now the default): the C++ 64-bit basecase path
   (`umul_basecase_unrolled`, plus 1 x 1) has not been run yet — neither gtest nor timing — and
   its thresholds are untuned.
8. **Tuner on plateaus**: the tie rule picks the largest threshold within 0.3%, which makes
   Toom-6.5/8.5 (and Toom-3 over the asm Karatsuba) thresholds jump between runs; a median over
   several runs, or a smaller tie band there, would make `--tune` repeatable.
9. Toom-3 `eval3` still has one `lshift1` (p2 = 2·(p1 + x2) − x0) — minor.
