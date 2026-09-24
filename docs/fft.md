# FFT multiplication: plan

Working plan for the next level of the multiplication stack (see `multiplication.md` for
everything below it). Status: **multi-prime NTT implemented and verified** — a portable scalar kernel
(`fft/ntt.hpp`, five 62-bit primes) and an AVX2 + FMA kernel (`fft/ntt_avx2.hpp`, six 49-bit
primes, the default when the compiler targets AVX2), 2 limbs per coefficient (`umul_fft.hpp`);
ahead of GMP from 2048 limbs up with AVX2; on by default (thresholds from the tuner, § 5 phase 5). Numbers marked *estimate* are
back-of-the-envelope; § 4–5 hold the measurements.

---

## 1. Why

From 12288 limbs up numetron falls behind GMP and keeps falling: `gmp/reuse` 0.86 at 12288,
0.76 at 16384 (`multiplication.md` § 1). GMP's time grows ~1.36x from 8192 to 12288 limbs (1.5x
the size, i.e. exponent ~0.76 there, the switch to its FFT), while Toom-8.5 grows like
N^log8(15) ≈ N^1.30. Every doubling beyond that costs numetron another ~1.2x against GMP.
No Toom variant closes this; an O(N log N)-class algorithm does.

Target: parity with GMP where it switches to FFT, and ahead above it, on both compilers.

---

## 2. The two candidates

### 2.1 Schönhage–Strassen (SSA) — what GMP does (`mpn_mul_fft`, Nussbaumer's variant)

Multiply modulo 2^N + 1: cut into 2^k pieces of M bits, work in the ring Z/(2^n' + 1) with
n' ≥ 2M + k, where 2 is a root of unity of order 2n'. Every twiddle is a power of 2.

- **Butterflies are shifts and adds of n'-bit numbers** — no multiplications; exactly the
  kernels we already have in asm (`add_n`/`sub_n`) or can write (shift-with-wraparound mod
  2^n' + 1).
- Pointwise products are n' × n' mod 2^n' + 1: recursively SSA for large n', else our
  Karatsuba/Toom plus a mod-(2^n'+1) reduction.
- O(N log N log log N). Memory ~2–3x the operands.
- Constant factor: GMP's is decades-tuned (the √2 trick, truncation via `mpn_mulmod_bnm1`,
  asm butterflies). Writing the same algorithm reaches *GMP level at best*; beating it needs
  better engineering of the same thing.
- **LGPL**: GMP's code can't be ported into our MIT headers — the algorithm is free, the code
  must be ours.

### 2.2 Multi-prime NTT (number-theoretic transform + CRT)

Split the operands into coefficients of b bits, transform modulo several word-size primes
p = c·2^k + 1 (roots of unity of order 2^k exist), multiply pointwise, transform back,
reconstruct each coefficient of the product by CRT, then carry-propagate.

- **Each butterfly does one modular multiplication** of 64-bit words (Shoup/Montgomery with
  precomputed twiddles: 2 `mul`/`mulx` + a few adds; *estimate* 2–4 cycles scalar).
- Fixed-size data (arrays of words), no recursion on big numbers, no carries until the end:
  simple memory layout, cache blocking (4-step / 6-step for sizes beyond L2), and it
  **vectorizes** — the modern fast implementations (FLINT `fft_small`, y-cruncher) are
  NTT/floating-point based and beat GMP's SSA by a clear factor on AVX2/AVX-512 hardware.
- Sizing with 62-bit primes and 64-bit coefficients (one limb each):
  the product coefficients are < L·2^128 (L = transform length), so
  - 3 primes (~2^186) allow L up to ~2^58 — 1 limb per coefficient, L ≈ un + vn;
  - 5 primes (~2^310) allow 2 limbs per coefficient — half the length for 5/3 the primes,
    ~17% less work per limb (*estimate*), more CRT work.
- O(N log N) for any practical size. Memory: primes × L words (can be streamed per prime).
- Header-only friendly: a portable C++ core with `umul1` (64x64→128) works on every compiler;
  SIMD and asm are optional layers on top.

### 2.3 Rough cost at 16384 x 16384 limbs (*estimate*)

3 primes, L = 32768 = 2^15, per prime 2 forward + 1 inverse transform:
3 × 3 × (L/2)·log2 L ≈ 2.2M butterflies, ~2–3 cycles each scalar → ~5–7M cycles, ~1.2–1.6 ms at
4.5 GHz, plus pointwise and CRT. That is in the range of GMP at this size (to be measured, § 4):
**a scalar NTT is roughly at parity; the win comes from SIMD** (AVX2: 4 lanes; FLINT-style
50-bit primes in doubles with FMA, or 32-bit primes in integer lanes), from truncated lengths
(no 2x jumps at powers of two) and from squaring (one forward transform fewer).

### 2.4 Recommendation

**Multi-prime NTT.** Simpler data flow than SSA, the only route that can clearly beat GMP (SIMD),
and a portable C++ core fits the header-only default. SSA would reproduce GMP's algorithm with,
at best, GMP's speed.

---

## 3. How it fits numetron

- **Place**: `include/numetron/limb_arithmetic/fft/` — primes and modular arithmetic, the
  transform, CRT/recomposition, `umul_fft.hpp` with `detail::umul_fft_impl(u, un, v, vn, rb,
  alloc)` under the same contract as the Toom cores (writes rb[0..un+vn), returns rb+un+vn).
- **Dispatch**: first check in `umul_dispatch()` / `umul()`, before Toom-8.5:
  `is_fft_applicable(un, vn)` = `vn >= fft_threshold()` (+ an unbalance limit to decide
  later — an FFT handles un ≠ vn natively, the length follows un + vn). A new runtime threshold
  (`set_fft_threshold()`, default from measurements) and a new stage in `tune_mul_thresholds()`
  (fft vs everything below).
- **Implementation choice**: `NUMETRON_FFT_IMPL` in `config/implementation.hpp` (scalar C++
  first; SIMD variants later), same pattern as Karatsuba/Toom-3.
- **Memory**: from the scratch allocator of `umul()` (`stack_allocator`, LIFO), sized up front:
  primes × L words for the transformed operands (the second operand can be transformed into the
  result slot of the first, one prime at a time), plus twiddle tables.
- **Twiddle tables**: per prime and log-length, computed on first use and cached (thread-safe
  static per (prime, k)), or computed on the fly per level — to be measured.
- **Squaring**: numetron has no squaring path yet; the FFT is where it pays most (2 transforms
  instead of 3 per prime). Detect u == v (same pointer and size) in the FFT entry.
- **Licensing**: all new code MIT; nothing taken from GMP.

---

## 4. First measurement: where GMP switches, what we have to beat

Scratch `fft_baseline.cpp` (not in the repository): n x n for n = 2048 .. 524288 limbs in √2
steps, GMP `mpn_mul` vs numetron `umul_dispatch` (default configuration: asm Karatsuba, current
thresholds), best of several runs. It prints both times, their ratio, and each one's local
exponent log(t2/t1) / log(n2/n1): Toom-8.5 stays near 1.30, GMP drops towards ~1.05–1.1 once its
FFT takes over — the drop shows GMP's `MUL_FFT_THRESHOLD` on this machine and build (the vcpkg
Windows GMP and the Ubuntu GMP are different builds, so both are measured).

This gives the targets for phase 1: the size range, and the time the NTT has to reach there.

**Measured** (2026-09-24, GMP 6.3.0 on both: the vcpkg build on Windows, Ubuntu's on Linux;
numetron default configuration with `NUMETRON_USE_ASM` — asm Karatsuba, thresholds 29 / 115 /
444 / 717 (GCC 675) / 967; every product checked against GMP):

| n | GMP MSVC-side (µs) | GMP Linux (µs) | numetron MSVC (µs) | numetron GCC (µs) | gmp/num MSVC | gmp/num GCC |
|---|---|---|---|---|---|---|
| 4096 | 398 | 391 | 336 | 333 | 1.19 | 1.17 |
| 8192 | 1010 | 1011 | 855 | 849 | 1.18 | 1.19 |
| 11585 | 1160 | 1018 | 1374 | 1390 | 0.84 | 0.73 |
| 16384 | 1949 | 1823 | 2140 | 2171 | 0.91 | 0.84 |
| 23170 | 2482 | 2173 | 3515 | 3454 | 0.71 | 0.63 |
| 32768 | 4770 | 4312 | 5706 | 5527 | 0.84 | 0.78 |
| 46341 | 5327 | 4833 | 8804 | 8596 | 0.61 | 0.56 |
| 65536 | 9662 | 9186 | 13822 | 14059 | 0.70 | 0.65 |
| 92682 | 11235 | 10944 | 22208 | 22873 | 0.51 | 0.48 |
| 131072 | 22681 | 23243 | 34421 | 34396 | 0.66 | 0.68 |
| 185364 | 28279 | 28416 | 55140 | 55358 | 0.51 | 0.51 |
| 262144 | 45960 | 45609 | 88832 | 88225 | 0.52 | 0.52 |
| 370728 | 61680 | 59698 | 137306 | 139388 | 0.45 | 0.43 |
| 524288 | 118984 | 115085 | 230293 | 221577 | 0.52 | 0.52 |

- **GMP switches to FFT between 8192 and 11585 limbs** on both builds: its time barely moves
  there (1010 → 1018–1160 µs). Up to 8192 numetron is 17–19% ahead of GMP (asm Karatsuba and
  the current thresholds included), and loses from the first FFT size on.
- **GMP's FFT is a sawtooth**: the sizes 2^k·√2 are cheap, the powers of two expensive (local
  exponent alternating ~0.3–0.8 / ~1.4–2.0; SSA's piece sizes suit the former). Its envelope over
  8192 → 524288 grows like N^1.15; numetron's Toom-8.5 like N^1.35 (1.30–1.49 locally), the same
  on both compilers.
- Targets (the cheap GMP points): ~1.0–1.2 ms at 11585, ~2.2–2.5 ms at 23170, ~4.8–5.3 ms at
  46341, ~28 ms at 185364 — about 90 → 150 ns per limb of n. At the powers of two GMP is up to
  1.9x slower than at the neighbouring √2 point, so an NTT with fine-grained lengths wins there
  first.
- For the scalar NTT (§ 2.3): at n = 11585 a power-of-two length (L = 32768) costs as much as at
  16384, ~1.2–1.6 ms (*estimate*) — above GMP's 1.0–1.2 ms. With a length of 3·2^13 = 24576 it is
  ~0.9–1.2 ms. **Mixed-radix / truncated lengths belong in the first working version**, not
  later.

---

## 5. Phase 2 result: the scalar NTT

`limb_arithmetic/fft/ntt.hpp` + `limb_arithmetic/umul_fft.hpp` (MIT, header-only, no asm):

- Primes: the three largest p = c·3·2^40 + 1 below 2^62 (0x3fffc00000000001, 0x3fff840000000001,
  0x3fff810000000001; primitive roots 11, 19, 5), found and checked by a scratch search
  (Miller–Rabin, all prime factors of p − 1). One limb per coefficient.
- Lengths 2^k and 3·2^k (the smallest ≥ un + vn − 1). Forward: one radix-3 DIF step (for 3·2^k)
  then radix-2 DIF; inverse: the mirror image (radix-2 DIT, then radix-3), so no bit-reversal
  permutation. Values lazily in [0, 2p).
- Shoup multiplication by roots from per-level tables (`root_tables`: one table per prime and
  level, shared by all lengths, built on first use, lock-free reads). Montgomery form for the
  operands and the pointwise products; the final scaling takes out L and R at once.
- Squaring (same operand) skips the second forward transform.
- Garner CRT with Shoup constants, summed into the result with a two-limb running carry.
- Dispatch: first check in `umul_dispatch()` / `umul()`, `vn >= fft_threshold()`; the threshold
  (`set_fft_threshold()`, `NUMETRON_FFT_THRESHOLD`) is **off by default** for now.

Verified (2026-09-24): 3877 products vs GMP — every shape up to 130 limbs, n = un + vn − 1 at
2^k, 3·2^k and ±1 for k ≤ 17, sizes up to 200000, squares, un ≫ vn, four data patterns, guard
limbs — with asserts on, both compilers, 0 failures (GCC `-Wall -Wextra` clean); the full gtest
suite (36 tests) passes with `NUMETRON_FFT_THRESHOLD=64` on GCC.

First timing, n x n (µs; toom = the current default chain):

| n | GMP | toom MSVC | FFT MSVC | FFT GCC | FFT/GMP MSVC | FFT/toom MSVC | FFT/toom GCC |
|---|---|---|---|---|---|---|---|
| 8192 | 1003–1023 | 846 | 913 | 1031 | 0.91 | 1.08 | 1.18 |
| 11585 | 996–1027 | 1365 | 1636 | 1790 | 1.64 | 1.20 | 1.21 |
| 16384 | 1774–1826 | 2117 | 1924 | 2208 | 1.08 | 0.91 | 1.00 |
| 23170 | 2094–2415 | 3448 | 3427 | 3804 | 1.42 | 0.99 | 1.06 |
| 32768 | 4152–4531 | 5429 | 4182 | 4744 | 0.92 | 0.77 | 0.82 |
| 65536 | 8872–9591 | 13580 | 8749 | 9859 | 0.91 | 0.64 | 0.69 |
| 131072 | 23139–23811 | 34663 | 19462 | 21089 | 0.82 | 0.56 | 0.60 |
| 262144 | 47204–47260 | 87172 | 40750 | 44326 | 0.86 | 0.47 | 0.49 |
| 524288 | 113225–113637 | 223007 | 86896 | 94486 | 0.77 | 0.39 | 0.42 |

- The FFT overtakes the Toom chain at ~16–24k limbs (MSVC) / ~24–32k (GCC) and is 2.4x faster
  at 524288.
- Against GMP it already wins at the powers of two (0.77–0.92 on MSVC from 8192, GCC from 131072),
  where GMP's SSA is at its worst, and loses at the √2 points by 1.1–1.8x — GMP's cheap sizes.
- MSVC's code is ~10% faster than GCC's here.
- The 3·2^k lengths cost more than their size: 11585 (L = 24576) takes 0.85 of 16384 (L = 32768),
  not 0.75 — the radix-3 step (Montgomery twiddles computed on the fly) is heavy.
- ~4 cycles per butterfly overall (at 16384: 2.2M butterflies, ~9M cycles incl. loading,
  pointwise and CRT): the phase-3 work (radix-4, cheaper radix-3, fewer passes over memory) aims
  there.

### Phase 3, first round (2026-09-24, scratch `fft_prof.cpp`, `bf_micro.cpp`)

Where the time goes (both compilers alike, n = 8192 .. 262144): the transforms ~83–88% (forward
~50%, inverse ~30–37%), CRT 6–8%, loading / pointwise / scaling 2–5% each. The butterflies run at
0.72–0.83 ns (MSVC) / 0.82–0.94 (GCC) — the same from L = 16K to 512K, so the transforms are
compute-bound, not memory-bound.

- **Radix-3 step from cached Shoup tables** (`root_tables::radix3`: w^j, w^2j and inverses per
  prime and k) instead of Montgomery twiddles chained on the fly: −5..−7% on the 3·2^k lengths
  (n = 11585: GCC 1754 → 1629 µs, MSVC 1615 → 1530). Kept.
- **The last two DIF / first two DIT levels without roots** (only i = w4 is not 1): ~7% fewer
  multiplications, within noise in time. Kept (it is simpler for those levels anyway).
- **Two levels fused into one pass** (4 elements per step, half the loads/stores): *slower* —
  GCC 0.82 → 0.93 ns per butterfly, MSVC 0.73 → 0.80, and 1.04 at L ≥ 256K on MSVC (four
  power-of-two strided streams). Dropped.
- **Montgomery-friendly primes** p = c·2^k + 1 with small c (e.g. 27·2^56 + 1, 57·2^55 + 1,
  69·2^55 + 1: q·p becomes shifts, 2 real multiplications per butterfly instead of 3): the
  butterfly micro-benchmark (one level in L1) gives GCC 0.76 → 0.66–0.70, MSVC 0.62 → 0.63 — not
  worth a change of primes. The butterfly (~20 instructions, ~3 cycles) is near what scalar code
  gets; the multiplier port is not the only limit.

**Conclusion of the first round**: the scalar transform is close to its ceiling. What is left for
scalar code is fewer butterflies — the 5-prime / 2-limb-per-coefficient variant (5 transforms of
half the length instead of 3: ~17% fewer butterflies, heavier CRT; *estimate* −12..−15% overall).
Beating GMP at its cheap sizes (n = 2^k·√2, still 1.4–1.6x faster there) needs the SIMD phase.

### Phase 3, second round: 2 limbs per coefficient, 5 primes (2026-09-24)

Two more primes of the same form (0x3fff540000000001, g = 5; 0x3fff450000000001, g = 10); the
product of the five is ~2^310 > L·2^256 for L < 2^54. A coefficient is two limbs, loaded as
redc(lo·r2) + redc(hi·r3) (r3 = 2^192 mod p); Garner over five residues (10 Shoup products per
coefficient), a mixed-radix Horner step to 5 limbs and a 3-limb running carry, two limbs emitted
per coefficient. Verified like phase 2 plus the 2-limb length boundaries with odd / even
un, vn (4269 products vs GMP, both compilers, 0 failures; gtest 36/36 with the FFT threshold at
64, GCC).

It is faster than the 1-limb / 3-prime version at every size: MSVC −9..−14%, GCC −12..−18%
(fft2/fft1 0.86–0.94 / 0.82–0.88 over n = 2048 .. 524288), so it replaced it:

| n | GMP | toom (MSVC) | FFT MSVC | FFT GCC | FFT/GMP MSVC | FFT/GMP GCC |
|---|---|---|---|---|---|---|
| 4096 | 384–389 | 335 | 396 | 390 | 1.03 | 1.00 |
| 8192 | 1008–1013 | 847 | 845 | 843 | 0.84 | 0.83 |
| 11585 | 964–1007 | 1368 | 1388 | 1385 | 1.38 | 1.44 |
| 16384 | 1738–1763 | 2144 | 1799 | 1792 | 1.02 | 1.03 |
| 23170 | 2058–2413 | 3472 | 2947 | 2904 | 1.22 | 1.41 |
| 32768 | 4135–4579 | 5650 | 3984 | 3799 | 0.87 | 0.92 |
| 65536 | 8791–9799 | 13601 | 8031 | 8251 | 0.82 | 0.94 |
| 131072 | 22479–22760 | 34520 | 17533 | 17401 | 0.77 | 0.77 |
| 185364 | 28690–29170 | 54663 | 27906 | 27570 | 0.97 | 0.95 |
| 262144 | 45399–47667 | 88146 | 37839 | 36702 | 0.79 | 0.81 |
| 524288 | 112891–114369 | 222876 | 79299 | 76734 | 0.69 | 0.68 |

- The FFT now matches the Toom chain at 8192 and beats it from there on (2.8x at 524288); the two
  compilers are level.
- Against GMP: ahead at the powers of two from 8192 (0.68–0.94), level or ahead at the √2 points
  from ~185k, still behind at the √2 points 11585–92682 (1.2–1.44).
- The default threshold stays off until the tuner knows the FFT (phase 5).

### Phase 4: AVX2 + FMA kernel (2026-09-24)

`limb_arithmetic/fft/ntt_avx2.hpp`, selected by `NUMETRON_FFT_IMPL` (`config/implementation.hpp`):
AVX2 by default when the compiler targets AVX2 + FMA (MSVC `/arch:AVX2`, GCC/Clang `-mavx2 -mfma`
or a `-march` with them), the scalar kernel otherwise. Still header-only, no runtime CPU check.

- **Arithmetic in doubles**, four lanes (the FLINT `fft_small` approach): six primes
  p = c·3·2^32 + 1 < 2^49 (0x0001fffe00000001 … 0x0001ff5600000001; product ~2^294, still 2 limbs
  per coefficient). Values are signed integers in doubles, |x| < 2p between operations.
  a·b mod p = fma(−q, p, h) + l with h = a·b, l = fma(a, b, −h), q = round(h/p): exact, and
  |result| ≤ 1.25p as long as |a·b/p| ≤ 2^51 (inputs < 4p and < p, or both < 2p — why p < 2^49).
  Sums are reduced (x − round(x/p)·p) after every butterfly.
- Radix-2 levels of span ≥ 8 four butterflies per vector; the last two DIF / first two DIT
  levels on four blocks of four at a time, transposed into four vectors; transforms under 16
  points in scalar doubles; the radix-3 step vectorized over j.
- **Loading** vectorized too: each limb split into 32-bit halves (exact as doubles via the 2^52
  trick), a coefficient = lo₀ + lo₁·2^32 + hi₀·2^64 + hi₁·2^96 with the powers mod p. Plain
  residues (no Montgomery form), so the final scaling is L^-1 only.
- **CRT**: the Garner digits in doubles, four coefficients per step; the mixed-radix Horner
  recomposition stays scalar (shared `crt_accumulator` with the scalar kernel).

The first version (scalar integer loading and CRT, as in the scalar kernel) was only 10–20%
faster than the scalar kernel on MSVC (27–36% on GCC): the transforms ran at 0.29–0.33 ns per
butterfly (scalar: 0.73–0.94) on both compilers, but loading (10–29%) and CRT (23–32%) became
the bulk. With both vectorized:

| n | GMP | toom | scalar FFT MSVC / GCC | AVX2 FFT MSVC / GCC | AVX2/GMP MSVC / GCC |
|---|---|---|---|---|---|
| 2048 | 149–150 | 127–128 | 188 / 169 | 122 / 115 | 0.82 / 0.77 |
| 4096 | 387–390 | 336–338 | 406 / 365 | 257 / 244 | 0.66 / 0.63 |
| 8192 | 1005–1027 | 856–867 | 857 / 826 | 543 / 535 | 0.54 / 0.52 |
| 11585 | 1007–1012 | 1385–1426 | 1388 / 1371 | 844 / 811 | 0.83 / 0.81 |
| 23170 | 2105–2123 | 3429–3527 | 2926 / 2964 | 1833 / 1747 | 0.86 / 0.83 |
| 46341 | 4632–5028 | 8536–8738 | 6258 / 6228 | 3807 / 3663 | 0.76 / 0.79 |
| 131072 | 21502–22735 | 34228–34796 | 17078 / 17050 | 10736 / 10458 | 0.47 / 0.49 |
| 262144 | 46431–46764 | 88158–90187 | 36384 / 35806 | 23109 / 23260 | 0.50 / 0.50 |
| 524288 | 113809–114593 | 222596–223280 | 78077 / 77425 | 51827 / 50770 | 0.45 / 0.45 |

- **Faster than GMP at every size from 2048 limbs up**, both compilers: 0.45–0.63 at the powers
  of two, 0.76–0.86 at GMP's cheap √2 sizes. 34–41% faster than the scalar kernel.
- It is ahead of the Toom chain from 2048 limbs already (122 vs 127 µs MSVC, 115 vs 128 GCC), so
  the FFT threshold will land below that (phase 5).
- Where the time goes now (both compilers): transforms ~55%, CRT 21–34% (almost all of it the
  scalar Horner step: 15 64-bit multiplications per coefficient), loading 7–9%, the rest < 5%.
- Verified: 8534 products vs GMP (both kernels, all shapes of phase 2/3), asserts on, both
  compilers, 0 failures; full gtest (36) with `NUMETRON_FFT_THRESHOLD=64` on both compilers.

### Phase 5: the threshold (2026-09-24)

`tune_mul_thresholds()` has an FFT stage after Toom-8.5 (searched from the Toom-4 threshold up to
`mul_tuning_options::fft_max` = 16384 against the whole Toom chain; the FFT is off while the
stages below it are tuned). `numetron_bench_mul --tune` prints it. Two runs each gave the same
FFT threshold, now the defaults in `toom/thresholds.hpp` (per kernel and compiler):

| kernel | MSVC | GCC |
|---|---|---|
| AVX2 (the default with AVX2) | 2696 | 2538 |
| scalar | 11530 | 13828 |

With those defaults `numetron_bench_mul` (gmp/reuse, > 1: numetron faster), both compilers:

| limbs | 2048 | 3072 | 4096 | 6144 | 8192 | 12288 | 16384 |
|---|---|---|---|---|---|---|---|
| AVX2, MSVC | 1.16 | 1.33 | 1.51 | 1.59 | 1.79 | 1.58 | 1.54 |
| AVX2, GCC | 1.15 | 1.39 | 1.59 | 1.69 | 1.98 | 1.57 | 1.61 |
| scalar, MSVC (`--tune` run) | 1.07 | 1.14 | 1.12 | 1.03 | 1.13 | 0.95 | 1.01 |
| scalar, GCC (`--tune` run) | 1.17 | 1.15 | 1.15 | 1.13 | 1.14 | 0.86 | 0.99 |

(Before the FFT: 0.75–0.86 at 12288–16384.) The full gtest suite passes with the defaults on both
compilers.

---

## 6. Phases

1. **Baseline** (§ 4): GMP's crossover and our gap per size, both compilers. Done.
2. **Correct scalar NTT**: 3 primes × 62 bits, 1 limb per coefficient, lengths 2^k and 3·2^k
   (§ 4: the powers of two alone would leave the first FFT sizes behind GMP), radix-2/4 Shoup
   butterflies, CRT + carry recomposition. Verified against GMP (random,
   all-ones, sparse; un ≠ vn; sizes around every power of two) before any timing. Hooked into
   the dispatch behind the threshold. **Done** (radix-2 only so far, squaring included), § 5.
3. **Scalar performance** (first round done, § 5): radix-4 butterflies, a cheaper radix-3 step, finer lengths (5·2^k,
   truncation) to smooth the remaining steps, cache blocking for L beyond L2, fused pointwise,
   5-prime/2-limb variant (done: second round, § 5) — each A/B'd
   on node benchmarks like the Toom work. Goal: parity with GMP at its crossover.
4. **SIMD**: AVX2 (and AVX-512 where available) kernels under a compile-time switch; the
   portable scalar core stays as the header-only default. Goal: clearly ahead of GMP. **Done
   for AVX2** (§ 5, phase 4); left: the scalar Horner step of the CRT (21–34%), AVX-512.
5. **Tuning and docs**: `fft_threshold` in the tuner, defaults per compiler, results into
   `multiplication.md`. **Done** (§ 5, phase 5).
