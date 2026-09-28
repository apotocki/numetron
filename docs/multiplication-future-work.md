# Multiplication and squaring: future work

The multiplication and squaring work stopped on 2026-09-28 with numetron ahead of GMP from
4 limbs up in the default configuration (`NUMETRON_USE_ASM`, AVX2 FFT), on both compilers:
products 1.11–1.96 of GMP's speed, squares 1.19–1.59 (`multiplication.md` § 1). This is what
was left, roughly by expected value. References of the form "§ 9 item N" are to
`multiplication.md`, which holds the design, the measurements and what was tried.

## Performance

- **A floating-point FFT for the middle range (~500–2000 limbs).** The largest expected gain
  and the most expensive item. YMP (Alexander Yee's library behind y-cruncher) is reported
  2–3x ahead of GMP from ~10⁴ to ~10⁶ bits. Our figures there: 1.4 at ~160 limbs, 1.13–1.17 at
  ~1500 limbs (Toom-6.5 / 8.5), 1.5–1.6 at ~16k limbs (the NTT). Most of that gap is likely
  the middle, where we run Toom without SIMD. The likely source (not verified; YMP is
  closed): an FFT over doubles, a few bits per point with a rigorous rounding-error bound.
  Its butterflies are plain FMAs on 4–8 lanes with no modular reduction, so it overtakes Toom
  far earlier than our exact NTT (six 49-bit primes, 2 limbs per coefficient, a CRT step),
  which pays off only from 2048 limbs. Work: the transform, a provable error bound for the
  chosen bits per point, AVX2 kernels, dispatch and thresholds against Toom-6.5 / 8.5.
- **Toom node overhead at 768–3072 limbs.** Products 1.11–1.31 of GMP, squares 1.19–1.31: the
  smallest margins above 16 limbs. The node's own linear work (evaluation, interpolation) is
  14–40% of a Toom-4 … 8.5 square node (§ 9 item 10, Left). On MSVC one line of the division
  steps (`_subborrow_u64` in `sbb1`) is 9% of a Toom-8.5 square node (VTune). Candidates: asm
  or AVX2 `lincomb` kernels for the interpolation, fewer passes over the scratch.
- **The scalar FFT kernel** (every build without AVX2). With the asm: products at 16384 limbs
  0.97–0.99 of GMP, squares at 8192–16384 limbs 0.86–0.92 (§ 9 item 11). Header-only MSVC:
  products 0.79–0.88 at 2048–4096 limbs. What `fft.md` § 6 lists: the scalar Horner step of the
  CRT, finer lengths (5·2^k, truncation) to flatten the staircase, AVX-512.
- **The FFT's staircase.** The fill rules (§ 2 item 0, § 9 item 10) pick the cheaper side of
  each length step but can't remove the step. 3072-limb squares stay at 1.19–1.22: one lower
  bound can't take both 3072 (full length) and 3073–3519 (4096 filled 0.75–0.86). Finer
  lengths (above) would; so would a Toom-3 node over FFT children at those sizes.

  FLINT 3's `fft_small` (Daniel Schultz) is the closest known design to ours: a small-prime
  NTT with ~50-bit primes in double arithmetic with SIMD, plus a CRT. From memory, not checked
  against its source, it differs in:
  - **truncated transforms** (van der Hoeven's TFT): the cost follows the coefficient count,
    not the next 2^k / 3·2^k. That removes the staircase itself, so the fill rules and the
    fill-dependent thresholds could go, and it is worth up to a third at the worst-filled sizes;
  - **adaptive packing**: the number of primes and the bits per coefficient are chosen per size,
    against our fixed six primes at 2 limbs per coefficient;
  - **cache-blocked large transforms** (Bailey-style passes) for lengths beyond L2.

  Its multithreading is a separate matter. The truncated FFT is the part to take first.
- **Header-only builds below ~1000 limbs.** MSVC products 0.70–0.88 of GMP, squares 0.44–0.83:
  no x64 inline asm, so no ADX rows and no squaring kernel. GCC with the ADX rows: 0.96–1.15.
  A C++ squaring kernel for MSVC (`_mulx_u64` rows with the doubling pass) is the obvious step;
  ADX itself is out of reach of MSVC's intrinsics (§ 9 item 7).
- **`assign_mul` overhead at 1–4 limbs**: ~2.6 ns against `mpz_mul`'s ~2.2 (§ 9 item 2, Left).
  The GCC 9–16 limb gap between `assign_mul` and the kernel alone (4–7 ns, with the looped rows)
  was not measured again after the straight-line rows.
- **`operator*` on MSVC**: ~22 ns of heap allocation per result (GCC ~6 ns). That is
  `basic_integer`'s allocator, not the multiplication.

## API

- **Fused `addmul` / `submul`** (`r += a·b`, `r -= a·b`; GMP's `mpz_addmul` / `mpz_submul`,
  `mpn_addmul_1`). Neither `basic_integer` nor `limb_arithmetic` has them. A sum of products
  (dot products, polynomial and matrix arithmetic) now builds a temporary for every product
  and adds it in a second pass. A fused operation:
  - needs no temporary;
  - for 1–2 limbs, is one short chain of multiply and add;
  - on larger sizes, adds the product into `r` as it is formed, with no extra pass.

  This is where mp++ (`mppp::integer<SSize>`) gets its 3–7x over `mpz_class` in its dot-product
  and accumulation benchmarks. The rest of that gain is values in place and no allocation,
  which `basic_integer<LimbT, N>` already has. Its multiplication itself is GMP's
  (`mpn_mul` above 2 limbs), so there is nothing to compare there.
- **A small-value benchmark against mp++**, together with `addmul`: dot products of vectors of
  1–2-limb values, numetron against `mppp::integer<1>` / `integer<2>`. It shows how far our
  path around 1–2 limbs is from the best known one. mp++ availability in vcpkg not checked; it
  may need a manual build.

## Tuning

- **Tuner repeatability** (§ 9 item 8). Toom-4 and Toom-6.5 / 8.5 still jump between plateau
  values from run to run (GCC Toom-4 394 / 564, MSVC Toom-6.5 / 8.5 761 → 2117–2389), with no
  measurable difference in the products (§ 9 item 11). A median over runs, or a smaller tie
  band, would make `--tune` repeatable; until then the defaults are chosen by hand.
- **`NUMETRON_USE_GMP_LGPL`**: the unbalanced thresholds were not tuned under the GMP-derived
  basecase (they keep the MIT values). The balanced ones were tuned on one CPU only (Alder Lake
  class, i.e. the alderlake variant); core2 and k8 were not measured.

  On that CPU the GMP-derived basecase is slower than numetron's own straight-line kernel
  (§ 9 item 11). The mode keeps it anyway: `NUMETRON_USE_GMP_LGPL` means the GMP-derived kernel
  runs wherever the CPU has one (decided 2026-09-28).

## Minor

- **Searching the interpolation sequences for our cost model** (low priority). The evaluation
  points are GMP's everywhere, and there is little room to pick better ones. Integer points
  bring odd divisors into the Vandermonde determinant: 3 from five points on, 63 = 4³ − 1 for
  sets of powers of 4. Points other than powers of two and their inverses cost real
  multiplications and coefficient growth. The sequences GMP's
  choice rests on (Bodrato & Zanoni, ISSAC 2007) are optimal in a model that counts operations:
  proven for Toom-3, found by search for Toom-4. Toom-6.5 / 8.5 have no such result. Our cost is
  passes over memory and carry chains rather than operations: fused shift-adds in C++ lost to
  separate asm `add_n` + shift (§ 4, Toom-3), and Toom-8.5 went from 6 to 4 Hensel
  divisions per half by reordering alone, the matrix unchanged. The plans are data, so a
  search could run over orderings, `lincomb_dual` fusion and where the divisions fall. The
  interpolation is 14–40% of a square node and less of a product node, so even 20% less linear
  work is ~3–8% of a node. That is below the asm / AVX2 `lincomb` kernels of the first item
  above.
- Toom-3 `eval3` still has one `lshift1` (§ 9 item 9).
- Explicit vs engine Toom-3 as the default (§ 9 item 4; equal speed now).

## Not started

- **ARM (aarch64)**: no assembly yet (`src/arch/aarch64` is a stub); the C++ path is used.
