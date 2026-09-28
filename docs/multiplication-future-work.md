# Multiplication and squaring: future work

The multiplication and squaring work stopped on 2026-09-28 with numetron ahead of GMP from
4 limbs up in the default configuration (`NUMETRON_USE_ASM`, AVX2 FFT), on both compilers:
products 1.11–1.96 of GMP's speed, squares 1.19–1.59 (`multiplication.md` § 1). This is what
was left, roughly by expected value. References of the form "§ 9 item N" are to
`multiplication.md`, which holds the design, the measurements and what was tried.

## Performance

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
- **Header-only builds below ~1000 limbs.** MSVC products 0.70–0.88 of GMP, squares 0.44–0.83:
  no x64 inline asm, so no ADX rows and no squaring kernel. GCC with the ADX rows: 0.96–1.15.
  A C++ squaring kernel for MSVC (`_mulx_u64` rows with the doubling pass) is the obvious step;
  ADX itself is out of reach of MSVC's intrinsics (§ 9 item 7).
- **`assign_mul` overhead at 1–4 limbs**: ~2.6 ns against `mpz_mul`'s ~2.2 (§ 9 item 2, Left).
  The GCC 9–16 limb gap between `assign_mul` and the kernel alone (4–7 ns, with the looped rows)
  was not measured again after the straight-line rows.
- **`operator*` on MSVC**: ~22 ns of heap allocation per result (GCC ~6 ns). That is
  `basic_integer`'s allocator, not the multiplication.

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

- Toom-3 `eval3` still has one `lshift1` (§ 9 item 9).
- Explicit vs engine Toom-3 as the default (§ 9 item 4; equal speed now).

## Not started

- **ARM (aarch64)**: no assembly yet (`src/arch/aarch64` is a stub); the C++ path is used.
