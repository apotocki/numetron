# Architecture-Specific Assembly Files

This directory contains architecture-specific assembly implementations for optimized arithmetic operations.

## File Sources and Licensing

### GMP-derived Files
Some assembly files in subdirectories are taken from the GMP (GNU Multiple Precision Arithmetic Library) project:
- Original GAS (GNU Assembler) format files from GMP
- MASM (Microsoft Macro Assembler) format files converted from original GMP GAS files

**License**: All GMP-derived assembly files are licensed under **LGPL** (GNU Lesser General Public License).

### Project Files
All other source files in this directory are part of the Numetron project, e.g.
`x86_64/detect_platform.*`, `x86_64/detect_mul_basecase.*`, `x86_64/add_sub_n.*`,
`x86_64/karatsuba_interp.*`, `x86_64/karatsuba_mul.*` and `x86_64/mul_basecase_adx.*`
(written for Numetron, not derived from GMP).

**License**: **MIT License**

### Which ones a build uses
By default (`NUMETRON_ASM_LICENSE_MIT`) only the MIT files are used: the CMake `numetron` library
is built from them alone, and the mul_basecase is `mul_basecase_adx` on CPUs with BMI2 + ADX (the
C++ basecase on older ones). The GMP-derived (LGPL) mul_basecase routines are opt-in: the CMake
option `NUMETRON_GMP_LGPL=ON` builds them (with `detect_platform.*` / `detect_mul_basecase.*`)
into the library and exports `NUMETRON_USE_GMP_LGPL`; without CMake, define
`NUMETRON_USE_GMP_LGPL` and link them. The MSVC project `msvc/numetron.vcxproj` assembles all
files into its static library, but only referenced objects are linked from it, so an MIT build
does not pull the LGPL ones in.

## Compilation Dependencies

The compilation and linking dependencies related to these assembly files are **optional** and controlled by the `NUMETRON_USE_ASM` compilation flag.

### Usage
- When `NUMETRON_USE_ASM` is **enabled**: Assembly optimizations are included in the build
- When `NUMETRON_USE_ASM` is **disabled**: Pure C++ implementations are used instead

This allows the project to be built with or without assembly optimizations depending on the target platform and requirements.

## Directory Structure

Each routine comes as a pair: `.s` (GAS, for GCC / Clang) and `.asm` (MASM, for MSVC).

```
src/arch/
|-- x86_64/
|   |-- add_sub_n.*              # MIT: r = u +/- v
|   |-- karatsuba_interp.*       # MIT: fused Karatsuba interpolation pass
|   |-- karatsuba_mul.*          # MIT: the whole Karatsuba recursion
|   |-- mul_basecase_adx.*       # MIT: schoolbook mul_basecase, BMI2 + ADX (the default)
|   |-- detect_platform.*        # MIT: CPU family from CPUID     (NUMETRON_GMP_LGPL only)
|   |-- detect_mul_basecase.*    # MIT: picks one of the below    (NUMETRON_GMP_LGPL only)
|   |-- alderlake/mul_basecase.* # LGPL, GMP-derived              (NUMETRON_GMP_LGPL only)
|   |-- core2/mul_basecase.*     # LGPL, GMP-derived              (NUMETRON_GMP_LGPL only)
|   `-- k8/mul_basecase.*        # LGPL, GMP-derived              (NUMETRON_GMP_LGPL only)
|-- aarch64/
|   `-- detect_mul_basecase.s    # stub, no ARM assembly yet
`-- README.md                    # this file
```

## Notes

- What the assembly gains over the pure C++ (header-only) build, whole multiplications
  (`numetron_bench_mul`, the default MIT assembly vs the default C++ basecase for the compiler;
  x86-64 with BMI2 + ADX + AVX2, 2026-09; time of the C++ build relative to the asm one):

  | limbs | GCC (C++ basecase with ADX inline asm, `-march=native`) | MSVC (C++ basecase with intrinsics) |
  |---|---|---|
  | 1–8 | about the same (+0–5%) | +5–30% |
  | 16–1024 | +5–25% | +30–60% |
  | from ~1500 (FFT) | about the same (±5%) | about the same (±5%) |

  The FFT range is C++ in both builds. A portable GCC build (no `-march`: the reference C++
  basecase, the scalar FFT) is ~1.7–1.85x slower than the asm one from 16 limbs up, but most of
  that is the target, not the assembly.
- Fallback C++ implementations ensure portability across all platforms
- The optional nature of assembly dependencies maintains build flexibility