# Architecture-Specific Assembly Files

This directory contains architecture-specific assembly implementations for optimized arithmetic operations.

## File Sources and Licensing

### GMP-derived Files
Some assembly files in subdirectories are taken from the GMP (GNU Multiple Precision Arithmetic Library) project:
- Original GAS (GNU Assembler) format files from GMP
- MASM (Microsoft Macro Assembler) format files converted from original GMP GAS files

**License**: All GMP-derived assembly files are licensed under **LGPL** (GNU Lesser General Public License).

### Project Files
All other source files in this directory are part of the Numetron project.

**License**: **MIT License**

## Compilation Dependencies

The compilation and linking dependencies related to these assembly files are **optional** and controlled by the `NUMETRON_USE_ASM` compilation flag.

### Usage
- When `NUMETRON_USE_ASM` is **enabled**: Assembly optimizations are included in the build
- When `NUMETRON_USE_ASM` is **disabled**: Pure C++ implementations are used instead

This allows the project to be built with or without assembly optimizations depending on the target platform and requirements.

## Directory Structure

```
src/arch/
??? x86_64/          # x86-64 architecture specific files
?   ??? *.s          # GAS format assembly files (from GMP)
?   ??? *.asm        # MASM format assembly files (converted from GMP)
?   ??? ...
??? README.md        # This file
```

## Notes

- Assembly files provide significant performance improvements for arithmetic operations
- Fallback C++ implementations ensure portability across all platforms
- The optional nature of assembly dependencies maintains build flexibility