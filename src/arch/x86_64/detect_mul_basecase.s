# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2025 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.
    
    .text
	.align	16, 0x90

    .globl __k8_mul_basecase
    .type __k8_mul_basecase, @function

    .globl __core2_mul_basecase
    .type __core2_mul_basecase, @function

    .globl __alderlake_mul_basecase
    .type __alderlake_mul_basecase, @function

	.globl	detect_mul_basecase
	.type	detect_mul_basecase,@function


detect_mul_basecase:
    # %rdi = platform_descriptor (Microsoft x64 calling convention)

    cmp $0x10000, %rdi
    jl  .Lamd_platform
    cmp $0x20000, %rdi
    jl  .Lintel_platform
    cmp $0x30000, %rdi
    jl  .Latom_platform
    cmp $0x40000, %rdi
    jl .Lvia_platform
    xor %rax, %rax                        # Unknown platform
    ret
    
.Lamd_platform:
    lea __k8_mul_basecase(%rip), %rax
    ret

.Lintel_platform:
    cmp $0x01069E, %rdi # Core2
    jge .Lcore2_platform
    lea __k8_mul_basecase(%rip), %rax   # Pre-Core2 Intel
    ret

.Lcore2_platform:
    cmp $0x10697, %rdi # Alder Lake
    jge .Lalderlake_platform
    lea __core2_mul_basecase(%rip), %rax
    ret

.Lalderlake_platform:
    lea __alderlake_mul_basecase(%rip), %rax
    ret

.Latom_platform:
.Lvia_platform:
    xor %rax, %rax                        # VIA - Unknown
    ret

    .size detect_mul_basecase,.-detect_mul_basecase
	.section .note.GNU-stack,"",@progbits
