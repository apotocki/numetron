# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2025 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.
    
    .text
	.align	16, 0x90

    .globl __k8_mul_basecase
    .hidden __k8_mul_basecase
    .type __k8_mul_basecase, @function

    .globl __core2_mul_basecase
    .hidden __core2_mul_basecase
    .type __core2_mul_basecase, @function

    .globl __alderlake_mul_basecase
    .hidden __alderlake_mul_basecase
    .type __alderlake_mul_basecase, @function

	.globl	detect_mul_basecase
	.type	detect_mul_basecase,@function


detect_mul_basecase:
    # %rdi = platform_descriptor (SysV x64 calling convention)

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
    cmp $0x001530, %rdi # Excavator
    jge .Lamd_15ge
    lea __k8_mul_basecase(%rip), %rax
    ret
.Lamd_jaguar:
    lea __k8_mul_basecase(%rip), %rax
    ret
.Lamd_15ge:
    cmp $0x001700, %rdi # less than Zen
    jl .Lamd_jaguar
    lea __alderlake_mul_basecase(%rip), %rax  # All Zen
    ret

.Lintel_platform:
    # Check Intel family 06h
    mov %edi, %eax 
    and $0xFFFF00, %eax                 # vendor + family
    cmp $0x010600, %eax                 # Intel + family 06h
    jne .Lintel_non6                    # Non-06h (e.g., NetBurst) -> k8 kernel

    # family 06h:
    mov %edi, %eax
    and $0xFF, %eax
    cmp $0x97, %eax
    jge .Lalderlake_platform
    lea __core2_mul_basecase(%rip), %rax
    ret

.Lalderlake_platform:
    lea __alderlake_mul_basecase(%rip), %rax
    ret
.Lintel_non6:
    lea __k8_mul_basecase(%rip), %rax   # Non-06h Intel (e.g., family 0Fh)
    ret

.Latom_platform:
.Lvia_platform:
    xor %rax, %rax                      # VIA - Unknown
    ret

    .size detect_mul_basecase,.-detect_mul_basecase
	.section .note.GNU-stack,"",@progbits
