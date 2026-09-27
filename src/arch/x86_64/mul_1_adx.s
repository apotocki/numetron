# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

# mul_1_adx.s -- one row of a schoolbook product with mulx + adcx/adox (BMI2 + ADX).
#
#   uint64_t numetron_mul_1_adx(uint64_t* rp, const uint64_t* up, size_t n, uint64_t v);
#       rp[0..n) = low n limbs of up[0..n) * v, returns the high limb
#   uint64_t numetron_addmul_1_adx(uint64_t* rp, const uint64_t* up, size_t n, uint64_t v);
#       rp[0..n) += up[0..n) * v, returns the limb carried out
#
# The rows of the squaring basecase (usqr_basecase), whose lengths vary. SysV x64 calling
# convention: rdi = rp, rsi = up, rdx = n, rcx = v; only caller-saved registers are used.
#
# A limb's product lo:hi goes in two independent carry chains: adcx (CF) adds the previous high
# half, adox (OF) adds rp[i] (addmul only). Between them the loops use only mov, lea and jrcxz,
# which leave both flags alone (dec would write OF). The n % 4 single limbs first, then n / 4
# blocks of four, the high halves alternating between r10 and rax.

    .text

    .align  16, 0x90
    .globl  numetron_mul_1_adx
    .hidden numetron_mul_1_adx
    .type   numetron_mul_1_adx, @function
numetron_mul_1_adx:
    mov     %rdx, %r8
    mov     %rcx, %rdx                  # v: mulx's implicit operand
    mov     %r8, %rcx
    shr     $2, %r8                     # r8 = n / 4
    and     $3, %ecx                    # rcx = n % 4
    xor     %eax, %eax                  # high half 0; CF = OF = 0
    jrcxz   .Lm1_blocks
.Lm1_single:
    mulx    (%rsi), %r9, %r10
    adcx    %rax, %r9
    mov     %r9, (%rdi)
    mov     %r10, %rax
    lea     8(%rsi), %rsi
    lea     8(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Lm1_blocks
    jmp     .Lm1_single
.Lm1_blocks:
    mov     %r8, %rcx
    jrcxz   .Lm1_done
.Lm1_loop:
    mulx    (%rsi), %r9, %r10
    adcx    %rax, %r9
    mov     %r9, (%rdi)
    mulx    8(%rsi), %r9, %rax
    adcx    %r10, %r9
    mov     %r9, 8(%rdi)
    mulx    16(%rsi), %r9, %r10
    adcx    %rax, %r9
    mov     %r9, 16(%rdi)
    mulx    24(%rsi), %r9, %rax
    adcx    %r10, %r9
    mov     %r9, 24(%rdi)
    lea     32(%rsi), %rsi
    lea     32(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Lm1_done
    jmp     .Lm1_loop
.Lm1_done:
    mov     $0, %r9d                    # mov keeps the flags
    adcx    %r9, %rax
    ret
    .size   numetron_mul_1_adx, .-numetron_mul_1_adx

    .align  16, 0x90
    .globl  numetron_addmul_1_adx
    .hidden numetron_addmul_1_adx
    .type   numetron_addmul_1_adx, @function
numetron_addmul_1_adx:
    mov     %rdx, %r8
    mov     %rcx, %rdx                  # v: mulx's implicit operand
    mov     %r8, %rcx
    shr     $2, %r8                     # r8 = n / 4
    and     $3, %ecx                    # rcx = n % 4
    xor     %eax, %eax                  # high half 0; CF = OF = 0
    jrcxz   .Lam_blocks
.Lam_single:
    mulx    (%rsi), %r9, %r10
    adcx    %rax, %r9
    adox    (%rdi), %r9
    mov     %r9, (%rdi)
    mov     %r10, %rax
    lea     8(%rsi), %rsi
    lea     8(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Lam_blocks
    jmp     .Lam_single
.Lam_blocks:
    mov     %r8, %rcx
    jrcxz   .Lam_done
.Lam_loop:
    mulx    (%rsi), %r9, %r10
    adcx    %rax, %r9
    adox    (%rdi), %r9
    mov     %r9, (%rdi)
    mulx    8(%rsi), %r9, %rax
    adcx    %r10, %r9
    adox    8(%rdi), %r9
    mov     %r9, 8(%rdi)
    mulx    16(%rsi), %r9, %r10
    adcx    %rax, %r9
    adox    16(%rdi), %r9
    mov     %r9, 16(%rdi)
    mulx    24(%rsi), %r9, %rax
    adcx    %r10, %r9
    adox    24(%rdi), %r9
    mov     %r9, 24(%rdi)
    lea     32(%rsi), %rsi
    lea     32(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Lam_done
    jmp     .Lam_loop
.Lam_done:
    mov     $0, %r9d                    # mov keeps the flags
    adcx    %r9, %rax
    adox    %r9, %rax
    ret
    .size   numetron_addmul_1_adx, .-numetron_addmul_1_adx

    .section .note.GNU-stack,"",@progbits
