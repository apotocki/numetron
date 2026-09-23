# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2025 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

# add_sub_n.s -- r[0..n) = u[0..n) +/- v[0..n), returning the carry/borrow out (0 or 1).
#
#   uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n);
#   uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n);
#
# SysV x64 calling convention: rdi = rp, rsi = up, rdx = vp, rcx = n. Only caller-saved
# registers are used.
#
# The carry/borrow stays in CF for the whole call: between the adc/sbb instructions the loops use
# only mov, lea, dec, jnz and jrcxz, none of which write CF. The n % 4 odd limbs are done first,
# one at a time (CF is still clear on entry), then n / 4 blocks of four. Every limb of up and vp
# is read before the same index of rp is written, so rp may coincide with up or vp (in-place use)
# or trail them.

    .text

    .align  16, 0x90
    .globl  numetron_add_n
    .hidden numetron_add_n
    .type   numetron_add_n, @function
numetron_add_n:
    mov     %ecx, %eax
    shr     $2, %rcx                    # rcx = n / 4
    and     $3, %eax                    # eax = n % 4; clears CF
    jz      .Ladd_blocks
.Ladd_odd:
    mov     (%rsi), %r8
    adc     (%rdx), %r8
    mov     %r8, (%rdi)
    lea     8(%rsi), %rsi
    lea     8(%rdx), %rdx
    lea     8(%rdi), %rdi
    dec     %eax
    jnz     .Ladd_odd
.Ladd_blocks:
    jrcxz   .Ladd_done
.Ladd_loop:
    mov     (%rsi), %r8
    mov     8(%rsi), %r9
    adc     (%rdx), %r8
    adc     8(%rdx), %r9
    mov     %r8, (%rdi)
    mov     %r9, 8(%rdi)
    mov     16(%rsi), %r8
    mov     24(%rsi), %r9
    adc     16(%rdx), %r8
    adc     24(%rdx), %r9
    mov     %r8, 16(%rdi)
    mov     %r9, 24(%rdi)
    lea     32(%rsi), %rsi
    lea     32(%rdx), %rdx
    lea     32(%rdi), %rdi
    dec     %rcx
    jnz     .Ladd_loop
.Ladd_done:
    mov     $0, %eax                    # mov, not xor: xor would clear CF
    adc     %eax, %eax                  # rax = CF
    ret
    .size   numetron_add_n, .-numetron_add_n

    .align  16, 0x90
    .globl  numetron_sub_n
    .hidden numetron_sub_n
    .type   numetron_sub_n, @function
numetron_sub_n:
    mov     %ecx, %eax
    shr     $2, %rcx                    # rcx = n / 4
    and     $3, %eax                    # eax = n % 4; clears CF
    jz      .Lsub_blocks
.Lsub_odd:
    mov     (%rsi), %r8
    sbb     (%rdx), %r8
    mov     %r8, (%rdi)
    lea     8(%rsi), %rsi
    lea     8(%rdx), %rdx
    lea     8(%rdi), %rdi
    dec     %eax
    jnz     .Lsub_odd
.Lsub_blocks:
    jrcxz   .Lsub_done
.Lsub_loop:
    mov     (%rsi), %r8
    mov     8(%rsi), %r9
    sbb     (%rdx), %r8
    sbb     8(%rdx), %r9
    mov     %r8, (%rdi)
    mov     %r9, 8(%rdi)
    mov     16(%rsi), %r8
    mov     24(%rsi), %r9
    sbb     16(%rdx), %r8
    sbb     24(%rdx), %r9
    mov     %r8, 16(%rdi)
    mov     %r9, 24(%rdi)
    lea     32(%rsi), %rsi
    lea     32(%rdx), %rdx
    lea     32(%rdi), %rdi
    dec     %rcx
    jnz     .Lsub_loop
.Lsub_done:
    mov     $0, %eax                    # mov, not xor: xor would clear CF
    adc     %eax, %eax                  # rax = CF (the borrow)
    ret
    .size   numetron_sub_n, .-numetron_sub_n

    .section .note.GNU-stack,"",@progbits
