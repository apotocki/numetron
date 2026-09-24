# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2026 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

# karatsuba_interp.s -- the fused middle-column pass of Karatsuba interpolation.
#
#   uint64_t numetron_karatsuba_interp(uint64_t* rp, size_t n, size_t h);
#
# On entry rp[0..2n) = v0 = L0 + H0*B^n and rp[2n..3n+h) = vinf = Li + Hi*B^n (Hi has h <= n
# limbs, 1 <= n). One pass over i in [0, n) computes, with X = H0 + Li:
#
#   rp[n+i]  = X[i] + L0[i]          (the column at n:  L0 + H0 + Li)
#   rp[2n+i] = X[i] + Hi[i]          (the column at 2n: H0 + Li + Hi; Hi[i] = 0 for i >= h)
#
# replacing three separate add_n passes. The carries out of the n-limb windows are returned,
# not applied: bits 0..31 hold cX + cL (the carry into limb 2n), bits 32..63 cX + cH (into limb
# 3n), each in [0, 2]; cX, cL, cH are the carries of X, X + L0, X + Hi.
#
# Reads at index i happen before the writes at index i, and rp[n+i] / rp[2n+i] are never read
# again after being written, so the pass is safe in place.
#
# SysV x64 calling convention: rdi = rp, rsi = n, rdx = h.
#
# Three carry chains run side by side. Each keeps its carry as 0/1 in a byte register between
# uses: `add $0xff, r8b` loads it into CF (0xff + 1 carries out), `setc r8b` stores it back.
# Blocks of four limbs amortize that over four adc's per chain. Register use:
#   rdi  p = rp + n + i        (H0 at (p), Li at (p,n8), Hi at (p,n8,2), L0 at (p,m))
#   rsi  n8 = 8n,  rbp  m = -8n,  rdx  limb count of the current phase
#   r8..r11   X (then X + L0),  r12..r15  X + Hi
#   al  cX,  bl  cL,  cl  cH
# n - h, the count of the second phase (no Hi), is kept in the red zone.

    .text

    .align  16, 0x90
    .globl  numetron_karatsuba_interp
    .hidden numetron_karatsuba_interp
    .type   numetron_karatsuba_interp, @function
numetron_karatsuba_interp:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15

    mov     %rsi, %rcx
    sub     %rdx, %rcx
    mov     %rcx, -8(%rsp)              # n - h (red zone)
    lea     (%rdi,%rsi,8), %rdi         # p = rp + n
    shl     $3, %rsi                    # n8
    mov     %rsi, %rbp
    neg     %rbp                        # m = -n8
    xor     %eax, %eax
    xor     %ebx, %ebx
    xor     %ecx, %ecx

# ---- phase A: i in [0, h), Hi present ----
    test    $3, %dl
    jz      .LA_blocks
.LA_single:
    mov     (%rdi), %r8
    add     $0xff, %al
    adc     (%rdi,%rsi), %r8            # X = H0 + Li
    setc    %al
    mov     (%rdi,%rsi,2), %r12
    add     $0xff, %cl
    adc     %r8, %r12                   # X + Hi
    setc    %cl
    add     $0xff, %bl
    adc     (%rdi,%rbp), %r8            # X + L0
    setc    %bl
    mov     %r8, (%rdi)
    mov     %r12, (%rdi,%rsi)
    lea     8(%rdi), %rdi
    dec     %rdx
    test    $3, %dl
    jnz     .LA_single
.LA_blocks:
    test    %rdx, %rdx
    jz      .LB_start
.LA_loop:
    mov     (%rdi), %r8
    mov     8(%rdi), %r9
    mov     16(%rdi), %r10
    mov     24(%rdi), %r11
    add     $0xff, %al
    adc     (%rdi,%rsi), %r8
    adc     8(%rdi,%rsi), %r9
    adc     16(%rdi,%rsi), %r10
    adc     24(%rdi,%rsi), %r11
    setc    %al
    mov     (%rdi,%rsi,2), %r12
    mov     8(%rdi,%rsi,2), %r13
    mov     16(%rdi,%rsi,2), %r14
    mov     24(%rdi,%rsi,2), %r15
    add     $0xff, %cl
    adc     %r8, %r12
    adc     %r9, %r13
    adc     %r10, %r14
    adc     %r11, %r15
    setc    %cl
    add     $0xff, %bl
    adc     (%rdi,%rbp), %r8
    adc     8(%rdi,%rbp), %r9
    adc     16(%rdi,%rbp), %r10
    adc     24(%rdi,%rbp), %r11
    setc    %bl
    mov     %r8, (%rdi)
    mov     %r9, 8(%rdi)
    mov     %r10, 16(%rdi)
    mov     %r11, 24(%rdi)
    mov     %r12, (%rdi,%rsi)
    mov     %r13, 8(%rdi,%rsi)
    mov     %r14, 16(%rdi,%rsi)
    mov     %r15, 24(%rdi,%rsi)
    lea     32(%rdi), %rdi
    sub     $4, %rdx
    jnz     .LA_loop

# ---- phase B: i in [h, n), Hi = 0 ----
.LB_start:
    mov     -8(%rsp), %rdx
    test    $3, %dl
    jz      .LB_blocks
.LB_single:
    mov     (%rdi), %r8
    add     $0xff, %al
    adc     (%rdi,%rsi), %r8
    setc    %al
    mov     %r8, %r12
    add     $0xff, %cl
    adc     $0, %r12
    setc    %cl
    add     $0xff, %bl
    adc     (%rdi,%rbp), %r8
    setc    %bl
    mov     %r8, (%rdi)
    mov     %r12, (%rdi,%rsi)
    lea     8(%rdi), %rdi
    dec     %rdx
    test    $3, %dl
    jnz     .LB_single
.LB_blocks:
    test    %rdx, %rdx
    jz      .Ldone
.LB_loop:
    mov     (%rdi), %r8
    mov     8(%rdi), %r9
    mov     16(%rdi), %r10
    mov     24(%rdi), %r11
    add     $0xff, %al
    adc     (%rdi,%rsi), %r8
    adc     8(%rdi,%rsi), %r9
    adc     16(%rdi,%rsi), %r10
    adc     24(%rdi,%rsi), %r11
    setc    %al
    mov     %r8, %r12
    mov     %r9, %r13
    mov     %r10, %r14
    mov     %r11, %r15
    add     $0xff, %cl
    adc     $0, %r12
    adc     $0, %r13
    adc     $0, %r14
    adc     $0, %r15
    setc    %cl
    add     $0xff, %bl
    adc     (%rdi,%rbp), %r8
    adc     8(%rdi,%rbp), %r9
    adc     16(%rdi,%rbp), %r10
    adc     24(%rdi,%rbp), %r11
    setc    %bl
    mov     %r8, (%rdi)
    mov     %r9, 8(%rdi)
    mov     %r10, 16(%rdi)
    mov     %r11, 24(%rdi)
    mov     %r12, (%rdi,%rsi)
    mov     %r13, 8(%rdi,%rsi)
    mov     %r14, 16(%rdi,%rsi)
    mov     %r15, 24(%rdi,%rsi)
    lea     32(%rdi), %rdi
    sub     $4, %rdx
    jnz     .LB_loop

.Ldone:
    movzbl  %al, %eax
    movzbl  %bl, %ebx
    movzbl  %cl, %ecx
    add     %eax, %ecx                  # cX + cH
    add     %ebx, %eax                  # cX + cL
    shl     $32, %rcx
    or      %rcx, %rax

    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
    .size   numetron_karatsuba_interp, .-numetron_karatsuba_interp

    .section .note.GNU-stack,"",@progbits
