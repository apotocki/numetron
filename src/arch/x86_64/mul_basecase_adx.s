# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2026 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

# mul_basecase_adx.s -- schoolbook multiplication with mulx + adcx/adox (BMI2 + ADX).
#
#   void numetron_mul_basecase_adx(uint64_t* rp, const uint64_t* up, size_t un,
#                                  const uint64_t* vp, size_t vn);
#
# rp[0..un+vn) = u[0..un) * v[0..vn), un >= vn >= 1, rp not overlapping u, v. The same contract
# as the mul_basecase routines, so it can take their place.
#
# Operand scanning: the first row writes rp[0..un] = u * v[0], each further row j adds
# u * v[j] into rp[j..j+un) and writes rp[j+un]. A row is two additions -- the low halves of the
# products into r, the high halves into the next position -- run as two independent carry
# chains: adcx (CF) adds the previous high half, adox (OF) adds r[i]. mulx doesn't touch the
# flags, and the loop control uses only lea and jrcxz, so both chains stay in the flags for the
# whole row. (Two rows per pass would need four chains -- five addends per limb -- which two
# flags can't carry, so it stays one row per pass.)
#
# A row runs a 16-limb unrolled body. For un % 16 != 0 the first pass enters the body at step
# k = -un & 15 (from a jump table, computed once), with the pointers moved back by k limbs, so
# there is no separate loop for the remainder; the high-half registers start at zero, which is
# the right "previous high" whichever step comes first. Every pass ends at step 15, whose high
# half is in r9. (adcx, adox and the branches all issue on the same two ports, two carry ops
# per limb plus jrcxz + jmp per pass: 16 limbs per pass keep the branches at ~6% of that port
# pressure, 8 limbs measured ~12% slower than GMP at 64x64.)
#
# SysV x64: rdi rp, rsi up, rdx un, rcx vp, r8 vn. Registers:
#   rdx  v[j] (mulx's implicit operand)      rsi  up cursor         rdi  rp cursor
#   rax  low half / sum                      r8, r9  high halves (even / odd steps)
#   rcx  pass counter (jrcxz)                rbp  passes per row    r10  8 * k
#   r11  entry into the row-0 body           r15  entry into the add-row body
#   rbx  rp of the row                       r12  up                r13  vp cursor
#   r14  rows left

    .text

.macro MSTEP s
    .if \s % 2
    mulx    8*\s(%rsi), %rax, %r9
    adcx    %r8, %rax
    .else
    mulx    8*\s(%rsi), %rax, %r8
    adcx    %r9, %rax
    .endif
    mov     %rax, 8*\s(%rdi)
.endm

.macro ASTEP s
    .if \s % 2
    mulx    8*\s(%rsi), %rax, %r9
    adcx    %r8, %rax                   # + previous high half (CF chain)
    .else
    mulx    8*\s(%rsi), %rax, %r8
    adcx    %r9, %rax
    .endif
    adox    8*\s(%rdi), %rax            # + r[i] (OF chain)
    mov     %rax, 8*\s(%rdi)
.endm

    .align  16, 0x90
    .globl  numetron_mul_basecase_adx
    .hidden numetron_mul_basecase_adx
    .type   numetron_mul_basecase_adx, @function
numetron_mul_basecase_adx:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15

    mov     %rdi, %rbx
    mov     %rsi, %r12
    mov     %rcx, %r13
    mov     %r8, %r14
    lea     15(%rdx), %rbp
    shr     $4, %rbp                    # passes per row: ceil(un / 16)
    mov     %rdx, %r10
    neg     %r10
    and     $15, %r10                   # k = -un & 15, the entry step
    lea     .Lmtab(%rip), %r11
    movslq  (%r11,%r10,4), %rax
    add     %rax, %r11                  # row-0 entry
    lea     .Latab(%rip), %r15
    movslq  (%r15,%r10,4), %rax
    add     %rax, %r15                  # add-row entry
    shl     $3, %r10                    # 8k: the pointers start k limbs early

# ---- row 0: rp[0..un] = u * v[0] (one chain) ----
    mov     (%r13), %rdx
    mov     %r12, %rsi
    sub     %r10, %rsi
    mov     %rbx, %rdi
    sub     %r10, %rdi
    mov     %rbp, %rcx
    xor     %r8d, %r8d
    xor     %r9d, %r9d                  # high halves = 0, CF = OF = 0
    jmp     *%r11
    .irp    s, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
.Lm\s:
    MSTEP   \s
    .endr
    lea     128(%rsi), %rsi
    lea     128(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Lmdone
    jmp     .Lm0
.Lmdone:
    mov     $0, %eax                    # mov keeps the flags
    adcx    %rax, %r9
    mov     %r9, (%rdi)                 # rp[un]

# ---- rows 1..vn-1: rp[j..j+un) += u * v[j], rp[j+un] = the limb above ----
    dec     %r14
    jz      .Lend
.Lrow:
    lea     8(%r13), %r13
    lea     8(%rbx), %rbx
    mov     (%r13), %rdx
    mov     %r12, %rsi
    sub     %r10, %rsi
    mov     %rbx, %rdi
    sub     %r10, %rdi
    mov     %rbp, %rcx
    xor     %r8d, %r8d
    xor     %r9d, %r9d                  # high halves = 0, CF = OF = 0
    jmp     *%r15
    .irp    s, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
.La\s:
    ASTEP   \s
    .endr
    lea     128(%rsi), %rsi
    lea     128(%rdi), %rdi
    lea     -1(%rcx), %rcx
    jrcxz   .Ladone
    jmp     .La0
.Ladone:
    mov     $0, %eax
    adcx    %rax, %r9
    adox    %rax, %r9
    mov     %r9, (%rdi)                 # rp[j+un]
    dec     %r14
    jnz     .Lrow

.Lend:
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

    .p2align 2
.Lmtab:
    .irp    s, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    .long   .Lm\s - .Lmtab
    .endr
.Latab:
    .irp    s, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    .long   .La\s - .Latab
    .endr
    .size   numetron_mul_basecase_adx, .-numetron_mul_basecase_adx

    .section .note.GNU-stack,"",@progbits
