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
# un <= 16 takes straight-line code instead (the UMUL macros below): each un has its row with
# every offset fixed, in caller-saved registers only, and the rows loop over v with only the
# pointers moving -- no entry table, counters or saved registers. At 4 x 4 that is 4.7 ns against
# 6.4 for the looped rows (mpn_mul 6.9); 1.43-2.27x mpn_mul's speed for the shapes up to 16
# (2026-09-27, scratch mul_asm.cpp).
#
# SysV x64: rdi rp, rsi up, rdx un, rcx vp, r8 vn. Registers (un > 16):
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

# ---- un <= 16: straight-line rows for a fixed un (rsi up, rdi rp, rcx vp, r8 rows left) ----
# The row set-up above (the entry, the counters, the saved registers) costs about as much as a
# short row's products, as in sqr_basecase_adx.s: here each un has its row as straight-line code
# (every offset fixed, caller-saved registers only), and the rows loop over v with just the
# pointers moving. The high halves alternate r9 / r10.

# one row: rp[0..un) (+)= u * rdx, rp[un] = the limb above; ADD: add into rp (OF chain)
.macro UROW un, add
    xor     %r10d, %r10d                # previous high half = 0, CF = OF = 0
    .set    t, 0
    .rept   \un
    .set    d, 8 * t
    .if t % 2
    mulx    d(%rsi), %rax, %r10
    adcx    %r9, %rax
    .else
    mulx    d(%rsi), %rax, %r9
    adcx    %r10, %rax
    .endif
    .if \add
    adox    d(%rdi), %rax
    .endif
    mov     %rax, d(%rdi)
    .set    t, t + 1
    .endr
    mov     $0, %eax                    # mov keeps the flags
    .set    d, 8 * \un
    .if (\un - 1) % 2                   # the last step's high half
    adcx    %rax, %r10
    .if \add
    adox    %rax, %r10
    .endif
    mov     %r10, d(%rdi)
    .else
    adcx    %rax, %r9
    .if \add
    adox    %rax, %r9
    .endif
    mov     %r9, d(%rdi)
    .endif
.endm

.macro UMUL un
.Lu\un:
    mov     (%rcx), %rdx
    UROW    \un, 0
    dec     %r8
    jz      .Lu\un\()_done
.Lu\un\()_row:
    lea     8(%rcx), %rcx
    lea     8(%rdi), %rdi
    mov     (%rcx), %rdx
    UROW    \un, 1
    dec     %r8
    jnz     .Lu\un\()_row
.Lu\un\()_done:
    ret
.endm

    .align  16, 0x90
    .globl  numetron_mul_basecase_adx
    .hidden numetron_mul_basecase_adx
    .type   numetron_mul_basecase_adx, @function
numetron_mul_basecase_adx:
    cmp     $16, %rdx
    ja      .Lgeneral
    lea     .Lutab(%rip), %rax
    movslq  -4(%rax,%rdx,4), %r9        # entry un - 1
    add     %rax, %r9
    jmp     *%r9
    .irp    un, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    .p2align 4
    UMUL    \un
    .endr

    .p2align 2
.Lutab:
    .irp    un, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    .long   .Lu\un - .Lutab
    .endr

    .p2align 4
.Lgeneral:
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
