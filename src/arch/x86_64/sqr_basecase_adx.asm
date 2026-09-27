; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; sqr_basecase_adx.asm -- schoolbook squaring with mulx + adcx/adox (BMI2 + ADX), Microsoft x64
; version of sqr_basecase_adx.s (see there for the algorithm: the rows above the diagonal with
; two carry chains, then one pass doubling them and adding the diagonal; straight-line code for
; n <= 16 and for the last 15 rows above that, a 16-limb body entered per row from a jump table
; for the longer rows).
;
;   void numetron_sqr_basecase_adx(uint64_t* rp, const uint64_t* up, size_t n);
;
; rp[0..2n) = u[0..n)^2, n >= 2, rp not overlapping u.
;
; Microsoft x64: rcx rp, rdx up, r8 n. n <= 16: a leaf procedure in volatile registers only
; (r11 up, rcx rp), no unwind info needed. Above: sqr_basecase_adx_general, which saves the
; non-volatile registers (with unwind info) and then uses the same registers as the GAS version:
;   rdx  u[i] (mulx's implicit operand)      rsi  up cursor         rdi  rp cursor
;   rax  low half / sum                      r8, r9  high halves (even / odd steps)
;   rcx  pass counter (jrcxz)                r10  8 * k             r11  entry
;   rbx  rp + 2i + 1 (the row's start)       r12  up + i            r13  L, the row's length
;   r14  n                                   r15  rp                rbp  up

OPTION CASEMAP:NONE

PUBLIC numetron_sqr_basecase_adx

MSTEP MACRO s
IF (s AND 1)
    mulx    r9, rax, QWORD PTR [rsi + 8*s]
    adcx    rax, r8
ELSE
    mulx    r8, rax, QWORD PTR [rsi + 8*s]
    adcx    rax, r9
ENDIF
    mov     QWORD PTR [rdi + 8*s], rax
ENDM

ASTEP MACRO s
IF (s AND 1)
    mulx    r9, rax, QWORD PTR [rsi + 8*s]
    adcx    rax, r8                     ; + previous high half (CF chain)
ELSE
    mulx    r8, rax, QWORD PTR [rsi + 8*s]
    adcx    rax, r9
ENDIF
    adox    rax, QWORD PTR [rdi + 8*s]  ; + rp[i] (OF chain)
    mov     QWORD PTR [rdi + 8*s], rax
ENDM

; rp[2s], rp[2s+1] = 2 * (rp[2s], rp[2s+1]) + u[s]^2 (+ both chains' carries)
DSTEP MACRO s, up, rp
    mov     rdx, QWORD PTR [up + 8*s]
    mulx    r8, rax, rdx                ; u[s]^2 = r8:rax
    mov     r9, QWORD PTR [rp + 16*s]
    adox    r9, r9                      ; doubled (OF chain)
    adcx    r9, rax                     ; + low half (CF chain)
    mov     QWORD PTR [rp + 16*s], r9
    mov     r10, QWORD PTR [rp + 16*s + 8]
    adox    r10, r10
    adcx    r10, r8                     ; + high half
    mov     QWORD PTR [rp + 16*s + 8], r10
ENDM

; Row set-up: rsi, rdi = the row's up / rp start moved back k limbs, rcx = ceil(L / 16) passes,
; r11 = the entry from the table, both high halves and both flags zero. In: r13 = L, rdx = u[i].
ROWSETUP MACRO tab
    mov     r10, r13
    neg     r10
    and     r10, 15                     ; k = -L & 15, the entry step
    lea     rcx, [r13 + 15]
    shr     rcx, 4                      ; passes: ceil(L / 16)
    lea     r11, tab
    mov     r11, QWORD PTR [r11 + r10*8]
    shl     r10, 3                      ; 8k: the pointers start k limbs early
    lea     rsi, [r12 + 8]
    sub     rsi, r10
    mov     rdi, rbx
    sub     rdi, r10
    xor     r8d, r8d
    xor     r9d, r9d                    ; high halves = 0, CF = OF = 0
ENDM

; ---- straight-line code for a fixed n ----

; rp[1..n] = u[1..n) * u[0]; step t multiplies u[1+t], the high halves alternating r8 / r9
SROW0 MACRO n, up, rp
    mov     rdx, QWORD PTR [up]
    xor     r9d, r9d                    ; previous high half = 0, CF = OF = 0
    sq_t = 0
    REPT n - 1
    sq_du = 8 * (1 + sq_t)
IF (sq_t AND 1)
    mulx    r9, rax, QWORD PTR [up + sq_du]
    adcx    rax, r8
ELSE
    mulx    r8, rax, QWORD PTR [up + sq_du]
    adcx    rax, r9
ENDIF
    mov     QWORD PTR [rp + sq_du], rax
    sq_t = sq_t + 1
    ENDM
    mov     eax, 0                      ; mov keeps the flags
IF ((n - 2) AND 1)                      ; the last step's high half
    adcx    r9, rax
    mov     QWORD PTR [rp + 8*n], r9
ELSE
    adcx    r8, rax
    mov     QWORD PTR [rp + 8*n], r8
ENDIF
ENDM

; rp[2i+1..i+n) += u[i+1..n) * u[i], rp[i+n] = the limb above
SROW MACRO i, n, up, rp
    mov     rdx, QWORD PTR [up + 8*i]
    xor     r9d, r9d
    sq_t = 0
    REPT n - 1 - i
    sq_du = 8 * (i + 1 + sq_t)
    sq_dr = 8 * (2 * i + 1 + sq_t)
IF (sq_t AND 1)
    mulx    r9, rax, QWORD PTR [up + sq_du]
    adcx    rax, r8
ELSE
    mulx    r8, rax, QWORD PTR [up + sq_du]
    adcx    rax, r9
ENDIF
    adox    rax, QWORD PTR [rp + sq_dr]
    mov     QWORD PTR [rp + sq_dr], rax
    sq_t = sq_t + 1
    ENDM
    mov     eax, 0
    sq_dr = 8 * (i + n)
IF ((n - 2 - i) AND 1)
    adcx    r9, rax
    adox    r9, rax
    mov     QWORD PTR [rp + sq_dr], r9
ELSE
    adcx    r8, rax
    adox    r8, rax
    mov     QWORD PTR [rp + sq_dr], r8
ENDIF
ENDM

; rp = 2 * rp + the diagonal
SDIAG MACRO n, up, rp
    xor     eax, eax                    ; CF = OF = 0
    sq_t = 0
    REPT n
    DSTEP   %sq_t, up, rp
    sq_t = sq_t + 1
    ENDM
ENDM

SQRN MACRO n
    mov     QWORD PTR [rcx], 0          ; rp[0]
    mov     QWORD PTR [rcx + 8*(2*n - 1)], 0 ; rp[2n-1]
    SROW0   n, r11, rcx
    sq_i = 1
    REPT n - 2
    SROW    %sq_i, n, r11, rcx
    sq_i = sq_i + 1
    ENDM
    SDIAG   n, r11, rcx
    ret
ENDM

.code

ALIGN 16
numetron_sqr_basecase_adx PROC
    cmp     r8, 16
    ja      sqr_basecase_adx_general
    mov     r11, rdx                    ; up (rdx is mulx's operand)
    lea     rax, stab
    jmp     QWORD PTR [rax + r8*8 - 16] ; entry n - 2
ALIGN 16
s2: SQRN 2
ALIGN 16
s3: SQRN 3
ALIGN 16
s4: SQRN 4
ALIGN 16
s5: SQRN 5
ALIGN 16
s6: SQRN 6
ALIGN 16
s7: SQRN 7
ALIGN 16
s8: SQRN 8
ALIGN 16
s9: SQRN 9
ALIGN 16
s10: SQRN 10
ALIGN 16
s11: SQRN 11
ALIGN 16
s12: SQRN 12
ALIGN 16
s13: SQRN 13
ALIGN 16
s14: SQRN 14
ALIGN 16
s15: SQRN 15
ALIGN 16
s16: SQRN 16

ALIGN 8
stab    DQ s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16
numetron_sqr_basecase_adx ENDP

ALIGN 16
sqr_basecase_adx_general PROC FRAME
    push    rbx
    .pushreg rbx
    push    rbp
    .pushreg rbp
    push    rdi
    .pushreg rdi
    push    rsi
    .pushreg rsi
    push    r12
    .pushreg r12
    push    r13
    .pushreg r13
    push    r14
    .pushreg r14
    push    r15
    .pushreg r15
    .endprolog

    mov     r15, rcx                    ; rp
    mov     rbp, rdx                    ; up
    mov     r14, r8                     ; n
    mov     r12, rdx                    ; u[0]
    lea     rbx, [rcx + 8]              ; rp + 1
    lea     r13, [r8 - 1]               ; L = n - 1
    mov     QWORD PTR [rcx], 0          ; rp[0]
    lea     rax, [rcx + r8*8]
    mov     QWORD PTR [rax + r8*8 - 8], 0 ; rp[2n-1]

; ---- row 0: rp[1..n] = u[1..n) * u[0] (one chain) ----
    mov     rdx, QWORD PTR [r12]
    ROWSETUP mtab
    jmp     r11
m0: MSTEP 0
m1: MSTEP 1
m2: MSTEP 2
m3: MSTEP 3
m4: MSTEP 4
m5: MSTEP 5
m6: MSTEP 6
m7: MSTEP 7
m8: MSTEP 8
m9: MSTEP 9
m10: MSTEP 10
m11: MSTEP 11
m12: MSTEP 12
m13: MSTEP 13
m14: MSTEP 14
m15: MSTEP 15
    lea     rsi, [rsi + 128]
    lea     rdi, [rdi + 128]
    lea     rcx, [rcx - 1]
    jrcxz   mdone
    jmp     m0
mdone:
    mov     eax, 0                      ; mov keeps the flags
    adcx    r9, rax
    mov     QWORD PTR [rdi], r9         ; rp[n]

; ---- rows i = 1..n-17: rp[2i+1..i+n) += u[i+1..n) * u[i], rp[i+n] = the limb above ----
row:
    dec     r13                         ; L = n - 1 - i
    cmp     r13, 16
    jb      tail
    lea     r12, [r12 + 8]
    lea     rbx, [rbx + 16]
    mov     rdx, QWORD PTR [r12]
    ROWSETUP atab
    jmp     r11
a0: ASTEP 0
a1: ASTEP 1
a2: ASTEP 2
a3: ASTEP 3
a4: ASTEP 4
a5: ASTEP 5
a6: ASTEP 6
a7: ASTEP 7
a8: ASTEP 8
a9: ASTEP 9
a10: ASTEP 10
a11: ASTEP 11
a12: ASTEP 12
a13: ASTEP 13
a14: ASTEP 14
a15: ASTEP 15
    lea     rsi, [rsi + 128]
    lea     rdi, [rdi + 128]
    lea     rcx, [rcx - 1]
    jrcxz   adone
    jmp     a0
adone:
    mov     eax, 0
    adcx    r9, rax
    adox    r9, rax
    mov     QWORD PTR [rdi], r9         ; rp[i+n]
    jmp     row

; ---- the last 15 rows (L = 15..1): the straight-line rows 0..14 of n = 16 at up + (n - 16),
; rp + 2 (n - 16) ----
tail:
    lea     rsi, [rbp + r14*8 - 128]    ; up + n - 16
    lea     rdi, [r15 + r14*8 - 256]
    lea     rdi, [rdi + r14*8]          ; rp + 2 (n - 16)
    sq_i = 0
    REPT 15
    SROW    %sq_i, 16, rsi, rdi
    sq_i = sq_i + 1
    ENDM

; ---- rp = 2 * rp + the diagonal ----
    mov     r10, r14
    neg     r10
    and     r10, 3                      ; k = -n & 3, the entry step
    lea     rcx, [r14 + 3]
    shr     rcx, 2                      ; passes: ceil(n / 4)
    lea     r11, dtab
    mov     r11, QWORD PTR [r11 + r10*8]
    shl     r10, 3
    mov     rsi, rbp
    sub     rsi, r10                    ; up - k
    add     r10, r10
    mov     rdi, r15
    sub     rdi, r10                    ; rp - 2k
    xor     eax, eax                    ; CF = OF = 0
    jmp     r11
d0: DSTEP 0, rsi, rdi
d1: DSTEP 1, rsi, rdi
d2: DSTEP 2, rsi, rdi
d3: DSTEP 3, rsi, rdi
    lea     rsi, [rsi + 32]
    lea     rdi, [rdi + 64]
    lea     rcx, [rcx - 1]
    jrcxz   ddone
    jmp     d0
ddone:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rdi
    pop     rbp
    pop     rbx
    ret

ALIGN 8
mtab    DQ m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
atab    DQ a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
dtab    DQ d0, d1, d2, d3
sqr_basecase_adx_general ENDP

END
