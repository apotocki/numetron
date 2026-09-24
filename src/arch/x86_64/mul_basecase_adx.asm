; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2026 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; mul_basecase_adx.asm -- schoolbook multiplication with mulx + adcx/adox (BMI2 + ADX), Microsoft
; x64 version of mul_basecase_adx.s (see there for the algorithm: rows with two carry chains, an
; 16-limb unrolled body entered at step -un & 15 from a jump table).
;
;   void numetron_mul_basecase_adx(uint64_t* rp, const uint64_t* up, size_t un,
;                                  const uint64_t* vp, size_t vn);
;
; Microsoft x64: rcx rp, rdx up, r8 un, r9 vp, [rsp+40] vn. After saving the non-volatile
; registers (with unwind info) the body uses the same registers as the GAS version:
;   rdx  v[j] (mulx's implicit operand)      rsi  up cursor         rdi  rp cursor
;   rax  low half / sum                      r8, r9  high halves (even / odd steps)
;   rcx  pass counter (jrcxz)                rbp  passes per row    r10  8 * k
;   r11  entry into the row-0 body           r15  entry into the add-row body
;   rbx  rp of the row                       r12  up                r13  vp cursor
;   r14  rows left

OPTION CASEMAP:NONE

PUBLIC numetron_mul_basecase_adx

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
    adox    rax, QWORD PTR [rdi + 8*s]  ; + r[i] (OF chain)
    mov     QWORD PTR [rdi + 8*s], rax
ENDM

.code

ALIGN 16
numetron_mul_basecase_adx PROC FRAME
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

    mov     rbx, rcx                    ; rp
    mov     r12, rdx                    ; up
    mov     r13, r9                     ; vp
    mov     r14, QWORD PTR [rsp + 104]  ; vn (8 pushes + return address + 32 home)
    lea     rbp, [r8 + 15]
    shr     rbp, 4                      ; passes per row: ceil(un / 16)
    mov     r10, r8
    neg     r10
    and     r10, 15                     ; k = -un & 15, the entry step
    lea     r11, mtab
    mov     r11, QWORD PTR [r11 + r10*8] ; row-0 entry
    lea     r15, atab
    mov     r15, QWORD PTR [r15 + r10*8] ; add-row entry
    shl     r10, 3                      ; 8k: the pointers start k limbs early

; ---- row 0: rp[0..un] = u * v[0] (one chain) ----
    mov     rdx, QWORD PTR [r13]
    mov     rsi, r12
    sub     rsi, r10
    mov     rdi, rbx
    sub     rdi, r10
    mov     rcx, rbp
    xor     r8d, r8d
    xor     r9d, r9d                    ; high halves = 0, CF = OF = 0
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
    mov     QWORD PTR [rdi], r9         ; rp[un]

; ---- rows 1..vn-1: rp[j..j+un) += u * v[j], rp[j+un] = the limb above ----
    dec     r14
    jz      all_done
row:
    lea     r13, [r13 + 8]
    lea     rbx, [rbx + 8]
    mov     rdx, QWORD PTR [r13]
    mov     rsi, r12
    sub     rsi, r10
    mov     rdi, rbx
    sub     rdi, r10
    mov     rcx, rbp
    xor     r8d, r8d
    xor     r9d, r9d                    ; high halves = 0, CF = OF = 0
    jmp     r15
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
    mov     QWORD PTR [rdi], r9         ; rp[j+un]
    dec     r14
    jnz     row

all_done:
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
numetron_mul_basecase_adx ENDP

END
