; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; mul_1_adx.asm -- one row of a schoolbook product with mulx + adcx/adox (BMI2 + ADX).
;
;   uint64_t numetron_mul_1_adx(uint64_t* rp, const uint64_t* up, size_t n, uint64_t v);
;       rp[0..n) = low n limbs of up[0..n) * v, returns the high limb
;   uint64_t numetron_addmul_1_adx(uint64_t* rp, const uint64_t* up, size_t n, uint64_t v);
;       rp[0..n) += up[0..n) * v, returns the limb carried out
;
; The rows of the squaring basecase (usqr_basecase), whose lengths vary. Microsoft x64 calling
; convention: rcx = rp, rdx = up, r8 = n, r9 = v. Leaf functions using only volatile registers
; (xmm0 holds the block count), so they need no unwind info.
;
; A limb's product lo:hi goes in two independent carry chains: adcx (CF) adds the previous high
; half, adox (OF) adds rp[i] (addmul only). Between them the loops use only mov, lea and jrcxz,
; which leave both flags alone (dec would write OF). The n % 4 single limbs first, then n / 4
; blocks of four, the high halves alternating between r8 and rax.

OPTION CASEMAP:NONE

PUBLIC numetron_mul_1_adx
PUBLIC numetron_addmul_1_adx

.code

ALIGN 16
numetron_mul_1_adx PROC
    mov     r11, rcx                    ; rp
    mov     r10, rdx                    ; up
    mov     rdx, r9                     ; v: mulx's implicit operand
    mov     rcx, r8
    shr     r8, 2
    movq    xmm0, r8                    ; n / 4
    and     ecx, 3                      ; rcx = n % 4
    xor     eax, eax                    ; high half 0; CF = OF = 0
    jrcxz   m1_blocks
m1_single:
    mulx    r8, r9, QWORD PTR [r10]
    adcx    r9, rax
    mov     QWORD PTR [r11], r9
    mov     rax, r8
    lea     r10, [r10 + 8]
    lea     r11, [r11 + 8]
    lea     rcx, [rcx - 1]
    jrcxz   m1_blocks
    jmp     m1_single
m1_blocks:
    movq    rcx, xmm0
    jrcxz   m1_done
m1_loop:
    mulx    r8, r9, QWORD PTR [r10]
    adcx    r9, rax
    mov     QWORD PTR [r11], r9
    mulx    rax, r9, QWORD PTR [r10 + 8]
    adcx    r9, r8
    mov     QWORD PTR [r11 + 8], r9
    mulx    r8, r9, QWORD PTR [r10 + 16]
    adcx    r9, rax
    mov     QWORD PTR [r11 + 16], r9
    mulx    rax, r9, QWORD PTR [r10 + 24]
    adcx    r9, r8
    mov     QWORD PTR [r11 + 24], r9
    lea     r10, [r10 + 32]
    lea     r11, [r11 + 32]
    lea     rcx, [rcx - 1]
    jrcxz   m1_done
    jmp     m1_loop
m1_done:
    mov     r9d, 0                      ; mov keeps the flags
    adcx    rax, r9
    ret
numetron_mul_1_adx ENDP

ALIGN 16
numetron_addmul_1_adx PROC
    mov     r11, rcx                    ; rp
    mov     r10, rdx                    ; up
    mov     rdx, r9                     ; v: mulx's implicit operand
    mov     rcx, r8
    shr     r8, 2
    movq    xmm0, r8                    ; n / 4
    and     ecx, 3                      ; rcx = n % 4
    xor     eax, eax                    ; high half 0; CF = OF = 0
    jrcxz   am_blocks
am_single:
    mulx    r8, r9, QWORD PTR [r10]
    adcx    r9, rax
    adox    r9, QWORD PTR [r11]
    mov     QWORD PTR [r11], r9
    mov     rax, r8
    lea     r10, [r10 + 8]
    lea     r11, [r11 + 8]
    lea     rcx, [rcx - 1]
    jrcxz   am_blocks
    jmp     am_single
am_blocks:
    movq    rcx, xmm0
    jrcxz   am_done
am_loop:
    mulx    r8, r9, QWORD PTR [r10]
    adcx    r9, rax
    adox    r9, QWORD PTR [r11]
    mov     QWORD PTR [r11], r9
    mulx    rax, r9, QWORD PTR [r10 + 8]
    adcx    r9, r8
    adox    r9, QWORD PTR [r11 + 8]
    mov     QWORD PTR [r11 + 8], r9
    mulx    r8, r9, QWORD PTR [r10 + 16]
    adcx    r9, rax
    adox    r9, QWORD PTR [r11 + 16]
    mov     QWORD PTR [r11 + 16], r9
    mulx    rax, r9, QWORD PTR [r10 + 24]
    adcx    r9, r8
    adox    r9, QWORD PTR [r11 + 24]
    mov     QWORD PTR [r11 + 24], r9
    lea     r10, [r10 + 32]
    lea     r11, [r11 + 32]
    lea     rcx, [rcx - 1]
    jrcxz   am_done
    jmp     am_loop
am_done:
    mov     r9d, 0                      ; mov keeps the flags
    adcx    rax, r9
    adox    rax, r9
    ret
numetron_addmul_1_adx ENDP

END
