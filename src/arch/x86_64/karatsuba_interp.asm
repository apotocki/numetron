; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2026 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; karatsuba_interp.asm -- the fused middle-column pass of Karatsuba interpolation.
;
;   uint64_t numetron_karatsuba_interp(uint64_t* rp, size_t n, size_t h);
;
; On entry rp[0..2n) = v0 = L0 + H0*B^n and rp[2n..3n+h) = vinf = Li + Hi*B^n (Hi has h <= n
; limbs, 1 <= n). One pass over i in [0, n) computes, with X = H0 + Li:
;
;   rp[n+i]  = X[i] + L0[i]          (the column at n:  L0 + H0 + Li)
;   rp[2n+i] = X[i] + Hi[i]          (the column at 2n: H0 + Li + Hi; Hi[i] = 0 for i >= h)
;
; replacing three separate add_n passes. The carries out of the n-limb windows are returned,
; not applied: bits 0..31 hold cX + cL (the carry into limb 2n), bits 32..63 cX + cH (into limb
; 3n), each in [0, 2]; cX, cL, cH are the carries of X, X + L0, X + Hi.
;
; Reads at index i happen before the writes at index i, and rp[n+i] / rp[2n+i] are never read
; again after being written, so the pass is safe in place.
;
; Microsoft x64 calling convention: rcx = rp, rdx = n, r8 = h. The body is the same as the GAS
; version (karatsuba_interp.s), register for register: the arguments are moved to rdi / rsi / rdx
; after saving the non-volatile registers (with unwind info), and n - h goes to the caller's home
; space instead of the red zone.
;
; Three carry chains run side by side. Each keeps its carry as 0/1 in a byte register between
; uses: `add r8b, 0FFh` loads it into CF (0FFh + 1 carries out), `setc r8b` stores it back.
; Blocks of four limbs amortize that over four adc's per chain. Register use:
;   rdi  p = rp + n + i        (H0 at [p], Li at [p+n8], Hi at [p+2*n8], L0 at [p+m])
;   rsi  n8 = 8n,  rbp  m = -8n,  rdx  limb count of the current phase
;   r8..r11   X (then X + L0),  r12..r15  X + Hi
;   al  cX,  bl  cL,  cl  cH

OPTION CASEMAP:NONE

PUBLIC numetron_karatsuba_interp

.code

ALIGN 16
numetron_karatsuba_interp PROC FRAME
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

    mov     rdi, rcx                    ; rp
    mov     rsi, rdx                    ; n
    mov     rdx, r8                     ; h
    mov     rcx, rsi
    sub     rcx, rdx
    mov     QWORD PTR [rsp + 72], rcx   ; n - h (home space: 8 pushes + return address)
    lea     rdi, [rdi + rsi*8]          ; p = rp + n
    shl     rsi, 3                      ; n8
    mov     rbp, rsi
    neg     rbp                         ; m = -n8
    xor     eax, eax
    xor     ebx, ebx
    xor     ecx, ecx

; ---- phase A: i in [0, h), Hi present ----
    test    dl, 3
    jz      A_blocks
A_single:
    mov     r8, QWORD PTR [rdi]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]   ; X = H0 + Li
    setc    al
    mov     r12, QWORD PTR [rdi + rsi*2]
    add     cl, 0FFh
    adc     r12, r8                     ; X + Hi
    setc    cl
    add     bl, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]   ; X + L0
    setc    bl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + rsi], r12
    lea     rdi, [rdi + 8]
    dec     rdx
    test    dl, 3
    jnz     A_single
A_blocks:
    test    rdx, rdx
    jz      B_start
A_loop:
    mov     r8, QWORD PTR [rdi]
    mov     r9, QWORD PTR [rdi + 8]
    mov     r10, QWORD PTR [rdi + 16]
    mov     r11, QWORD PTR [rdi + 24]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]
    adc     r9, QWORD PTR [rdi + rsi + 8]
    adc     r10, QWORD PTR [rdi + rsi + 16]
    adc     r11, QWORD PTR [rdi + rsi + 24]
    setc    al
    mov     r12, QWORD PTR [rdi + rsi*2]
    mov     r13, QWORD PTR [rdi + rsi*2 + 8]
    mov     r14, QWORD PTR [rdi + rsi*2 + 16]
    mov     r15, QWORD PTR [rdi + rsi*2 + 24]
    add     cl, 0FFh
    adc     r12, r8
    adc     r13, r9
    adc     r14, r10
    adc     r15, r11
    setc    cl
    add     bl, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]
    adc     r9, QWORD PTR [rdi + rbp + 8]
    adc     r10, QWORD PTR [rdi + rbp + 16]
    adc     r11, QWORD PTR [rdi + rbp + 24]
    setc    bl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + 8], r9
    mov     QWORD PTR [rdi + 16], r10
    mov     QWORD PTR [rdi + 24], r11
    mov     QWORD PTR [rdi + rsi], r12
    mov     QWORD PTR [rdi + rsi + 8], r13
    mov     QWORD PTR [rdi + rsi + 16], r14
    mov     QWORD PTR [rdi + rsi + 24], r15
    lea     rdi, [rdi + 32]
    sub     rdx, 4
    jnz     A_loop

; ---- phase B: i in [h, n), Hi = 0 ----
B_start:
    mov     rdx, QWORD PTR [rsp + 72]
    test    dl, 3
    jz      B_blocks
B_single:
    mov     r8, QWORD PTR [rdi]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]
    setc    al
    mov     r12, r8
    add     cl, 0FFh
    adc     r12, 0
    setc    cl
    add     bl, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]
    setc    bl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + rsi], r12
    lea     rdi, [rdi + 8]
    dec     rdx
    test    dl, 3
    jnz     B_single
B_blocks:
    test    rdx, rdx
    jz      done
B_loop:
    mov     r8, QWORD PTR [rdi]
    mov     r9, QWORD PTR [rdi + 8]
    mov     r10, QWORD PTR [rdi + 16]
    mov     r11, QWORD PTR [rdi + 24]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]
    adc     r9, QWORD PTR [rdi + rsi + 8]
    adc     r10, QWORD PTR [rdi + rsi + 16]
    adc     r11, QWORD PTR [rdi + rsi + 24]
    setc    al
    mov     r12, r8
    mov     r13, r9
    mov     r14, r10
    mov     r15, r11
    add     cl, 0FFh
    adc     r12, 0
    adc     r13, 0
    adc     r14, 0
    adc     r15, 0
    setc    cl
    add     bl, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]
    adc     r9, QWORD PTR [rdi + rbp + 8]
    adc     r10, QWORD PTR [rdi + rbp + 16]
    adc     r11, QWORD PTR [rdi + rbp + 24]
    setc    bl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + 8], r9
    mov     QWORD PTR [rdi + 16], r10
    mov     QWORD PTR [rdi + 24], r11
    mov     QWORD PTR [rdi + rsi], r12
    mov     QWORD PTR [rdi + rsi + 8], r13
    mov     QWORD PTR [rdi + rsi + 16], r14
    mov     QWORD PTR [rdi + rsi + 24], r15
    lea     rdi, [rdi + 32]
    sub     rdx, 4
    jnz     B_loop

done:
    movzx   eax, al
    movzx   ebx, bl
    movzx   ecx, cl
    add     ecx, eax                    ; cX + cH
    add     eax, ebx                    ; cX + cL
    shl     rcx, 32
    or      rax, rcx

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rdi
    pop     rbp
    pop     rbx
    ret
numetron_karatsuba_interp ENDP

END
